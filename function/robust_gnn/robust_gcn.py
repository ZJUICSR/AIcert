"""
Implementation of the paper:
'Certifiable Robustness and Robust Training for Graph Convolutional Networks'
by Daniel Zügner, Stephan Günnemann
Published at KDD 2019 in Anchorage, USA.
Copyright (C) 2019
Daniel Zügner
Technical University of Munich
"""

import torch
from torch import nn
import scipy.sparse as sp
import numpy as np
from torch.nn import functional as F
from torch.nn.functional import relu
from torch import optim

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def preprocess_adj(adj):
    """
    Symmetric preprocessing of the input adjacency matrix
    for GCN according to [Kipf and Welling 2017]
    Parameters
    ----------
    adj: sp.spmatrix
        The input adjacency matrix

    Returns
    -------
    adj_norm: sp.spmatrix
        The normalized adjacency matrix.
    """
    N = adj.shape[0]
    adj_tilde = adj + sp.eye(N)
    degs_inv = np.power(adj_tilde.sum(0), -0.5)
    return adj_tilde.multiply(degs_inv).multiply(degs_inv.T)

def sparse_tensor(spmat: sp.spmatrix, grad:bool = False):
    """
    Convert a scipy.sparse matrix to a torch.SparseTensor.
    Parameters
    ----------
    spmat: sp.spmatrix
        The input (sparse) matrix.
    grad: bool
        Whether the resulting tensor should have "requires_grad".

    Returns
    -------
    sparse_tensor: torch.SparseTensor
        The output sparse tensor.

    """
    if str(spmat.dtype) == "float32":
        dtype = torch.float32
    elif str(spmat.dtype) == "float64":
        dtype = torch.float64
    elif str(spmat.dtype) == "int32":
        dtype = torch.int32
    elif str(spmat.dtype) == "int64":
        dtype = torch.int64
    elif str(spmat.dtype) == "bool":
        dtype = torch.uint8
    else:
        dtype = torch.float32
    return torch.sparse_coo_tensor(spmat.nonzero(), spmat.data, size=spmat.shape,
                                                dtype=dtype, requires_grad=grad).coalesce()


class RobustGCNLayer(nn.Module):
    """
    GCN layer that works as a normal layer in the forward pass
    but also provides a backward pass through the dual network.
    """

    def __init__(self, adj: sp.spmatrix, dims):
        super().__init__()
        self.weights = nn.Parameter(nn.init.xavier_normal_(torch.zeros(dims, device="cuda")))
        self.bias = nn.Parameter(nn.init.normal_(torch.zeros(dims[1], device="cuda")))
        self.D, self.H = dims
        if adj.format != "csr":
            adj = adj.tocsr()

        self.adj = adj.astype("float32")
        self.adj_tensor = sparse_tensor(self.adj).cuda()

    def slice_adj(self, nodes):
        """
        Slice the adjacency matrix to contain as rows the input nodes
        and as columns the set of neighbors of all input nodes.

        Parameters
        ----------
        nodes:  numpy.array, int64 dim [Num L-l-1 hop neighbors,]
            Input nodes. If None, all nodes are used.
        Returns
        -------
        adj_slice: torch.sparse_tensor float32 [len(nodes), len(Neighbors of input nodes)]
            The sliced adjacency matrix.
        nbs: np.array, int64 dim [len(Neighbors of input nodes)]
            The set of neighbors of all input nodes.
        """
        nbs = np.unique(self.adj[nodes].nonzero()[1])
        adj_slice = sparse_tensor(self.adj[nodes][:, nbs]).cuda()
        return adj_slice, nbs

    def forward(self, input, nodes=None, slice_input=False):
        if nodes is not None:
            adj_slice, nbs = self.slice_adj(nodes)
            if slice_input:
                input = input[nbs]
        else:
            adj_slice = self.adj_tensor

        return adj_slice.mm(input.mm(self.weights)) + self.bias

    def bounds_continuous(self, input_lower, input_upper, nodes, slice_input=False):
        """
        Compute lower and upper bounds of the hidden activations of the
        current layer l given continuous inputs. This is used for all hidden
        layers except for the first one (where we can use the more accurate
        bounds_binary function).

        Parameters
        ----------
        input_lower: torch.tensor float32 dim [batch, K, Num. L-l+1 hop neighbors, H_{l-1}]
            Lower bounds on the input features.
        input_upper: torch.tensor float32 dim [batch, K, Num. L-l+1 hop neighbors, H_{l-1}]
            Upper bounds on the input features
        nodes:  numpy.array, int64 dim [Num l hop neighbors,]
            L-l hop neighbors of the target nodes.
        slice_input: bool
            Whether the input is the whole attribute matrix. If True, we slice the
            node features accordingly.

        Returns
        -------
        lower_bound: torch.tensor float32 dim [batch, K, Num L-l hop neighbors, H_{l}]
            The lower bounds of the activations in the current layer l.
        upper_bound: torch.tensor float32 dim [batch, K, Num L-l hop neighbors, H_{l}]
            The upper bounds of the activations in the current layer l.
        """
        W_plus = F.relu(self.weights)
        W_minus = F.relu(-self.weights)
        adj_slice, nbs = self.slice_adj(nodes)
        if slice_input:
            input_lower = input_lower[nbs]
            input_upper = input_upper[nbs]

        lower_bound = adj_slice @ (input_lower @ W_plus - input_upper @ W_minus) + self.bias
        upper_bound = adj_slice @ (input_upper @ W_plus - input_lower @ W_minus) + self.bias

        return lower_bound, upper_bound

    def bounds_binary(self, input: torch.tensor, nodes,
                      q:int, Q:int, lower_bound=True, slice_input=False):
        """
        Compute bounds on the first layer for binary node attributes.
        Parameters
        ----------
        input: torch.tensor (boolean) dimension [Num. L-1 hop neighbors, D]
            binary node attributes (one vector for all neighbors of the input nodes)
            OR: [N, D] for the whole graph when slice_input=True
        nodes:  numpy.array, int64 dim [Num l hop neighbors,]
            L-l hop neighbors of the target nodes.
        q:  int
            per-node constraint on the number of attribute perturbations
        Q:  int
            global constraint on the number of attribute perturbations
        lower_bound: bool
            Whether to compute the lower bounds (True) or upper bounds (False)
        slice_input: bool
            Whether the input is the whole attribute matrix. If True, we slice the
            node features accordingly.

        Returns
        -------
        bounds: torch.tensor (float32) dimension [Num. L-2 hop neighbors x H_2]
            Lower/upper bounds on the hidden activations in the second layer.
        """

        # Convention:
        # N: number of nodes in the current layer
        # N_nbs: number of neighbors of the nodes in the current layer
        # D: dimension of the node attributes (i.e. H_1)
        # H: dimension of the first hidden layer (i.e. H_2)

        adj_slice, nbs = self.slice_adj(nodes)
        N_nbs = len(nbs)
        N = len(nodes)
        if slice_input:
            input = input[nbs]

        # [N_nbs x D] => [N_nbs x D x 1]
        input_extended = input.unsqueeze(2)

        # get the positive and negative parts of the weights
        # [D x H]  => [1 x D X H]
        W_plus = F.relu(self.weights).unsqueeze(0)
        W_minus = F.relu(-self.weights).unsqueeze(0)

        # [N_nbs x D x H]
        if lower_bound:
            bounds_nbs = input_extended.mul(W_plus) + (1 - input_extended).mul(W_minus)
        else:
            bounds_nbs = (1 - input_extended).mul(W_plus) + input_extended.mul(W_minus)
        # top q entries per dimension in D
        # => [N_nbs x q x H]
        top_q_vals = bounds_nbs.topk(q, 1)[0]

        # => [N_nbs x q*H]
        top_q_vals = top_q_vals.reshape([N_nbs, -1])

        # [N x N_nbs x 1]
        adj_extended = adj_slice.unsqueeze(2).to_dense()

        # per-node bounds (after aggregating the neighbors)
        # [N x N_nbs x q x H]
        aggregated = adj_extended.mul(top_q_vals).reshape([N, N_nbs, q, -1])

        # sum of the top Q values of the top q values per dimension
        # [N, Q, H] => [N, H]
        n_sel = min(Q, N_nbs * q)
        top_Q_vals = aggregated.reshape([N, -1, self.H]).topk(k=n_sel, dim=1)[0].sum(1)

        if lower_bound:
            top_Q_vals *= -1

        # Add the normal hidden activations for the input
        bounds = top_Q_vals + self.forward(input, nodes)
        return bounds

    def phi_backward(self, phi, nodes, is_last=False):
        """
        Compute Phi_hat of layer l based on Phi of layer l+1.
        This is implemented in a batch manner for all classes at once.

        Parameters
        ----------
        phi: torch.tensor, float32 dim [batch, K, Num. L-l-1 hop neighbors, H_{l+1}]
            The Phi variable of the layer above, denoted as l+1. See below for the
            case of the last (i.e. output) layer.
        nodes:  numpy.array, int64 dim [Num L-l-1 hop neighbors,]
            L-l-1 hop neighbors of the target nodes.
        is_last: bool
            Whether this is the final (output) layer of the GCN. In this case
            phi has dimension [batch, K, K] and so we have to perform the computations
            a bit differently.

        Returns
        -------
        phi_hat: torch.tensor, float32 dim [batch, K, Num. L-l hop neighbors, H_{l}]
            The Phi_hat variable of the current layer, i.e. layer l.
        """

        # slice adjacency matrix to [Num. L-l-1 hop neighbors, Num L-l hop neighbors]
        adj_slice, nbs = self.slice_adj(nodes)
        if is_last:
            phi_hat_right = torch.einsum("ij,ilm->iljm", adj_slice.to_dense(), phi)
        else:
            phi_hat_right = torch.einsum("ijkl,km->ijml", phi, adj_slice.to_dense())

        # [batch, K, Num L-l hop neighbors, H_{l}] => [., ., ., H_{l-1}]
        phi_hat = torch.tensordot(phi_hat_right, self.weights, dims=((3,), (1,)))
        return phi_hat

    def dual_backward(self, phi, nodes, bounds, is_last=False, compute_objective=True,
                      omega=None):
        """
        Perform a backward step through the "dual network". Current layer is denotet
        as l, the layer above as l+1.

        Parameters
        ----------
        phi: torch.tensor, float32 dim [batch, K, Num. L-l-1 hop neighbors, H_{l+1}]
            The Phi variable of the layer above, denoted as l+1. See below for the
            case of the last (i.e. output) layer.
        nodes:  numpy.array, int64 dim [Num L-l-1 hop neighbors]
            L-l-1 hop neighbors of the target nodes.
        bounds: tuple (lower_bounds, upper_bounds), each of shape [Num. L-l hop neighbors, H_{l}]
            The lower and upper bounds of the activations at the previous layer.
        is_last: bool
            Whether this is the final (i.e. output layer).
        compute_objective: bool
            Whether the term in the final objective g(.) should be computed.
            This has to be done for layers 2...L-1 (see equation in the paper).

        Returns
        -------
        next_phi: torch.tensor float32 dim [batch, K, Num. L-l hop neighbors, H_{l}]
            Phi l.
        bias_term: torch.tensor float32 dim [batch, K]
            The term in the objective function g(.) involving the bias of the
            current layer and phi of the layer above.
        objective_term: torch.tensor float32 dim [batch, K] or None
            The term in the objective function g(.) involving [phi_hat]_+ and the
            upper/lower bounds.
        """

        # upper and lower bounds
        lb, ub = bounds

        # Compute phi hat, see section 4.3
        phi_hat = self.phi_backward(phi, nodes, is_last)
        phi_hat_plus = relu(phi_hat)
        phi_hat_minus = relu(-phi_hat)

        if omega is None:
            omega = ub / (ub - lb + 1e-9)
        else:
            omega = torch.clamp(omega, 0, 1)

        # consider the cases where the upper and lower bounds have different signs
        I = ((lb < 0) & (ub > 0)).float()
        I_plus = ((lb > 0) & (ub > 0)).float()

        phi_left = phi_hat.mul(I_plus)
        phi_right_1 = phi_hat_plus.mul(ub / (ub - lb + 1e-9))
        phi_right_2 = phi_hat_minus.mul(omega)
        phi_right = (phi_right_1 - phi_right_2).mul(I)

        # Phi l
        next_phi = phi_left + phi_right
        if compute_objective:
            final_objective_term = phi_hat_plus.mul(ub.mul(lb)/(ub - lb + 1e-9)).mul(I).sum((-2,-1))
        else:
            final_objective_term = None

        return next_phi, self.bias_objective_term(phi), final_objective_term
    
    def bias_objective_term(self, phi):
        """
        Dot product of phi l+1 with the current layer's bias term, summed
        over all nodes in the neighborhood.

        Parameters
        ----------
        phi: torch.tensor, float32 dim [batch, K, Num. L-l-1 hop neighbors, H_{l+1}]
            The Phi variable of the layer above, denoted as l.

        Returns
        -------
        bias_term: torch.tensor float32 dim [batch, K]
            Dot product of phi l+1 with the current layer's bias term, summed
            over all nodes in the neighborhood.
        """
        return (phi @ self.bias.unsqueeze(1)).sum(2).squeeze()


class RobustGCNModel(nn.Module):
    """
    GCN model that works as a normal one in the forward pass
    but also enables robustness certification and robust training
    via the backward pass through the dual network.
    """

    def __init__(self, adj, dims):
        super(RobustGCNModel, self).__init__()
        adj_prep = preprocess_adj(adj).tocsr()
        self.adj_norm = adj_prep
        self.layers = []
        self.dims = dims
        self.K = int(dims[-1])
        self.N = self.adj_norm.shape[0]

        self.omegas = []
        previous = dims[0]  # data dimension
        for ix,hidden in enumerate(dims[1:]):
            self.layers.append(RobustGCNLayer(self.adj_norm, [previous, hidden]))
            self.add_module(f"conv:{ix}", self.layers[-1])
            previous = hidden
            if ix + 2 < len(dims):
                self.omegas.append(torch.zeros([self.N, dims[ix+1]], requires_grad=True))

    def predict(self, input, nodes):
        """
        Predict the classes of the input nodes

        Parameters
        ----------
        input: torch.tensor, float32, [N, D]
            The node attributes (of all nodes in the graph)
        nodes: np.array, int64, dim [B,]
            The batch of nodes to predict the classes for.

        Returns
        -------
        predicted_classes: torch.tensor int64 dim [B,]
            The predicted classes of the input nodes.

        """
        return self.forward(input, nodes, ).max(-1)[1].detach().cpu()

    def get_neighbors(self, nodes):
        """
        Get the neighbors of the input nodes in the given graph (self-loops included).

        Parameters
        ----------
        nodes:  numpy.array, int64 dim [Num l hop neighbors,]
            Input nodes

        Returns
        -------
        neighbors: np.array
            The set of all neighbors of the input nodes.
        """
        return np.unique(self.adj_norm[nodes].nonzero()[1])


    def get_neighborhoods(self, nodes):
        """
        Get the increasing hop neighbors of the input nodes up to the
        L-2 hop neighbors, e.g. the 1-hop neighbors for a GCN with
        L=3, i.e. one hidden layer.
        Parameters
        ----------
        nodes:  numpy.array, int64
            Input nodes

        Returns
        -------
        neighborhoods: list of np.arrays
            List of the sets of all X-hop neighbors of the input nodes (see above).
        """
        neighborhoods = []
        for ix, layer in enumerate(self.layers[::-1]):
            if ix == 0:
                neighborhoods.append(nodes)
                continue
            neighborhoods.append(self.get_neighbors(neighborhoods[-1]))
        return neighborhoods

    def forward(self, input, nodes=None):
        """
        Forward computation of the GCN, computes the logits for the
        input nodes. Slices the adjacency matrices accordingly in all
        the hidden layers so that only the necessary hidden activations
        are computed.
        Parameters
        ----------
        input: torch.tensor binary, shape [N, D]
            The binary node attributs of the graph (can be sparse
            only if all nodes are used).
        nodes:  numpy.array, int64
            Input nodes. If None, we use all nodes in the graph.

        Returns
        -------
        logits: torch.tensor float32 dim [len(nodes), K]
            The log probabilities of the input nodes.

        """

        # get neighborhoods first, in a backward manner
        if nodes is not None:
            neighborhoods = self.get_neighborhoods(nodes)
            neighborhoods = neighborhoods[::-1]

        hidden = input
        for ix,layer in enumerate(self.layers):
            # We only slice the attributes in the input layer,
            # afterwards we just pass them along.
            slice_input = ix == 0
            if nodes is None:
                hidden = layer(hidden)
            else:
                hidden = layer(hidden, neighborhoods[ix], slice_input=slice_input)

            if ix != len(self.layers) - 1:
                hidden = relu(hidden)
        return hidden

    def dual_backward(self, input, nodes, q, Q, target_classes=None,
                      initialize_omega=False, optimize_omega=False,
                      return_perturbations=False):
        """
        Backward computation through the "dual network" to get lower bounds
        on the worst-case logit margins achievable given the provided local
        and global constraints on the perturbations.

        Parameters
        ----------
        input: torch.tensor float32 or int, dim [N, D]
            The binary node attributes.

        nodes: numpy.array, int64
            The input nodes for which to compute the worst-case margins.
        q:  int
            per-node constraint on the number of attribute perturbations
        Q:  int
            global constraint on the number of attribute perturbations
        target_classes: torch.tensor, int64, dim [B,] or None
            The target classes of the nodes in the batch. For nodes in the training set,
            this should be the correct (known) class. For the unlabeled nodes, this should be
            the predicted class given the current weights.
        initialize_omega: bool
            Whether the omega matrices should be initialized to their default value,
            which is upper_bound/(upper_bound-lower_bound). This is only relevant for
            robustness certification (not for robust training, which always uses
            the default values for omega).

        Returns
        -------
        worst_case_bounds: torch.tensor float32, dim [len(nodes), K]
            Lower bounds on the worst-case logit margins achievable given the input constraints.
            A negative worst-case logit margin lower bound means that we cannot certify robustness.
            A positive worst-case logit margin lower bound guarantees that the prediction will not
            change, i.e. we can issue a robustness certificate if for a node ALL worst-case
            logit margins are positive.
        """

        if not (torch.sort(input.unique().long().cpu())[0] == torch.tensor([0,1])).all():
            raise ValueError("Node attributes must be binary.")

        input = input.float()

        # compute upper/lower bounds first
        batch_size = len(nodes)
        bounds = []
        neighborhoods = self.get_neighborhoods(nodes)[::-1]
        for ix, layer in enumerate(self.layers[:-1]):
            layer = self.layers[ix]
            nbh = neighborhoods[ix]
            if ix == 0:
                lower_bound = layer.bounds_binary(input, nbh, q, Q, slice_input=True,
                                                  lower_bound=True)
                upper_bound = layer.bounds_binary(input, nbh, q, Q, slice_input=True,
                                                  lower_bound=False)
                bounds.append((lower_bound, upper_bound))
            else:
                bounds.append(tuple(layer.bounds_continuous(bounds[-1][0], bounds[-1][1],
                                                      nbh, slice_input=False)))

        if target_classes is None:
            # if no target classes are supplied, we use the current predictions
            target_classes = self.predict(input, nodes, )

        predicted_onehot = torch.eye(self.K)[target_classes]
        # [Batch, K, K]
        C_tensor = (predicted_onehot.unsqueeze(1) - torch.eye(self.K)).cuda()
        phis = [-C_tensor]

        # final_objective = torch.zeros([batch_size, self.K], device="cuda")
        bias_terms = torch.zeros([batch_size, self.K], device="cuda")
        I_terms = torch.zeros([batch_size, self.K], device="cuda")

        for ix in np.arange(1,len(self.layers))[::-1]:
            layer = self.layers[ix]
            phi = phis[-1]
            nodes = neighborhoods[ix]

            compute_objective = ix > 0
            is_last_layer = ix == len(self.layers) - 1

            if optimize_omega:
                if initialize_omega:
                    lb, ub = bounds[ix - 1]
                    I = ((ub>0) & (lb < 0)).float()
                    omega = (ub / (ub-lb + 1e-9)).mul(I).clone().detach().requires_grad_(True)
                    with torch.no_grad():
                        self.omegas[ix-1].index_put_((torch.LongTensor(neighborhoods[ix-1]),),
                                                     omega.cpu())
                omega = self.omegas[ix - 1][neighborhoods[ix - 1]].cuda()
            else:
                omega = None
            ret = layer.dual_backward(phi, nodes, bounds[ix - 1], is_last_layer,
                                      compute_objective, omega=omega)

            next_phi, bias_term, objective_term = ret
            phis.append(next_phi)
            bias_terms += bias_term
            if objective_term is not None:
                I_terms += objective_term

        # get the L-2 hop neighbors of the target nodes
        # e.g. the 1-hop neighbors for a 3-layer GCN (i.e. one hidden layer)
        nodes_first = neighborhoods[0]
        phi_1_hat = self.layers[0].phi_backward(phis[-1], nodes=nodes_first)
        nbs_first = self.get_neighbors(nodes_first)

        # sum up the bias terms of the layers
        # [B, K]
        bias_terms += self.layers[0].bias_objective_term(phis[-1])

        # [B, K, Num. L-1 hop neighbors, D]
        Delta = relu(phi_1_hat).mul(1 - input[nbs_first]) \
              + relu(-phi_1_hat).mul(input[nbs_first])
        # [B, K, Num. L-1 hop neighbors, q]
        q_largest_local, q_ixs = Delta.topk(q, dim=3)

        # [B, K, (Num L-1 hop neighbors) * q]
        q_largest_overall = q_largest_local.reshape([batch_size, self.K, -1])
        # Select the Q-th largest element of the q-th largest elements
        # [B, K, Q]
        n_sel = min(Q, len(nbs_first)*q)
        Q_largest, Q_ixs = q_largest_overall.topk(n_sel, -1)
        # [B, K, 1]
        rho = Q_largest[:, :, -1].unsqueeze(-1)

        # Indices of the perturbations
        if return_perturbations:
            q_ixs_reshape = q_ixs.reshape(batch_size, self.K, -1)
            pert_node_ixs = nbs_first[(Q_ixs // q).cpu()]
            pert_dim_ixs = q_ixs_reshape.gather(-1, Q_ixs).cpu().numpy()
            perturbation_ixs = np.stack([pert_node_ixs, pert_dim_ixs], axis=-1)

        # Select the smallest of the q largest values per node,
        # or 0 if it is smaller than rho.
        # [B, K, Num L-1 hop neighbors]
        eta = relu(q_largest_local[:, :, :, -1] - rho)
        # Compute Psi (c.f. the paper) and sum over it
        # [B, K]
        Psi_term = relu(Delta - (rho + eta).unsqueeze(-1)).abs().sum((2, 3))
        # [B, K]
        trace_term = input[nbs_first].mul(phi_1_hat).sum((2, 3))

        # [B, K] lower-bound worst-case margins w.r.t. all other classes
        final_objective = I_terms - bias_terms  - trace_term  - Psi_term - q*eta.sum(-1) - Q*rho.squeeze(-1)

        if return_perturbations:
            return final_objective, perturbation_ixs

        return final_objective


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def train(gcn_model, X, y, idx_train, idx_unlabeled, q, Q=12, n_iters=1000, method="Robust Hinge U", burn_in=100,
          margin_train=np.log(90/10), margin_unlabeled=np.log(60/40), batch_size=8,
          margin_iters=5, learning_rate=1e-2, weight_decay=1e-5):
    """
    Train a (robust) GCN.

    Parameters
    ----------
    gcn_model: RobustGCNModel
        The model to be trained.
    X: torch.tensor  [N, D]
        The node attributes. Should be binary for robust training.
    y: torch.tensor [N,]
        The node labels.
    idx_train: np.array, int64
        The indices of the labeled (training) nodes.
    idx_unlabeled: np.array, int64
        The indices of the unlabeled (test) nodes.
    q: int
        The number of allowed perturbations per node.
    Q: int
        The number of allowed perturbations globally.
    n_iters: int
        The number of training iterations.
    method: str in ["Normal", "Robust Hinge", "Robust Hinge U", "Robust Cross Entropy"]
        The training method used. See our paper for specifics.
    burn_in: int
        Number of iterations in the beginning for which the robust loss is not optimized.
    margin_train: float, default log(90/10)
        The classification margin to optimize for the training nodes.
    margin_unlabeled: float, default log(60/40)
        The classification margin to optimize for the unlabeled nodes.
    batch_size: int
        The batch size. We simulate larger batches by storing the gradients for
        `margin_iters` iterations before updating.
    margin_iters: int
        The number of iterations to store the gradients of the robust loss before updating
        the weights to simulate larger batch sizes.
    learning_rate: float
        The learning rate used for training.
    weight_decay: float
        The weight decay used for regularization.

    Returns
    -------
    None
    """

    implemented_methods = ["Normal", "Robust Hinge", "Robust Hinge U", "Robust Cross Entropy"]
    if method not in implemented_methods:
        raise NotImplementedError(f"Method not in {implemented_methods}.")

    attrs_dense = X.to_dense()
    params = list(gcn_model.parameters())
    weights = [p for p in params if p.requires_grad and len(p.shape) == 2]
    biases = [p for p in params if p.requires_grad and len(p.shape) == 1]

    param_list = [{'params': weights, "weight_decay": weight_decay},
                  {'params': biases, "weight_decay": 0.}]
    opt = optim.Adam(param_list, lr=learning_rate)

    if "Robust Hinge" in method:
        margins = np.ones(gcn_model.N)
        margins[idx_train] = margin_train
        margins[idx_unlabeled] = margin_unlabeled
    elif method == "Robust Cross Entropy":
        ce_loss = nn.CrossEntropyLoss()

    tq = tqdm(range(n_iters))
    for it in tq:
        gcn_model.train()

        if it > burn_in and method != "Normal":
            for inner in range(margin_iters):

                if method == "Robust Hinge U":
                    batch = np.random.randint(0, gcn_model.N, batch_size)
                else:
                    batch = np.random.choice(idx_train, batch_size)

                is_training_node = np.isin(batch, idx_train)
                predicted = gcn_model.predict(attrs_dense, batch).numpy()
                predicted[is_training_node] = y[batch[is_training_node]].cpu()
                predicted_onehot = torch.eye(gcn_model.K)[predicted]

                lower_bound_margins = gcn_model.dual_backward(attrs_dense, batch, q, Q,
                                                              target_classes=predicted,
                                                              initialize_omega=True)

                if "Robust Hinge" in method:
                    margins_tensor = torch.tensor(margins[batch], dtype=torch.float).unsqueeze(1)
                    margin_term = (1 - predicted_onehot).mul(margins_tensor)
                    lower_bound_margins = relu(-lower_bound_margins + margin_term.cuda())
                    robust_loss = (lower_bound_margins.mean()) / margin_iters
                elif method == "Robust Cross Entropy":
                    robust_loss = ce_loss(-lower_bound_margins, torch.LongTensor(predicted).cuda())/margin_iters
                with torch.no_grad():
                    robust_loss.backward()

        if it <= burn_in or method in ["Robust Hinge", "Robust Hinge U", "Normal"]:
            logits = gcn_model(X)
            classification_loss = nn.functional.cross_entropy(logits[idx_train], target=y[idx_train])
            with torch.no_grad():
                classification_loss.backward()

        opt.step()
        opt.zero_grad()


def certify(gcn_model, attrs, q, nodes=None, Q=12, optimize_omega=False, optimize_steps=5, batch_size=8,
           certify_nonrobustness=False, progress=False):
    """
    Certify (non-) robustness of the input nodes given the input GCN and attributes.

    Parameters
    ----------
    gcn_model: RobustGCNModel
        The input neural network.
    attrs: sp.spmatrix, [N, D]
        The binary node attributes.
    q: int
        The number of allowed perturbations per node.
    nodes: np.array, int64
        The input node indices to compute certificates for.
    Q: int
        The number of allowed perturbations globally.
    optimize_omega: bool, default False
        Whether to optimize (True) over Omega or to use the default value (False).
        If True, optimization takes significantly longer but will lead to more certificates.
        False positives (i.e. falsely issued certificates) are never possible.
    optimize_steps: int
        The number of steps to optimize Omega for. Ignored if optimize_omega is False.
    batch_size: int
        The batch size to use. Larger means faster computation but requires more GPU memory.
    certify_nonrobustness: bool, default: False
        Whether to also certify non-robustness. This works by determining the optimal perturbation
        for the relaxed GCN and feeding it into the original GCN. If this perturbation changes the
        predicted class, we have certified non-robustness via an example.
    progress: bool, default: False
        Whether to display a progress bar using the package `tqdm`. If it is not installed,
        we silently ignore this parameter.

    Returns
    -------
    robust_nodes: np.array, bool, [N,]
        A boolean flag for each of the input nodes indicating whether a robustness certificate
        can be issued.
    nonrobust_nodes: np.array, bool, [N,]
        A boolean flag for each of the input nodes indicating whether we can prove non-robustness.
        If certify_nonrobustness is False, this contains False for every entry.
    """

    node_attrs = sparse_tensor(attrs).cuda().to_dense()

    N = gcn_model.N
    K = gcn_model.K

    if optimize_omega:
        opt_omega = optim.Adam([{'params': x, "weight_decay": 0} for x in gcn_model.omegas])
    else:
        optimize_steps = 0

    if nodes is None:
        nodes = np.arange(N)

    for step in range(optimize_steps):
        for chunk in chunker(nodes, batch_size):

            obj, pert = gcn_model.dual_backward(X_t.to_dense(), chunk, q, Q,
                                                initialize_omega=(step==0),
                                                optimize_omega=True)

            margin_loss = (-obj.min(1)[0].mean())
            with torch.no_grad():
                margin_loss.backward()
            opt_omega.step()
            opt_omega.zero_grad()

    lower_bounds = []
    import time
    nonrobust_nodes = np.zeros(N)

    _iter = chunker(np.arange(N), batch_size)
    if progress:
        _iter = tqdm(_iter, total=int(np.ceil(N/batch_size)))

    for chunk in _iter:
        lb, pert = gcn_model.dual_backward(node_attrs, chunk, q, Q,
                                           initialize_omega=not optimize_omega,
                                           return_perturbations=True)
        lb = lb.detach()
        lower_bounds.append(lb.cpu().numpy())
        if certify_nonrobustness:
            predicted_before = gcn_model.predict(node_attrs, chunk)
            predicted_after = []
            for ix,node in enumerate(chunk):
                attack_successful = []
                for cl in torch.sort(lb[ix])[1]:
                    if lb[ix,cl] >= 0:
                        # only test for nonrobustness when we cannot certify robustness
                        attack_successful.append(False)
                        continue

                    pert_ixs = tuple(torch.tensor(pert[ix,cl].T))
                    node_attrs.index_put_(pert_ixs, 1-node_attrs[pert_ixs])

                    after = gcn_model.predict(node_attrs, [node])[0].cpu().numpy()
                    success = after != predicted_before[ix].numpy()
                    attack_successful.append(success)
                    node_attrs.index_put_(pert_ixs, 1-node_attrs[pert_ixs])

                    if success:
                        # once we have an adversarial example for a single class we can stop
                        break

                predicted_after.append(np.any(attack_successful))
            predicted_after = np.row_stack(predicted_after)
            nonrobust_nodes[chunk] = predicted_after[:,0]

    lower_bounds = np.row_stack(lower_bounds)
    robust_nodes = ((lower_bounds > 0).sum(1) == K-1)

    return robust_nodes, nonrobust_nodes