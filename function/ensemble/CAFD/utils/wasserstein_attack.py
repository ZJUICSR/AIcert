import torch

from advertorch.attacks import Attack

from .sparse_tensor import initialize_dense_cost
from .sparse_tensor import initialize_sparse_cost
from .sparse_tensor import initialize_sparse_coupling


def _violation_nonnegativity(pi):
    diff = pi.clamp(max=0.).abs().sum(dim=(1, 2, 3)).max().item()
    return diff


def _check_nonnegativity(pi, tol, verbose=False):
    """pi: tensor of size (batch_size, c, img_size, img_size)"""
    # diff = pi.clamp(max=0.).abs().sum(dim=(1, 2, 3)).max().item()
    diff = _violation_nonnegativity(pi)

    if verbose:
        print("check nonnegativity: {:.9f}".format(diff))

    assert diff < tol


def _violation_marginal_constraint(pi, X):
    batch_size, c, h, w = X.size()
    img_size = h * w

    diff = (pi.sum(dim=-1) - X.view(batch_size, c, img_size)).abs().sum(dim=(1, 2)).max().item()

    return diff


def _check_marginal_constraint(pi, X, tol, verbose=False):
    """
    pi: dense tensor of size (batch_size, c, img_size, img_size)
                          or (batch_size, c, img_size, kernel_size^2)
    X: tensor of size (batch_size, c, h, w)
    """
    diff = _violation_marginal_constraint(pi, X)

    if verbose:
        print("check marginal constraint: {:.9f}".format(diff))

    assert diff < tol


def _violation_transport_cost(pi, cost, eps):
    diff = (cost * pi).sum(dim=(1, 2, 3)).max().item()
    return diff


def _check_transport_cost(pi, cost, eps, tol, verbose=False):
    """
    pi: dense tensor of size (batch_size, c, img_size, img_size)
                          or (batch_size, c, img_size, kernel_size^2)
    cost: tensor of size (img_size, img_size)
                      or (img_size, kernel_size^2)
    """
    diff = _violation_transport_cost(pi, cost, eps)

    if verbose:
        print("check transportation cost: {:.9f}".format(diff))

    assert diff < eps + tol



class Coulping2adversarial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pi, size, forward_idx, backward_idx):
        ctx.save_for_backward(pi, forward_idx, backward_idx)
        ctx.size = size

        batch, c, h, w = size

        pi = pi.view(batch, c, -1)[:, :, forward_idx]
        pi = pi.view(batch, c, h * w, -1)

        return pi.sum(dim=3).view(batch, c, h, w)

    @staticmethod
    def backward(ctx, grad_output):
        pi, forward_idx, backward_idx = ctx.saved_tensors
        size = ctx.size

        batch, c, h, w = size

        grad_input = pi.new_zeros(pi.size())
        grad_input += grad_output.view(batch, c, h * w, 1)

        grad_input = grad_input.view(batch, c, -1)[:, :, backward_idx]
        grad_input = grad_input.view(batch, c, h * w, -1)

        return grad_input, None, None, None


class WassersteinAttack(Attack):

    def __init__(self,
                 predict, loss_fn,
                 eps, kernel_size,
                 device,
                 postprocess=False,
                 verbose=True,
                 ):
        """
        Args:
            kernel_size (None or int): None indicates dense cost coupling
        """
        super().__init__(predict, loss_fn, clip_min=0., clip_max=1.)
        self.eps = eps
        self.kernel_size = kernel_size

        self.device = device

        """post-processing parameters"""
        self.postprocess = postprocess

        self.cost = None

        """variables supporting sparse matrices operations"""
        self.cost_indices = None
        self.forward_idx = None
        self.backward_idx = None

        """other parameters"""
        self.verbose = verbose

        """
        parameters for the ease of recording experimental results

        group 1:record (total projection/conjugate running time,
                        total projection/conjugate iterations,
                        total projection/conjugate function calls)
        """
        self.run_time = 0.0
        self.num_iter = 0
        self.func_calls = 0


        """group 2: flags for projected Sinkhorn"""
        self.converge = True
        self.overflow = False


        """group 3: loss and accuracy in each batch"""
        self.lst_loss = []
        self.lst_acc = []

    def initialize_cost(self, X, inf=10000):
        """
        Return a cost matrix of size (img_size, img_size)
                                  or (img_size, kernel_size^2)
        """
        if self.cost is not None:
            return self.cost

        batch_size, c, h, w = X.size()

        if self.kernel_size is None:
            self.cost = initialize_dense_cost(h, w).to(self.device)
        else:
            indices, values = initialize_sparse_cost(h, w, self.kernel_size, inf=inf)

            self.cost_indices = indices.to(self.device)
            self.forward_idx = self.cost_indices[1, :].argsort()
            self.backward_idx = self.forward_idx.argsort()

            self.cost = values.view(h * w, self.kernel_size ** 2).to(self.device)

        return self.cost

    def initialize_coupling(self, X):
        """
        Return a coupling of size (batch_size, channel, img_size, img_size)
                               or (batch_size, channel, img_size, kernel_size^2)
        """
        batch_size, c, h, w = X.size()
        img_size = h * w

        if self.kernel_size is None:
            pi = torch.zeros([batch_size, c, img_size, img_size], dtype=torch.float, device=self.device)
            pi[:, :, range(img_size), range(img_size)] = X.view(batch_size, c, img_size).to(self.device)
        else:
            indices, values = initialize_sparse_coupling(X.to("cpu"), self.kernel_size)
            pi = values.view(batch_size, c, img_size, self.kernel_size ** 2).to(self.device)

        return pi

    def coupling2adversarial(self, pi, X):
        """Return adversarial examples from the coupling"""
        batch_size, c, h, w = X.size()

        if self.kernel_size is None:
            return pi.sum(dim=2).view(batch_size, c, h, w)
        else:
            # return torch.sparse.sum(self.dense2sparse(pi, X), dim=2).to_dense().view(batch_size, c, h, w)
            return Coulping2adversarial.apply(pi, X.size(), self.forward_idx, self.backward_idx)

    def check_nonnegativity(self, pi, tol=1e-4, verbose=False):
        _check_nonnegativity(pi=pi, tol=tol, verbose=verbose)

    def check_marginal_constraint(self, pi, X, tol=1e-4, verbose=False):
        _check_marginal_constraint(pi=pi, X=X, tol=tol, verbose=verbose)

    def check_transport_cost(self, pi, tol=1e-4, verbose=False):
        _check_transport_cost(pi=pi, cost=self.cost, eps=self.eps, tol=tol, verbose=verbose)

    def print_info(self, acc):
        print("accuracy under attack ------- {:.2f}%".format(acc))
        print("total dual running time ----- {:.3f}ms".format(self.run_time))
        print("total number of dual iter --- {:d}".format(self.num_iter))
        print("total number of fcall ------- {:d}".format(self.func_calls))

    def save_info(self, acc, save_info_loc):
        torch.save((acc,
                    self.run_time,
                    self.num_iter,
                    self.func_calls,
                    self.overflow,
                    self.converge,
                    self.lst_loss,
                    self.lst_acc,
                    ),
                   save_info_loc)


def test_cost_initialization():
    dense_attacker = WassersteinAttack(predict=lambda x: x,
                                       loss_fn=lambda x: x,
                                       eps=0.5,
                                       kernel_size=None,
                                       device="cuda")

    sparse_attacker = WassersteinAttack(predict=lambda x: x,
                                        loss_fn=lambda x: x,
                                        eps=0.5,
                                        kernel_size=7,
                                        device="cuda")

    X = torch.zeros((5, 3, 28, 28), dtype=torch.float, device="cuda")

    dense_cost = dense_attacker.initialize_cost(X)

    sparse_cost = sparse_attacker.initialize_cost(X, inf=100)

    full_dense_cost = torch.sparse_coo_tensor(sparse_attacker.cost_indices,
                                              sparse_cost.view(-1),
                                              dtype=torch.float,
                                              device="cuda").to_dense()

    mask = (full_dense_cost > 0) * (full_dense_cost < 100)
    diff = (mask * dense_cost - mask * full_dense_cost).abs().sum()

    print("difference of cost {:f}".format(diff))


def test_coupling_initialization():
    sparse_attacker = WassersteinAttack(predict=lambda x: x,
                                        loss_fn=lambda x: x,
                                        eps=0.5,
                                        kernel_size=7,
                                        device="cuda")

    X = torch.rand((5, 3, 28, 28), dtype=torch.float, device="cuda")

    pi = sparse_attacker.initialize_coupling(X)

    sparse_attacker.check_marginal_constraint(pi, X, tol=1e-6, verbose=True)

    adv = sparse_attacker.coupling2adversarial(pi, X)
    print((adv - X).abs().sum().item())


def gradient_checking():
    sparse_attacker = WassersteinAttack(predict=lambda x: x,
                                        loss_fn=lambda x: x,
                                        eps=0.5,
                                        kernel_size=5,
                                        device="cuda")

    X = torch.randn((2, 3, 28, 28), dtype=torch.float, device="cuda")

    pi = sparse_attacker.initialize_coupling(X).clone().double().requires_grad_(True)
    sparse_attacker.initialize_cost(X, inf=10000)

    input = (pi, X.size(), sparse_attacker.forward_idx, sparse_attacker.backward_idx)

    from torch.autograd import gradcheck
    test = gradcheck(lambda x, y, z, w: Coulping2adversarial.apply(x, y, z, w).sum(), input, eps=1e-6, atol=1e-4)
    print(test)

    loss = (Coulping2adversarial.apply(*input) * torch.randn(X.size(), dtype=torch.float, device="cuda")).sum()
    loss.backward()

if __name__ == "__main__":
    # test_cost_initialization()
    # test_coupling_initialization()
    gradient_checking()
