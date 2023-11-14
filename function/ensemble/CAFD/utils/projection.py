import math
import torch

# from utils import _violation_nonnegativity, _check_nonnegativity
# from utils import _violation_marginal_constraint, _check_marginal_constraint
# from utils import _violation_transport_cost, _check_transport_cost

# from utils import bgss
def bisection_search(grad_fn, a, b, max_iter, grad_tol, int_tol, verbose=False):
    assert (a < b).all()

    for i in range(max_iter):
        mid = (a + b) / 2
        grad = grad_fn(mid)
        idx = grad > 0.

        a[idx] = mid[idx]
        b[~idx] = mid[~idx]

        assert (a < b).all()

        if grad_tol is not None and (grad.abs() < grad_tol).all():
            break

        if int_tol is not None and torch.max(b - a) < int_tol:
            break

        if verbose:
            print("bisection iter {:2d}, gradient".format(i), grad_fn(mid))

    pnt = True

    if grad_tol is not None and (grad.abs() < grad_tol).all():
        pnt = False

    if int_tol is not None and torch.max(b - a) < int_tol:
        pnt = False

    if grad_tol is None and int_tol is None:
        pnt = False

    if pnt:
        print("WARNING: bisection search does not converge in {:2d} iterations".format(max_iter))

    return b, i + 1

def tensor_norm(tensor, p=2):
    """
    Return the norm for a batch of samples
    Args:
        tensor: tensor of size (batch, channel, img_size, last_dim)
        p: 1, 2 or inf

        if p is inf, the size of tensor can also be (batch, channel, img_size)
    Return:
        tensor of size (batch, )
    """
    assert tensor.layout == torch.strided

    if p == 1:
        return tensor.abs().sum(dim=(1, 2, 3))
    elif p == 2:
        return torch.sqrt(torch.sum(tensor * tensor, dim=(1, 2, 3)))
    elif p == 'inf':
        return tensor.abs().view(tensor.size(0), -1).max(dim=-1)[0]
    else:
        assert 0

def simplex_projection(pi, X):
    """
    Projection to a simplex
    pi: tensor of size (batch_size, channel, img_size, img_size)
                    or (batch_size, channel, img_size, kernel_size^2)
    """
    batch_size, c, h, w = X.size()
    img_size = h * w

    device = X.device

    lst_dim = pi.size(3)

    sorted_pi = torch.sort(pi, dim=-1, descending=True)[0]

    """
    csum is of size (batch_size, channel, img_size, img_size)
                 or (batch_size, channel, img_size, kernel_size^2)
    """
    csum = torch.cumsum(sorted_pi, dim=-1) - X.view(batch_size, c, img_size, 1)
    ind = torch.arange(1, lst_dim + 1, dtype=torch.float, device=device)
    cond = ((ind * sorted_pi - csum) > 0.0).type(torch.float) * ind

    # rho.size = (batch_size, channel, img_size)
    rho = torch.max(cond, dim=-1)[1]

    # theta.size = (batch_size, channel, img_size)
    theta = csum[torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1),
                 torch.arange(c, dtype=torch.long, device=device).view(1, -1, 1),
                 torch.arange(img_size, dtype=torch.long, device=device).view(1, 1, -1),
                 rho] / (1. + rho.type(torch.float))


    return torch.max(pi - theta.view(batch_size, c, img_size, 1),
                     pi.new_zeros((1, 1, 1, 1))
                     )


def dual_projection(G, X, cost, eps, dual_max_iter, grad_tol, int_tol):
    batch_size = G.size(0)

    left = X.new_zeros(batch_size)
    right = 2 * tensor_norm(G, p='inf') + tensor_norm(X, p='inf')

    def recover(G, lam, X, cost, eps):
        pi_star = simplex_projection(G - lam.view(-1, 1, 1, 1) * cost, X)
        return pi_star

    def grad_fn(lam):
        tilde_pi = recover(G, lam, X, cost, eps)
        return (tilde_pi * cost).sum(dim=(1, 2, 3)) - eps

    lam_star, num_iter = bisection_search(grad_fn,
                                          left,
                                          right,
                                          max_iter=dual_max_iter,
                                          grad_tol=grad_tol,
                                          int_tol=int_tol,
                                          verbose=False)

    pi_star = recover(G, lam_star, X, cost, eps)

    return pi_star, num_iter


def subtract_column(pi, mu, transpose_idx, detranspose_idx):
    """
    Assuming that pi is of size (batch, c, img_size, img_size)
                             or (batch, c, img_size, kernel_size^2)
              and mu is of size (batch, c, h, w)
    """
    batch_size, c, h, w = mu.size()
    # img_size = h * w

    if transpose_idx is None:
        pi = pi - mu.view(batch_size, c, 1, h * w)
    else:
        pi = pi.view(batch_size, c, -1)[:, :, transpose_idx]
        pi = pi.view(batch_size, c, h * w, -1)

        pi = pi - mu.view(batch_size, c, h * w, 1)

        pi = pi.view(batch_size, c, -1)[:, :, detranspose_idx]
        pi = pi.view(batch_size, c, h * w, -1)

    return pi


def dual_capacity_constrained_projection(G, X, cost, eps, transpose_idx, detranspose_idx, coupling2adversarial, verbose=False):
    """
    Assuming that G is of size (batch, c, img_size, img_size)
                             or (batch, c, img_size, kernel_size^2)
           and cost is of size (img_size, img_size)
                             or (img_size, kernel_size^2)
    """
    batch_size, c, h, w = X.size()

    lam = X.new_ones(batch_size)
    mu = X.new_ones(X.size())

    mu_y_prev = X.new_zeros(X.size())

    def recover(lam, mu):
        pi_tilde = simplex_projection(subtract_column(G - lam.view(-1, 1, 1, 1) * cost,
                                                      mu,
                                                      transpose_idx,
                                                      detranspose_idx),
                                      X)
        return pi_tilde

    def grad(lam, mu):
        pi_tilde = recover(lam, mu)

        d_lam = (pi_tilde * cost).sum(dim=(1, 2, 3)) - eps
        d_mu = coupling2adversarial(pi_tilde, X) - X.new_ones(X.size())
        return d_lam, d_mu

    def optimize_lam(mu):
        left = X.new_zeros(batch_size)
        right = 2 * tensor_norm(G, p='inf') + 2 * tensor_norm(mu, p='inf') + tensor_norm(X, p='inf')

        lam = bisection_search(lambda x: grad(x, mu)[0],
                               left,
                               right,
                               max_iter=50,
                               grad_tol=1e-4,
                               int_tol=1e-4,
                               verbose=False)[0]
        return lam

    for i in range(5000):
        if i % 20 == 0:
            lam = optimize_lam(mu)

        d_mu = grad(lam, mu)[1]

        if i <= 1000:
            eta = 1e-1
        elif i <= 3000:
            eta = 1e-2
        else:
            eta = 1e-3

        gamma = 0.9

        mu_y = (mu + eta * d_mu).clamp(min=0.)
        mu = (1 + gamma) * mu_y - gamma * mu_y_prev

        mu_y_prev = mu_y

        if verbose and i % 10 == 0:
            print("iter {:5d}".format(i),
                  "d_mu max {:11.8f}".format(d_mu.max().item()),
                  "d_mu min {:11.8f}".format(d_mu.min().item()),
                  )
            print(lam)
            print(mu.max())

    # up-weight lambda slightly to ensure the transportation cost constraint
    # pi_tilde = recover(lam + 1e-4, mu)

    mu = mu_y

    """Run bisection method again to ensure strict feasibility of transportation cost constraint"""
    lam = optimize_lam(mu)

    pi_tilde = recover(lam, mu)

    return pi_tilde


# def dykstra_projection(pi, X, cost, inf, eps, dykstra_max_iter, return_trajectory=True):
#     pi_simp = pi.new_zeros(pi.size())
#     pi_half = pi.new_zeros(pi.size())
#     q_simp = pi.new_zeros(pi.size())
#     q_half = pi.new_zeros(pi.size())

#     pi_half = pi.clone().detach()

#     lst_pi_half = []
#     lst_pi_simp = []

#     for t in range(dykstra_max_iter):
#         tmp_simp = pi_simp.clone()
#         pi_simp = simplex_projection(pi_half - q_simp, X)

#         _check_nonnegativity(pi_simp, tol=1e-4, verbose=False)
#         _check_marginal_constraint(pi_simp, X, tol=1e-4, verbose=False)

#         q_simp = pi_simp - pi_half + q_simp

#         tmp_half = pi_half.clone()
#         pi_half = halfspace_projection(pi_simp - q_half, cost, inf, eps)

#         _check_transport_cost(pi_half, cost, eps, tol=eps * 1e-2, verbose=False)

#         q_half = pi_half - pi_simp + q_half

#         if tensor_norm(pi_simp - pi_half, p=1).max() < 1e-5:
#             break

#         if t % 10 == 0:
#             max_transport_cost = (cost * pi_simp).sum(dim=(1, 2, 3)).max().item()
#             print("dykstra iteration {:d} transport cost {:.6f}".format(t + 1, max_transport_cost))
#             _check_marginal_constraint(pi_half, X, tol=100, verbose=True)
#             print("simp norm : {:.9f}".format(tensor_norm(tmp_simp - pi_simp, p=1).max().item()))
#             print("half norm : {:.9f}".format(tensor_norm(tmp_half - pi_half, p=1).max().item()))
#             print("simp half : {:.9f}".format(tensor_norm(pi_simp - pi_half, p=1).max().item()))


#         if return_trajectory:
#             lst_pi_half.append(pi_half.clone().detach())
#             lst_pi_simp.append(pi_simp.clone().detach())

#     if return_trajectory is False:
#         return pi_half
#     else:
#         return pi_half, lst_pi_half, lst_pi_simp
