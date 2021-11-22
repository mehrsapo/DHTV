from odl.operator.tensor_ops import MatrixOperator
import numpy as np
import odl

import pycsou.linop.base as base
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.opt.proxalgs import PrimalDualSplitting

def solve_admm(tri, lmbda, sigma, niter, file=None, random=0):

    L_op = MatrixOperator(tri.L * lmbda)
    H_op = MatrixOperator(tri.H)

    stack_op = odl.BroadcastOperator(H_op, L_op)

    x_values_np = tri.data_values
    data_fit = odl.solvers.L2NormSquared(H_op.range).translated(x_values_np)
    reg_func = odl.solvers.L1Norm(L_op.range)  # lmbda is inside L_op

    g = odl.solvers.SeparableSum(data_fit, reg_func)
    # We don't use the f functional, setting it to zero
    f = odl.solvers.ZeroFunctional(stack_op.domain)

    # --- Select parameters and solve using LADMM --- #
    # Estimated operator norm, add 10 percent for some safety margin
    op_norm = 1.05 * odl.power_method_opnorm(stack_op, maxiter=4000)

    tau = sigma / op_norm ** 2  # Step size for f.proximal
    print(sigma, tau)
    print(f'admm (tau, sigma): ({tau}, {sigma})')

    '''if random:
        print('random', np.mean(x_values_np), np.std(x_values_np)*2)
        z = np.random.normal(np.mean(x_values_np), np.std(x_values_np)*2, size=(tri.n_grid_points,))
    else:'''

    z = tri.grid_values

    rn_space = odl.rn(z.shape[0], dtype='float32')
    z_odl = rn_space.element(z)

    if file is not None:
        callback = (odl.solvers.CallbackPrintIteration(step=2000) &
                    odl.solvers.CallbackShowConvergence(tri.L, tri.H, tri.data_values, lmbda, step=2000, file_name=file))
    else:
        callback = odl.solvers.CallbackPrintIteration(step=2000)

    odl.solvers.admm_linearized(z_odl, f, g, stack_op, tau, sigma, niter, callback=callback)

    z_admm = z_odl.asarray()

    return z_admm


def solve_PDS(tri, lmbda, min_iter, file=None):

    L_operator = base.SparseLinearOperator(tri.L)
    H_operator = base.SparseLinearOperator(tri.H)

    L_operator.compute_lipschitz_cst()
    H_operator.compute_lipschitz_cst()

    l2_loss = SquaredL2Loss(dim=H_operator.shape[0], data= tri.data_values)
    F = l2_loss

    if lmbda == 0:
        H = None
    else:
        H = lmbda * L1Norm(dim=L_operator.shape[0])

    pds = PrimalDualSplitting(dim=H_operator.shape[1], x0=tri.data_values,
                              F=F, H=H, K=L_operator, verbose=2000, min_iter=min_iter, file=file, tri=tri,
                              lambbda=lmbda,values=tri.data_values)
    estimate, converged, diagnostics = pds.iterate()

    z = estimate['primal_variable']

    return z