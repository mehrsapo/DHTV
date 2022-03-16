from math import gamma
from odl.operator.tensor_ops import MatrixOperator
import numpy as np
import odl
from torch._C import device

'''import pycsou.linop.base as base
from pycsou.func.loss import SquaredL2Loss
from pycsou.func.penalty import L1Norm
from pycsou.opt.proxalgs import PrimalDualSplitting'''
import scipy
import torch

import scipy.sparse.linalg
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import identity


def solve_admm(tri, lmbda, sigma, niter, file=None, random=0, x0=None):

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

    if x0 is None:
        z = tri.grid_values 
    else:
        z = x0

    rn_space = odl.rn(z.shape[0], dtype='float32')
    z_odl = rn_space.element(z)

    if file is not None:
        callback = (odl.solvers.CallbackPrintIteration(step=5000) &
                    odl.solvers.CallbackShowConvergence(tri.L, tri.H, tri.data_values, lmbda, step=5000, file_name=file))
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

def fista_for_dual(y, D, s, lmbda, niter):

    DT = D.transpose()
    DDT = D @ DT
    mDy = -D @ y
    alpha = (1 / s) * 0.98
    y_k = np.zeros((D.shape[0]))
    x_k = np.zeros((D.shape[0]))
    t_k = 1
    print(alpha)
    for i in range(niter):
        print(i)
        x_kp1 = np.clip(y_k - alpha * (2 * mDy + 2 * DDT @ y_k), -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        y_kp1 = x_kp1 + ((t_k-1)/t_kp1) * (x_kp1 - x_k)
        print(x_kp1)
        x_k = x_kp1
        t_k = t_kp1
        y_k = y_kp1

        if i % 10 == 0:
            beta = y - DT @ x_kp1
            df_loss = np.sum((beta-y)**2) / 2
            htv_loss = np.linalg.norm(D.dot(beta), 1)
            total_loss = df_loss + lmbda * htv_loss

            print(i, df_loss, htv_loss, total_loss)

    beta = y - DT @ x_kp1
    df_loss = np.sum((beta - y) ** 2) / 2
    htv_loss = np.linalg.norm(D.dot(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    print('FINAL', df_loss, htv_loss, total_loss)
    beta = y - DT @ x_kp1

    return beta

def fista_for_dual_v2(y, D, s, lmbda, niter):

    DT = D.transpose()
    mDy = -D @ y
    alpha = (1 / s) * 0.99
    y_k = np.zeros((D.shape[0]))
    x_k = np.zeros((D.shape[0]))
    t_k = 1
    t_loss_list = list()
    print(alpha)
    for i in range(niter):
        x_kp1 = np.clip(y_k - alpha * (2 * mDy + 2 * D@(DT @ y_k)), -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        y_kp1 = x_kp1 + ((t_k-1)/t_kp1) * (x_kp1 - x_k)
        x_k = x_kp1
        t_k = t_kp1
        y_k = y_kp1

        if i % 1000 == 0:
            beta = y - DT @ x_kp1
            df_loss = np.sum((beta-y)**2) / 2
            htv_loss = np.linalg.norm(D.dot(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
            t_loss_list.append(total_loss)
            print(i, df_loss, htv_loss, total_loss)

    beta = y - DT @ x_kp1
    df_loss = np.sum((beta - y) ** 2) / 2
    htv_loss = np.linalg.norm(D.dot(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    print('FINAL', df_loss, htv_loss, total_loss)
    beta = y - DT @ x_kp1

    return beta, t_loss_list


def fista_for_dual_gpu(y, D, s, lmbda, niter, stop_rel = 1e-5):

    device = torch.device('cuda:1')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)
    alpha = (1 / s) * 0.99
    y_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    t_k = 1
    t_loss_list = list()


    for i in range(niter):
        
        x_kp1 = torch.clip(y_k - alpha * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        y_kp1 = x_kp1 + ((t_k-1)/t_kp1) * (x_kp1 - x_k)
        x_k = x_kp1
        t_k = t_kp1
        y_k = y_kp1

        if i == 0:
            beta = y - DT.matmul(x_kp1)
            df_loss = ((beta-y)**2).sum() / 2
            htv_loss = torch.norm(D.matmul(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
        else:
            t_old = total_loss
            beta = y - DT.matmul(x_kp1)
            df_loss = ((beta-y)**2).sum() / 2
            htv_loss = torch.norm(D.matmul(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
            rel_mod = ((t_old - total_loss)/ t_old) * 100

        if i % 10000 == 9999:
            t_loss_list.append(total_loss)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item(), rel_mod.item())

        if i> 1 and rel_mod.item()>0 and rel_mod.item() < stop_rel:
            break

    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    print('FINAL', i,  df_loss.item(), htv_loss.item(), total_loss.item(), rel_mod.item())
    beta = y - DT @ x_kp1

    return beta, t_loss_list


def fista_for_dual_gpu_v2(y, D, s, lmbda, niter):

    device = torch.device('cuda:0')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)
    alpha = (1 / s) * 0.99
    y_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    t_k = 1
    t_loss_list = list()

    r = 4
    p = 0.99
    q = 5000

    for i in range(niter):
        
        x_kp1 = torch.clip(y_k - alpha * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        t_kp1 = (p + np.sqrt(r*t_k**2+q)) / 2
        y_kp1 = x_kp1 + ((t_k-1)/t_kp1) * (x_kp1 - x_k)
        x_k = x_kp1
        t_k = t_kp1
        y_k = y_kp1

        beta = y - DT.matmul(x_kp1)
        df_loss = ((beta-y)**2).sum() / 2
        htv_loss = torch.norm(D.matmul(beta), 1)
        total_loss = df_loss + lmbda * htv_loss



        if i % 1000 == 0:

            t_loss_list.append(total_loss)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item())

    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    print('FINAL', df_loss.item(), htv_loss.item(), total_loss.item())
    beta = y - DT @ x_kp1

    return beta, t_loss_list


class ADMM_cg:

    def __init__(self, tri, rho, lmbda):

        self.D = tri.L
        self.X = tri.H
        self.Dt = self.D.transpose()
        self.Xt = self.X.transpose()

        self.tri = tri
        self.y = tri.data_values
        self.DtD = self.Dt @ self.D
        self.XtX = self.Xt @ self.X
        self.Xty = self.Xt @ self.y
        self.M = self.XtX + rho * self.DtD

        self.lmbda = lmbda
        self.rho = rho

    def soft_thrsh(self, x, t):
        signs = ((x < -t)*1 - (x > t)*1)
        out = np.abs(signs) * x + t * signs
        return out

    def iterate(self, n_iter=200):
        alpha_k = self.D @ self.tri.grid_values
        w_k = alpha_k * 0

        for iter in range(n_iter):
            if iter > 1: #and iter % 500 == 0:
                print(iter)
                htv_vec = np.abs(self.tri.L @ b_k)
                htv_loss_admm = np.sum(htv_vec)
                df_loss_admm = np.sum((self.tri.H.dot(b_k) - self.y) ** 2) 
                total_loss_admm = df_loss_admm + self.lmbda * htv_loss_admm * 2
                print(df_loss_admm, htv_loss_admm, total_loss_admm)

            b_k = scipy.sparse.linalg.spsolve(self.M, (self.Xty + self.rho * self.Dt @ (alpha_k - w_k)))
            #b_k = scipy.linalg.solve(self.M.todense(), (self.Xty + self.rho * self.Dt @ (alpha_k - w_k)))
            alpha_k = self.soft_thrsh(self.D @ b_k + w_k, self.lmbda / self.rho)
            w_k = w_k + self.D @ b_k - alpha_k

        return b_k


def fista_for_dual_gpu3(y, D, s, lmbda, niter, stop_rel = 1e-5):

    device = torch.device('cuda:3')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)
    alpha = (1 / s) * 0.99
    y_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    t_k = 1
    t_loss_list = list()
    t_dloss_list = list()

    for i in range(niter):
        
        x_kp1 = torch.clip(y_k - alpha * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        y_kp1 = x_kp1 + ((t_k-1)/t_kp1) * (x_kp1 - x_k)
        x_k = x_kp1
        t_k = t_kp1
        y_k = y_kp1

        
        if i % 20000 == 0:
            beta = y - DT.matmul(x_kp1)
            df_loss = ((beta-y)**2).sum() / 2
            htv_loss = torch.norm(D.matmul(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
            t_loss_list.append(total_loss)
            dloss = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
            t_dloss_list.append(dloss)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item(), dloss.item())



    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    dloss = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
    t_dloss_list.append(dloss)
    print('FINAL', i,  df_loss.item(), htv_loss.item(), total_loss.item())
    beta = y - DT @ x_kp1

    return beta, t_loss_list, t_dloss_list


def ista_for_dual_gpu(y, D, s, lmbda, niter):

    device = torch.device('cuda:0')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)
    alpha = (1 / s) * 0.99
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    t_loss_list = list()
    t_dloss_list = list()

    for i in range(niter):
        
        x_kp1 = torch.clip(x_k - alpha * (2 * mDy + 2 * D.matmul(DT.matmul(x_k))), -lmbda , lmbda)
        x_k = x_kp1


        if i % 20000 == 0:
            beta = y - DT.matmul(x_kp1)
            df_loss = ((beta-y)**2).sum() / 2
            htv_loss = torch.norm(D.matmul(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
            
            dloss = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
            t_dloss_list.append(dloss)
            t_loss_list.append(total_loss)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item(), dloss.item())

    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    dloss = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
    t_dloss_list.append(dloss)
    print('FINAL', df_loss.item(), htv_loss.item(), total_loss.item())
    beta = y - DT @ x_kp1

    return beta, t_loss_list, t_dloss_list


def greedy_fista_gpu(y, D, s, lmbda, niter):

    device = torch.device('cuda:3')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)
    gamma = (1 / s) * 0.99
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    y_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    x_km1 = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    t_loss_list = list()


    for i in range(niter):
        
        y_k = x_k + (x_k - x_km1)
        x_kp1 = torch.clip(y_k - gamma * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        x_km1 = x_k
        x_k = x_kp1

        if (y_k - x_kp1).dot((x_kp1-x_k)) >= 0: 
            y_k = x_k
        
        #if torch.norm(x_kp1-x_k) >= S * torch.norm(x_1-x_0):
         #   gamma = np.max(zeta*gamma, gamma)

        if i % 20000 == 0:
            beta = y - DT.matmul(x_kp1)
            df_loss = ((beta-y)**2).sum() / 2
            htv_loss = torch.norm(D.matmul(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
            t_loss_list.append(total_loss)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item())

    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    print('FINAL', df_loss.item(), htv_loss.item(), total_loss.item())
    beta = y - DT @ x_kp1

    return beta, t_loss_list


def fista_restart_for_dual_v2(y, D, s, lmbda, niter):

    device = torch.device('cuda:0')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)

    L_inv = (1 / s) * 0.99
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    x_km1 = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    theta_k = 1
    theta_km1 = 1 
    restart = False
    loss_primal = list()
    loss_dual = list()

    for i in range(niter):
        restart = False
        y_k = x_k + theta_k * (1/theta_km1 - 1) * (x_k - x_km1)
        x_kp1 = torch.clip(y_k - L_inv * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        theta_kp1 = (np.sqrt(theta_k**4 + 4*theta_k**2) - theta_k**2) / 2
        
        beta = y - DT.matmul(x_kp1)
        df_loss = ((beta-y)**2).sum() / 2
        htv_loss = torch.norm(D.matmul(beta), 1)
        total_loss = df_loss + lmbda * htv_loss
        
        if i == 0:
            loss_primal_k = total_loss
        else:
            loss_primal_kp1 = total_loss
            if (loss_primal_kp1 > loss_primal_k):
                restart = True
            loss_primal_k = loss_primal_kp1

        if restart: 
            y_k = x_k
            theta_k = theta_km1 
            theta_kp1 = theta_k
            x_kp1 =  torch.clip(y_k - L_inv * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        
        x_km1 = x_k
        x_k = x_kp1
        theta_km1 = theta_k
        theta_k = theta_kp1

        if i % 20000 == 0:
            loss_primal.append(total_loss)
            loss_d = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
            loss_dual.append(loss_d)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item(), loss_d.item())



    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss
    loss_primal.append(total_loss)
    dloss = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
    loss_dual.append(dloss)
    print('FINAL', i,  df_loss.item(), htv_loss.item(), total_loss.item(), dloss.item())
    beta = y - DT @ x_kp1

    return beta, loss_primal, loss_dual

def fista_restart_for_dual(y, D, s, lmbda, niter):

    device = torch.device('cuda:0')

    DT = D.transpose().tocsr()
    D = torch.sparse_csr_tensor(D.indptr.tolist(), D.indices.tolist(), D.data.tolist(), dtype=torch.double, device=device)
    DT = torch.sparse_csr_tensor(DT.indptr.tolist(), DT.indices.tolist(), DT.data.tolist(), dtype=torch.double, device=device)
    y = torch.from_numpy(y).double().to(device)

    mDy = -D.matmul(y)

    L_inv = (1 / s) * 0.99
    x_k = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    x_km1 = torch.from_numpy(np.zeros((D.shape[0]))).double().to(device)
    theta_k = 1
    theta_km1 = 1 
    restart = False
    loss_primal = list()
    loss_dual = list()

    for i in range(niter):
        restart = False
        y_k = x_k + theta_k * (1/theta_km1 - 1) * (x_k - x_km1)
        x_kp1 = torch.clip(y_k - L_inv * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        theta_kp1 = (np.sqrt(theta_k**4 + 4*theta_k**2) - theta_k**2) / 2
        
        beta = y - DT.matmul(x_kp1)
        df_loss = ((beta-y)**2).sum() / 2
        htv_loss = torch.norm(D.matmul(beta), 1)
        total_loss = df_loss + lmbda * htv_loss
        
        if i == 0:
            loss_primal_k = total_loss
        else:
            loss_primal_kp1 = total_loss
            if (loss_primal_kp1 > loss_primal_k):
                restart = True
            loss_primal_k = loss_primal_kp1

        if restart: 
            y_k = x_k
            x_kp1 =  torch.clip(y_k - L_inv * (2 * mDy + 2 * D.matmul(DT.matmul(y_k))), -lmbda , lmbda)
        
        x_km1 = x_k
        x_k = x_kp1
        theta_km1 = theta_k
        theta_k = theta_kp1

        if i % 20000 == 0:
            loss_primal.append(total_loss)
            loss_d = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
            loss_dual.append(loss_d)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item(), loss_d.item())



    beta = y - DT.matmul(x_kp1)
    df_loss = ((beta-y)**2).sum() / 2
    htv_loss =torch.norm(D.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss
    loss_primal.append(total_loss)
    dloss = 1/2 * (torch.norm(y-DT.matmul(x_kp1), 2)**2)
    loss_dual.append(dloss)
    print('FINAL', i,  df_loss.item(), htv_loss.item(), total_loss.item(), dloss.item())
    beta = y - DT @ x_kp1

    return beta, loss_primal, loss_dual


def ADMM(tri, rho, lmbda, niter=200):

    device = torch.device('cuda:0')

    L = torch.tensor(tri.L.toarray(), device=device)
    LT = L.transpose()

    H = torch.tensor(tri.H.toarray(), device=device)
    HT = H.transpose()

    y = torch.from_numpy(tri.data_values).double().to(device)

    gamma_k = torch.matmul(L, y)
    alpha_k = gamma_k - gamma_k

    for i in range(niter):
        print(i)
        theta_kp1 = torch.linalg.solve(rho*LT.matmul(L)+HT.matmul(H), HT.matmul(y) + rho * LT.matmul(gamma_k - 1/rho * alpha_k))
        x = L.matmul(theta_kp1 + 1/rho * alpha_k)
        t = lmbda / rho

        signs = ((x < -t)*1 - (x > t)*1)
        gamma_kp1 = torch.abs(signs) * x + t * signs
        alpha_kp1 = alpha_k + rho * (L.matmul(theta_kp1) - gamma_kp1)

        alpha_k = alpha_kp1
        gamma_k = gamma_kp1


    return theta_kp1