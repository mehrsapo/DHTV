import numpy as np

import torch

import os

def prox_Lc_l1(z, L, LT, lip_L, lmbda, niter, device='cuda:2', verbose=False):

    device = torch.device(device)

    mLz = -L.matmul(z)
    alpha = (1 / lip_L) 

    u_k = torch.from_numpy(np.zeros((L.shape[0]))).double().to(device)
    v_k = torch.from_numpy(np.zeros((L.shape[0]))).double().to(device)

    u_kp1 = torch.from_numpy(np.zeros((L.shape[0]))).double().to(device)
    v_kp1 = torch.from_numpy(np.zeros((L.shape[0]))).double().to(device)

    t_k = 1
    t_loss_list = list()
    t_dloss_list = list()

    for i in range(niter):
    
        if verbose and (i % 10000 == 0):
            beta = z - LT.matmul(u_kp1)
            df_loss = ((beta-z)**2).sum() / 2
            htv_loss = torch.norm(L.matmul(beta), 1)
            total_loss = df_loss + lmbda * htv_loss
            t_loss_list.append(total_loss)
            dloss = 1/2 * (torch.norm(z-LT.matmul(u_kp1), 2)**2)
            t_dloss_list.append(dloss)
            print(i, df_loss.item(), htv_loss.item(), total_loss.item(), dloss.item())

        u_kp1 = torch.clip(v_k - alpha * (mLz + L.matmul(LT.matmul(v_k))), -lmbda , lmbda)
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        v_kp1 = u_kp1 + ((t_k-1)/t_kp1) * (u_kp1 - u_k)
        u_k = u_kp1
        t_k = t_kp1
        v_k = v_kp1

    
    beta = z - LT.matmul(u_kp1)
    df_loss = ((beta-z)**2).sum() / 2
    htv_loss =torch.norm(L.matmul(beta), 1)
    total_loss = df_loss + lmbda * htv_loss

    dloss = 1/2 * (torch.norm(z-LT.matmul(u_kp1), 2)**2)
    t_dloss_list.append(dloss)

    if verbose:
        print('FINAL', i,  df_loss.item(), htv_loss.item(), total_loss.item())

    beta = z - LT @ u_kp1

    return beta, t_loss_list, t_dloss_list

def double_fista(y, H, L, lip_H, lip_L, lmbda, niter_fista, niter_prox, device='cuda:2', verbose=False, save_dir=None):

    device = torch.device(device)
    HT = H.transpose().tocsr()
    H = torch.sparse_csr_tensor(H.indptr.tolist(), H.indices.tolist(), H.data.tolist(), dtype=torch.double, size = H.shape, device=device)
    HT = torch.sparse_csr_tensor(HT.indptr.tolist(), HT.indices.tolist(), HT.data.tolist(), dtype=torch.double, device=device)

    LT = L.transpose().tocsr()
    L = torch.sparse_csr_tensor(L.indptr.tolist(), L.indices.tolist(), L.data.tolist(), dtype=torch.double, size = L.shape, device=device)
    LT = torch.sparse_csr_tensor(LT.indptr.tolist(), LT.indices.tolist(), LT.data.tolist(), dtype=torch.double, size = (L.shape[1], L.shape[0]), device=device)

    y = torch.from_numpy(y).double().to(device)

    mHty = -1 * HT.matmul(y)
    alpha = (1 / lip_H) 

    c_k = torch.from_numpy(np.zeros((H.shape[1]))).double().to(device)
    d_k = torch.from_numpy(np.zeros((H.shape[1]))).double().to(device)

    c_kp1 = torch.from_numpy(np.zeros((H.shape[1]))).double().to(device)
    d_kp1 = torch.from_numpy(np.zeros((H.shape[1]))).double().to(device)

    t_k = 1
    t_loss_list = list()

    if save_dir is not None: 
        if os.path.exists(save_dir):
            os.remove(save_dir)

    for i in range(niter_fista):
        
        if save_dir is not None and (i % 5 == 0) :
            df_loss = ((H.matmul(c_k)-y)**2).sum() / 2
            htv_loss = torch.norm(L.matmul(c_k), 1)
            total_loss = df_loss + lmbda * htv_loss
            with open(save_dir, 'a') as file:
                file.write(str(df_loss.item()) + ' ' + str(htv_loss.item()) + ' ' + str(total_loss.item()) + '\n')
        if verbose and (i % 100 == 0):
            df_loss = ((H.matmul(c_k)-y)**2).sum() / 2
            htv_loss = torch.norm(L.matmul(c_k), 1)
            total_loss = df_loss + lmbda * htv_loss
            t_loss_list.append(total_loss)
            
            print(i, df_loss.item(), htv_loss.item(), total_loss.item())

        c_kp1 = prox_Lc_l1(d_k - alpha * (mHty + HT.matmul(H.matmul(d_k))), L, LT, lip_L, alpha * lmbda, niter_prox, device, verbose=False)[0]
        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        d_kp1 = c_kp1 + ((t_k-1)/t_kp1) * (c_kp1 - c_k)
        c_k = c_kp1
        t_k = t_kp1
        d_k = d_kp1
    
    df_loss = ((H.matmul(c_k)-y)**2).sum() / 2
    htv_loss = torch.norm(L.matmul(c_k), 1)
    total_loss = df_loss + lmbda * htv_loss
    t_loss_list.append(total_loss)

    if verbose:
        print('FINAL', i,  df_loss.item(), htv_loss.item(), total_loss.item())
    
    return c_kp1, t_loss_list