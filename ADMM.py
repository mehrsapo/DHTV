import scipy.sparse.linalg
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import identity

class ADMM:

    def __init__(self, tri, rho, lmbda):

        self.D = tri.L
        self.X = tri.H
        self.Dt = self.D.transpose()
        self.Xt = self.X.transpose()

        self.DtD = self.Dt * self.D
        self.XtX = identity(self.X.shape[0])

        J = self.XtX + rho * self.DtD
        print(J.shape)
        self.XtX_rho_DtD_inverse = scipy.sparse.linalg.inv(J)

        self.tri = tri
        self.y = tri.data_values
        self.rho = rho
        self.lmbda = lmbda

    def soft_thrsh(self, x, t):
        signs = ((x < -t)*1 - (x > t)*1)
        out = np.abs(signs) * x + t * signs
        return out

    def iterate(self, n_iter=200):
        alpha_k = self.D * self.y
        w_k = alpha_k * 0
        b_k = alpha_k * 0

        for iter in range(n_iter):
            if iter % 500 == 499:
                htv_vec = np.abs(self.tri.L * b_k)
                htv_loss_admm = np.sum(htv_vec)
                df_loss_admm = np.sum((self.tri.H * b_k - self.y) ** 2)
                total_loss_admm = df_loss_admm + self.lmbda * htv_loss_admm * 2
                print(df_loss_admm, htv_loss_admm, total_loss_admm)

            b_k = self.XtX_rho_DtD_inverse * (self.Xt * self.y + self.rho * self.Dt * (alpha_k - w_k))
            alpha_k = self.soft_thrsh(self.D * b_k + w_k, self.lmbda / self.rho)
            w_k = w_k + self.D * b_k - alpha_k

        return b_k


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

        
        self.rho = rho

    def soft_thrsh(self, x, t):
        signs = ((x < -t)*1 - (x > t)*1)
        out = np.abs(signs) * x + t * signs
        return out

    def iterate(self, n_iter=200):
        alpha_k = self.D @ self.tri.grid_values
        w_k = alpha_k * 0

        for iter in range(n_iter):
            if iter > 1:
                print(iter)
                htv_vec = np.abs(self.tri.L @ b_k)
                htv_loss_admm = np.sum(htv_vec)
                df_loss_admm = np.sum((self.tri.H.dot(b_k) - self.y) ** 2) 
                total_loss_admm = df_loss_admm + self.lmbda * htv_loss_admm * 2
                print(df_loss_admm, htv_loss_admm, total_loss_admm)

            # b_k = scipy.sparse.linalg.spsolve(self.M, (self.Xty + self.rho * self.Dt @ (alpha_k - w_k)))
            b_k = scipy.linalg.solve(self.M.todense(), (self.Xty + self.rho * self.Dt @ (alpha_k - w_k)))
            alpha_k = self.soft_thrsh(self.D @ b_k + w_k, self.lmbda / self.rho)
            w_k = w_k + self.D @ b_k - alpha_k

        return b_k