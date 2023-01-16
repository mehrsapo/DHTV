import torch

class RBF:

    def __init__(self, data, eps, lmbda):
        self.data = data
        self.eps = eps
        self.lmbda = lmbda

        self.input = self.data.train['input']
        self.values = self.data.train['values']

        self.init_gram_mat()
        self.init_coeffs()

    def init_gram_mat(self):
        loc_diff_pairs = self.input.unsqueeze(0) - self.input.unsqueeze(1)
        distance_pairs = torch.norm(loc_diff_pairs, dim=-1)
        self.gram_mat = torch.exp(-(self.eps * distance_pairs.double()) ** 2)

    def init_coeffs(self):
        A = self.gram_mat + self.lmbda * torch.ones_like(self.gram_mat[:, 0]).diag()
        B = self.values.unsqueeze(-1).to(dtype=self.gram_mat.dtype, device=self.gram_mat.device)
        X, _ = torch.lstsq(B, A)
        self.coeffs = X.squeeze(-1).float()

    def construct_H_mat(self, x):
        loc_diff_pairs = x.unsqueeze(1) - self.input.unsqueeze(0)
        distance_pairs = torch.norm(loc_diff_pairs, dim=-1)
        H_mat = torch.exp(-(self.eps * distance_pairs) ** 2)
        return H_mat

    def evaluate(self, x):
        H_mat = self.construct_H_mat(x)
        output = torch.mv(H_mat, self.coeffs)
        return output
