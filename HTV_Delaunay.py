from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy import special
import numpy as np
import torch


class MyDelaunay(Delaunay):

    simplices = None

    valid_simplices = None
    valid_transform = None
    valid_neighbors = None
    graph_i = None
    graph_j = None
    centers_barycentric_coordinates = None
    k = 0

    def __init__(self, data_points, values, grid_points=None, regular=False):
        """
        Construct the delaunay triangulation
        """

        self.n_data_points, self.dimension = data_points.shape

        self.centers_barycentric_coordinates = np.ones((self.dimension+1, )) * (1/(self.dimension+1))

        if grid_points is None:
            self.grid_points = data_points
            self.grid_values = values
        else:
            self.grid_points = grid_points
            self.grid_values = np.zeros((self.grid_points.shape[0], ))

        super().__init__(self.grid_points)
        if regular:
            counter = 0
            self.keep = np.ones(len(self.simplices), dtype=int)
            for i, t in enumerate(self.simplices):
                if abs(np.linalg.det(np.hstack((self.points[t], np.ones([1, self.dimension + 1]).T)))) < 1E-6:
                    self.keep[i] = -1  # Point is coplanar, we don't want to keep it
                else:
                    self.keep[i] = counter
                    counter = counter + 1

            self.keep_map = np.concatenate((self.keep, np.array([-1])), axis=0)
            self.zero_simplices = np.where(self.keep < 0)

            self.valid_simplices_idx = np.where(self.keep > -1)

            self.data_points = data_points
            self.data_values = values

            self.valid_simplices = self.simplices[self.valid_simplices_idx]
            self.valid_transform = self.transform[self.valid_simplices_idx]
            self.valid_neighbors = self.keep_map[self.neighbors[self.valid_simplices_idx]]

        else:

            self.data_points = data_points
            self.data_values = values
            self.keep_map = np.concatenate((np.arange(len(self.simplices), dtype=int), np.array([-1])), axis=0)
            self.valid_simplices = self.simplices
            self.valid_transform = self.transform
            self.valid_neighbors = self.neighbors

        self.n_grid_points, _ = self.grid_points.shape
        self.n_simplices, self.n_simplex_vertices = self.valid_simplices.shape

        assert self.data_values.shape == (self.n_data_points,)
        assert self.n_simplex_vertices == self.dimension + 1

        self.simplices_of_points = self.find_simplex_valid(data_points)

        self.grid_points_of_simplices_of_points = self.valid_simplices[self.simplices_of_points]

        data_points_transform = self.valid_transform[self.simplices_of_points]
        t = data_points_transform[:, :self.dimension, :]
        r = data_points_transform[:, self.dimension, :]

        c = np.einsum('BNi,Bi ->BN', t, self.data_points - r)
        self.data_points_barycentric_coordinate = np.concatenate((c, (1 - np.sum(c, axis=1))[:, np.newaxis]), axis=1)

        self.simplices_sorted = np.sort(self.valid_simplices, axis=1)
        self.H = None
        self.L = None
        self.unit_normals = None
        self.final_T = None
        self.calculate_grad_for_simplices()

        self.valid_simplices_centers = np.matmul(self.centers_barycentric_coordinates,
                                                 self.grid_points[self.simplices_sorted])

        self.lat_coeffs = self.give_affine_coef(self.valid_simplices_centers)

        return

    def find_simplex_valid(self, points):

        simplices = self.find_simplex(points)
        return self.keep_map[simplices]

    def construct_forward_matrix(self):

        rows = np.repeat(np.arange(self.n_data_points), self.dimension+1)
        cols = self.grid_points_of_simplices_of_points.flatten()
        data = self.data_points_barycentric_coordinate.flatten()

        self.H = csr_matrix((data, (rows, cols)), shape=(self.n_data_points, self.n_grid_points), dtype='float32')
        self.H.eliminate_zeros()
        self.H.check_format()

        return

    def calculate_normal_vectors(self):

        point_batches = self.grid_points[self.valid_simplices]
        m, n, _ = point_batches.shape
        point_batches_one_added = np.concatenate((point_batches, np.ones((m, n, 1))), axis=2)
        normals = np.linalg.inv(point_batches_one_added)
        normals = np.delete(normals, self.dimension, axis=1)
        norm = np.linalg.norm(normals, axis=1)[:, None]
        unit_normals = normals / norm
        self.unit_normals = unit_normals.transpose(0, 2, 1)

        return


    def calculate_grad_for_simplices(self):

        points = np.delete(self.grid_points[self.simplices_sorted], 0, axis=1) - \
                 self.grid_points[self.simplices_sorted][:, 0, None, :]
        point_invs = np.linalg.inv(points)
        A_by_1 = np.matmul(point_invs, -1 * np.ones((np.shape(self.valid_simplices)[0], self.dimension, 1)))
        self.final_T = np.concatenate((A_by_1, point_invs), axis=2)

        return

    def vol_simplex(self, simplex):
        simplex = np.flip(simplex)
        combs = torch.tensor(list(combinations(simplex, self.dimension)))
        X = torch.tensor(self.grid_points[combs])
        D = torch.cdist(X, X, 2) ** 2
        D = torch.cat((D, torch.ones(self.dimension + 1, self.dimension, 1)), 2)
        D = torch.cat((D, torch.cat((torch.ones(self.dimension + 1, 1, self.dimension),
                                     torch.zeros(self.dimension + 1, 1, 1)), 2)), 1)
        det = torch.det(D)
        vol_2 = ((-1) ** self.dimension) * det / ((special.factorial(self.dimension - 1) ** 2) *
                                                  np.power(2, self.dimension - 1))
        vol = torch.sqrt(vol_2).numpy()

        return vol


    def calculate_all_volumes(self):
        self.volumes = np.array(list(map(self.vol_simplex, self.valid_simplices)))

        return

    def construct_regularization_matrix(self):

        self.calculate_all_volumes()
        self.calculate_normal_vectors()

        simplices = self.simplices_sorted
        neighbors = self.valid_neighbors
        final_T = self.final_T
        unit_normals = self.unit_normals

        temp_n = neighbors.reshape((neighbors.shape[0] * (self.dimension+1),))
        valid_neighbors = np.where(temp_n > -1)[0]
        ns_vector = temp_n[valid_neighbors]
        i_vector = np.repeat(np.arange(neighbors.shape[0]), (self.dimension+1))
        j_vector = np.tile(np.arange((self.dimension+1)), neighbors.shape[0])
        i_vector = i_vector[valid_neighbors]
        j_vector = j_vector[valid_neighbors]

        neigh_sim = np.sort(np.concatenate((i_vector[:, None], ns_vector[:, None]), axis=1), axis=1)
        un_neigh_sim = np.unique(neigh_sim, return_index=True, axis=0)
        ni = un_neigh_sim[1]

        i_vector = i_vector[ni]
        j_vector = j_vector[ni]
        ns_vector = ns_vector[ni]

        self.graph_i = i_vector
        self.graph_j = ns_vector

        temp1 = np.tile(simplices, (1, (self.dimension+1))).reshape(simplices[neighbors].shape)
        temp2 = simplices[neighbors]
        temp3 = np.concatenate((temp1, temp2), axis=2)
        temp4 = temp3.reshape((temp3.shape[0] * (self.dimension+1), (self.dimension+1)*2))
        temp5 = temp4[valid_neighbors]

        temp5 = temp5[ni]

        out = np.apply_along_axis(lambda x: np.unique(x, return_index=True, return_counts=True), 1, temp5)
        unique = out[:, 0]
        ret_idx = out[:, 1]
        counts = out[:, 2]
        b = np.where(counts == 1)
        good = unique[b].reshape((unique.shape[0], 2))
        unique[b] = -1
        u_sorted = np.sort(unique, axis=1)[:, 2:]
        cols = np.concatenate((np.concatenate((good[:, 0][:, None], u_sorted), axis=1), good[:, 1][:, None]), axis=1)
        ids_id = ret_idx[b].reshape((unique.shape[0], 2))
        ids = np.sort(ids_id, axis=1)
        ids[:, 1] = ids[:, 1] - (self.dimension+1)

        change_plcaes = np.where(ids_id[:, 1] - ids_id[:, 0] < 0)[0]

        T1 = final_T[i_vector]
        T2 = final_T[ns_vector]
        c1 = T1[np.arange(ids.shape[0]), :, ids[:, 0]]
        c2 = T2[np.arange(ids.shape[0]), :, ids[:, 1]]
        mask1 = np.tile(np.arange((self.dimension+1)), ids.shape[0]).reshape((ids.shape[0], (self.dimension+1)))
        mask1[np.arange(ids.shape[0]), ids[:, 0]] = -1
        mask1 = np.sort(mask1)[:, 1:]
        x1 = np.repeat(np.arange(ids.shape[0]), self.dimension * self.dimension)
        x2 = np.tile(np.repeat(np.arange(self.dimension), self.dimension), ids.shape[0])
        x3 = np.tile(mask1, self.dimension).flatten()

        mask2 = np.tile(np.arange((self.dimension+1)), ids.shape[0]).reshape((ids.shape[0], (self.dimension+1)))
        mask2[np.arange(ids.shape[0]), ids[:, 1]] = -1
        mask2 = np.sort(mask2)[:, 1:]
        x4 = np.tile(mask2, self.dimension).flatten()

        T1_r = T1[x1, x2, x3].reshape((ids.shape[0], self.dimension, self.dimension))
        T2_r = T2[x1, x2, x4].reshape((ids.shape[0], self.dimension, self.dimension))

        T1_n = np.concatenate((np.concatenate((c1[:, :, None], T1_r), axis=2), np.zeros(c1[:, :, None].shape)), axis=2)
        T2_n = np.concatenate((np.concatenate((np.zeros(c2[:, :, None].shape), T2_r), axis=2), c2[:, :, None]), axis=2)
        T_diff = T1_n - T2_n
        U = unit_normals[i_vector, j_vector, :]
        T_U = np.matmul(U[:, None, :], T_diff).reshape((ids.shape[0], (self.dimension+2)))
        g1 = T_U[:, 0].copy()
        g1[change_plcaes] = T_U[change_plcaes, -1]
        g2 = T_U[:, -1].copy()
        g2[change_plcaes] = T_U[change_plcaes, 0]
        T_U[:, 0] = g1
        T_U[:, -1] = g2
        data = T_U * self.volumes[i_vector, j_vector][:, None]

        rows = np.repeat(np.arange(data.shape[0]), (self.dimension+2))
        data = data.flatten()
        cols = cols.flatten()

        self.L = csr_matrix((data, (rows, cols)), shape=(i_vector.shape[0], self.n_grid_points), dtype='float32')

        self.L.eliminate_zeros()
        self.L.check_format()

        return

    def evaluate(self, points, values=None):

        if values is None:
            values = self.grid_values

        simplices_of_points = self.find_simplex_valid(points)
        non_valid_simplices = np.where(simplices_of_points == -1)

        values_of_simplices_of_points = values[self.valid_simplices[simplices_of_points]]

        data_points_transform = self.valid_transform[simplices_of_points]
        t = data_points_transform[:, :self.dimension, :]
        r = data_points_transform[:, self.dimension, :]

        c = np.einsum('BNi,Bi ->BN', t, points - r)
        points_barycentric_coordinate = np.concatenate((c, (1 - np.sum(c, axis=1))[:, np.newaxis]), axis=1)

        values = np.einsum('ij,ij-> i', values_of_simplices_of_points, points_barycentric_coordinate)
        values[non_valid_simplices] = 0

        return values

    def update_values(self, grid_values):
        self.grid_values = grid_values
        self.data_values = self.H * self.grid_values
        self.lat_coeffs = self.give_affine_coef(self.valid_simplices_centers)
        return

    def give_affine_coef(self, points):
        simplex = self.find_simplex_valid(points)
        temp = self.grid_values[self.simplices_sorted[simplex]][:, :, np.newaxis]
        coefs = np.matmul(self.final_T[simplex], temp)
        coefs = np.reshape(coefs.ravel(), (points.shape[0], self.dimension))

        return coefs
