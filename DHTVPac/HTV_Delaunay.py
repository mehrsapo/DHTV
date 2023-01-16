from itertools import combinations
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
from scipy import special
import numpy as np
import torch
import scipy.sparse.linalg
from scipy.spatial import ConvexHull
from quadprog import solve_qp
import scipy

torch.set_grad_enabled(False)
torch.set_num_threads(4)

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

class MyDelaunay(Delaunay):

    simplices = None

    valid_simplices = None
    valid_transform = None
    valid_neighbors = None
    graph_i = None
    graph_j = None
    centers_barycentric_coordinates = None
    k = 0

    def __init__(self, data_points, values, grid_points=None, trim=True, two_prun=True):
        """
        Construct the delaunay triangulation
        """

        self.identity = False
        self.two_prun = False
        data_points = data_points 
        self.n_data_points, self.dimension = data_points.shape

        self.centers_barycentric_coordinates = np.ones((self.dimension+1, )) * (1/(self.dimension+1))


        if grid_points is None:

            self.identity = True
            records_array =data_points
            _, inverse, count = np.unique(records_array, return_inverse=True,
                                        return_counts=True, axis=0)

            idx_vals_repeated = np.where(count > 0)[0]

            rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])

            _, inverse_rows = np.unique(rows, return_index=True)
            res = np.split(cols, inverse_rows[1:])
            
            values_unique = np.array([np.mean(values[i]) for i in res])
            data_unique = np.array([list(data_points[i[0]]) for i in res])
            
            self.res = res

            self.grid_points = data_unique.copy()
            self.grid_values = values_unique.copy()
            data_points = data_unique.copy()
            values = values_unique.copy()
            self.n_data_points, self.dimension = data_points.shape
        else:
            self.grid_points = grid_points
            self.grid_values = np.zeros((self.grid_points.shape[0], ))

        super().__init__(self.grid_points)

        self.hull_b =  ConvexHull(self.grid_points)

        self.simplices_vols = list()

        if trim:
            counter = 0
            self.keep = np.ones(len(self.simplices), dtype=int)
            for i, t in enumerate(self.simplices):
                self.simplices_vols.append(abs(np.linalg.det(np.hstack((self.points[t], np.ones([1, self.dimension + 1]).T)))))
                if abs(np.linalg.det(np.hstack((self.points[t], np.ones([1, self.dimension + 1]).T)))) < 1E-7:
                    self.keep[i] = -1  # Point is coplanar, we don't want to keep it
                else:
                    self.keep[i] = counter
                    counter = counter + 1

            self.keep_map = np.concatenate((self.keep, np.array([-1])), axis=0)
            self.zero_simplices = np.where(self.keep < 0)

            self.valid_simplices_idx = np.where(self.keep > -1)[0]

            self.data_points = data_points
            self.data_values = values

            self.valid_simplices = self.simplices[self.valid_simplices_idx]
            self.valid_vols_sim = np.array(self.simplices_vols).ravel()[self.valid_simplices_idx]
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
        self.calculate_all_volumes()
        area = np.sum(self.volumes, axis=1)
        
        self.valid_simplices_centers = np.matmul(self.centers_barycentric_coordinates,
                                                 self.grid_points[self.simplices_sorted])
        self.lat_coeffs = self.give_affine_coef(self.valid_simplices_centers)

        
        if two_prun: 
            counter = 0
            lat_coeff_norm = np.linalg.norm(self.lat_coeffs, 2, axis=1)
            self.keep2 = np.ones(len(self.valid_simplices), dtype=int)
            for i, t in enumerate(self.valid_simplices):
                if (lat_coeff_norm[i] >  np.mean(lat_coeff_norm) + 5 * np.std(lat_coeff_norm)) or (area[i] > np.mean(area) + 5 * np.std(area)):
                    self.keep2[i] = -1  # Point is related to tiny triangles with large area
                else:
                    self.keep2[i] = counter
                    counter = counter + 1

            self.keep_map2 = np.concatenate((self.keep2, np.array([-1])), axis=0)
            self.valid_simplices_idx = np.where(self.keep2 > -1)

            self.valid_simplices = self.valid_simplices[self.valid_simplices_idx]
            self.valid_vols_sim = self.valid_vols_sim[self.valid_simplices_idx]
            self.valid_transform = self.valid_transform[self.valid_simplices_idx]
            self.valid_neighbors = self.keep_map2[self.valid_neighbors[self.valid_simplices_idx]]
            self.volumes = self.volumes[self.valid_simplices_idx]
            self.n_simplices, self.n_simplex_vertices = self.valid_simplices.shape
            # self.final_T = self.final_T[self.valid_simplices_idx]
            self.two_prun = True
            self.simplices_of_points = self.find_simplex_valid(data_points)
            self.grid_points_of_simplices_of_points = self.valid_simplices[self.simplices_of_points]

            data_points_transform = self.valid_transform[self.simplices_of_points]
            t = data_points_transform[:, :self.dimension, :]
            r = data_points_transform[:, self.dimension, :]

            c = np.einsum('BNi,Bi ->BN', t, self.data_points - r)
            self.data_points_barycentric_coordinate = np.concatenate((c, (1 - np.sum(c, axis=1))[:, np.newaxis]), axis=1)

            self.simplices_sorted = np.sort(self.valid_simplices, axis=1)
            
            self.valid_simplices_centers = np.matmul(self.centers_barycentric_coordinates,
                                                    self.grid_points[self.simplices_sorted])
                                                    
            self.lat_coeffs = self.give_affine_coef(self.valid_simplices_centers)
            self.calculate_grad_for_simplices()

        #print('per keep', self.n_simplices / self.simplices.shape[0])
        return

    def find_simplex_valid(self, points):

        simplices = self.find_simplex(points, bruteforce=False)
        if self.two_prun:
            return self.keep_map2[self.keep_map[simplices]]
        else:
            return self.keep_map[simplices]

    def construct_forward_matrix(self):
        
        if self.identity: 
            self.H = scipy.sparse.csr_matrix(scipy.sparse.identity(self.n_data_points))
            self.lip_H = 1.0

        else:
            rows = np.repeat(np.arange(self.n_data_points), self.dimension+1)
            cols = self.grid_points_of_simplices_of_points.flatten()
            data = self.data_points_barycentric_coordinate.flatten()

            self.H = csr_matrix((data, (rows, cols)), shape=(self.n_data_points, self.n_grid_points), dtype='float64')
            self.H.eliminate_zeros()
            self.H.check_format()

            x = self.H
            nonzero_mask = np.array(np.abs(x[x.nonzero()]) < 1e-5)[0]
            rows = x.nonzero()[0][nonzero_mask]
            cols = x.nonzero()[1][nonzero_mask]
            x[rows, cols] = 0
            self.H = x
            self.H.eliminate_zeros()
            self.H.check_format()

            _, s, _ = scipy.sparse.linalg.svds(self.H, k=1)
            self.lip_H = s[0] ** 2 

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

    def cal_neigh_vectors(self): 

        neighbors = self.valid_neighbors  

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
        self.i_vector = i_vector
        j_vector = j_vector[ni]
        self.j_vector = j_vector
        ns_vector = ns_vector[ni]
        self.ns_vector = ns_vector
        
    def construct_regularization_matrix(self):

        #self.calculate_all_volumes()
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
        self.i_vector = i_vector
        j_vector = j_vector[ni]
        self.j_vector = j_vector
        ns_vector = ns_vector[ni]
        self.ns_vector = ns_vector
        temp5 = simplices[un_neigh_sim[0]].reshape((ni.shape[0], 2*self.dimension+2))

        out = np.apply_along_axis(lambda x: np.unique(x, return_index=True, return_counts=True), 1, temp5)
        unique = out[:, 0]
        ret_idx = out[:, 1]
        counts = out[:, 2]
        b = np.where(counts == 1)
        good = unique[b].reshape((unique.shape[0], 2))
        unique[b] = -1
        u_sorted = np.sort(unique, axis=1)[:, 2:]
        
        ids_id = ret_idx[b].reshape((unique.shape[0], 2))
        ids = np.sort(ids_id, axis=1)
        ids[:, 1] = ids[:, 1] - (self.dimension+1)

        change_plcaes = np.where(ids_id[:, 1] - ids_id[:, 0] < 0)[0]
        temp = good[change_plcaes, 0]
        good[change_plcaes, 0] = good[change_plcaes, 1]
        good[change_plcaes, 1] = temp
        cols = np.concatenate((np.concatenate((good[:, 0][:, None], u_sorted), axis=1), good[:, 1][:, None]), axis=1)


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
      
        data = T_U * self.volumes[i_vector, j_vector][:, None]

        rows = np.repeat(np.arange(data.shape[0]), (self.dimension+2))
        data = data.flatten()
        data = np.nan_to_num(data, nan = 0)

        cols = cols.flatten()
        
        self.L = csr_matrix((data, (rows, cols)), shape=(i_vector.shape[0], self.n_grid_points), dtype='float64')
        
        self.L.eliminate_zeros()
        self.L.check_format()

        _, s, _ = scipy.sparse.linalg.svds(self.L, k=1)
        self.lip_L = s[0] ** 2 
        

        return

    def find_closest_point(self, point):
        dist_2 = np.sum((self.grid_points - point)**2, axis=1)
        return np.argmin(dist_2)


    def proj_extrapolate(self, z):
        equations = self.hull_b.equations
        G = np.eye(len(z), dtype='float64')
        a = np.array(z, dtype='float64')
        C = np.array(-equations[:, :-1], dtype='float64')
        b = np.array(equations[:, -1], dtype='float64')
        x, f, xu, itr, lag, act = solve_qp(G, a, C.T, b, meq=0, factorized=True)
        return x

    

    def evaluate_base(self, points, values, neigh_n=False):
        
        points = np.array([points])
        points_org = points.copy()
        simplices_of_points = self.find_simplex(points)
        i = 0 

        if neigh_n:
            return self.grid_values[self.find_closest_point(points_org)]
        else:
            while simplices_of_points[0] == -1: 
                i = i + 1
                points = trunc(self.proj_extrapolate(points[0]), 10)
                points = np.array([points])
                simplices_of_points = self.find_simplex(points, bruteforce=True)
                if i > 5: 
                    return self.grid_values[self.find_closest_point(points_org)]

        values_of_simplices_of_points = values[self.simplices[simplices_of_points]]

        data_points_transform = self.transform[simplices_of_points]
        t = data_points_transform[:, :self.dimension, :]
        r = data_points_transform[:, self.dimension, :]

        c = np.einsum('BNi,Bi ->BN', t, points - r)
        points_barycentric_coordinate = np.concatenate((c, (1 - np.sum(c, axis=1))[:, np.newaxis]), axis=1)

        values = np.einsum('ij,ij-> i', values_of_simplices_of_points, points_barycentric_coordinate)
        return values

    def evaluate(self, points, values=None, neigh_n=False):
        
        if values is None:
            values = self.grid_values

        simplices_of_points = self.find_simplex(points)
        out_simplices = np.where(simplices_of_points == -1)[0]
        values_of_simplices_of_points = values[self.simplices[simplices_of_points]]

        data_points_transform = self.transform[simplices_of_points]
        t = data_points_transform[:, :self.dimension, :]
        r = data_points_transform[:, self.dimension, :]

        c = np.einsum('BNi,Bi ->BN', t, points - r)
        points_barycentric_coordinate = np.concatenate((c, (1 - np.sum(c, axis=1))[:, np.newaxis]), axis=1)

        values_o = np.einsum('ij,ij-> i', values_of_simplices_of_points, points_barycentric_coordinate)


        for x in out_simplices:
            if neigh_n:
                z = points[x]
            else:
                z = self.proj_extrapolate(points[x])
            values_o[x] = self.evaluate_base(trunc(z, 10), values, neigh_n)

        return values_o

    def update_values(self, grid_values):
        self.grid_values = grid_values
        self.data_values = self.H.dot(self.grid_values)
        self.lat_coeffs = self.give_affine_coef(self.valid_simplices_centers)
        return

    def give_affine_coef(self, points):
        simplex = self.find_simplex_valid(points)
        temp = self.grid_values[self.simplices_sorted[simplex]][:, :, np.newaxis]
        coefs = np.matmul(self.final_T[simplex], temp)
        coefs = np.reshape(coefs.ravel(), (points.shape[0], self.dimension))

        return coefs

    def update_data_points(self, data_points, values):
        
        self.n_data_points, self.dimension = data_points.shape
        self.data_points = data_points
        self.data_values = values
        self.simplices_of_points = self.find_simplex_valid(data_points)

        self.grid_points_of_simplices_of_points = self.valid_simplices[self.simplices_of_points]

        data_points_transform = self.valid_transform[self.simplices_of_points]
        t = data_points_transform[:, :self.dimension, :]
        r = data_points_transform[:, self.dimension, :]

        c = np.einsum('BNi,Bi ->BN', t, self.data_points - r)
        self.data_points_barycentric_coordinate = np.concatenate((c, (1 - np.sum(c, axis=1))[:, np.newaxis]), axis=1)

        self.identity = False
        self.construct_forward_matrix()

        return


