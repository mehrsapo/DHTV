import numpy as np
from MyPloter import *
from MyDelaunay import MyDelaunay
from opt_solver import solve_admm
from matplotlib import pyplot as plt

data_points = np.loadtxt('face_data_points.txt')
values = np.loadtxt('face_data_values.txt')

reg_points = np.loadtxt('face_grid_points.txt')
reg_points = reg_points[np.lexsort((reg_points[:,0], reg_points[:,1]))]
tri_reg = MyDelaunay(data_points, values, grid_points=reg_points)
tri_reg.construct_forward_matrix()
tri_reg.construct_regularization_matrix()

mehrsa = np.loadtxt('mehrsa_face.txt')



tri_reg.update_values(mehrsa)
plot_with_gradient_map(tri_reg)