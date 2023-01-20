import pickle

import numpy as np

# generate grid for comparison
from DHTVPac.HTV_Delaunay import MyDelaunay

learn = False
plot = False

np.random.seed(2022)

if learn:   
    n_g = 40
    d = 4
    data_name = 'ale'

elif plot:
    n_g = 10000
    d = 2
    data_name = ''
else: 
    n_g = 200
    d = 2

if plot:
    HTV_control_grid = np.random.uniform(-0.5, high=1.5, size=(n_g, d))
 
elif learn:
    HTV_control_grid = np.random.normal(0, 3, size=(n_g, d))
else:
    HTV_control_grid = np.random.normal(0, 1, size=(n_g, d))

HTV_control_grid_values = np.zeros((n_g, ))
tri_htv = MyDelaunay(HTV_control_grid, HTV_control_grid_values)
tri_htv.construct_forward_matrix()
tri_htv.construct_regularization_matrix()

if learn:
    pickle.dump(tri_htv, open('random_tri/' + str(d) + 'd_learn_'+data_name+'.pickle', 'wb'))
elif plot:
    pickle.dump(tri_htv, open('random_tri/' + str(d) + 'd_plot.pickle', 'wb'))
else:
    pickle.dump(tri_htv, open('random_tri/' + str(d) + 'd.pickle', 'wb'))