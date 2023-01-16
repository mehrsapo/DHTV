import argparse
import os
import pickle
import shutil

import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from data import Data
from rbf import * 
from NN_trainer import *

from DHTVPac.HTV_Delaunay import MyDelaunay
from DHTVPac.opt_solver import double_fista

np.random.seed(2022)

parser = argparse.ArgumentParser()
parser.add_argument('-data', choices=['powerplant', 'ale', 'housing', 'autompg'], type=str)
parser.add_argument('-method', choices=['dhtv', 'dhtv_irr', 'nn2', 'nn6', 'rbf', 'lr'], type=str)
parser.add_argument('-device', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], type=str)

args = parser.parse_args()
params = vars(args)

if params['data'] is None:
    params['data'] = 'ale'

if params['method'] is None:
    params['method'] = 'dhtv'

if params['device'] is None:
    params['device'] = 'cuda:3'

data_name = params['data']
method_name = params['method']
device = params['device']

train_per = 70
n_trials = 30
eps = 5e-2

folder_name = 'comp_results/' + data_name + '_' + method_name

if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
os.mkdir(folder_name)

if data_name == 'ale':
    data = pd.read_csv('data/mcs_ds_edited_iter_shuffled.csv')
    data = data.dropna()

    y = data.iloc[:, 4].values.astype(np.float64) 
    X = data.iloc[:, 0:4].values.astype(np.float64) 

    tri_random = pickle.load(open('random_tri/4d.pickle', 'rb'))
    tri_learn = pickle.load(open('random_tri/4d_learn_ale.pickle', 'rb'))


if data_name == 'powerplant':
    data = np.array(pd.read_excel('data/4d-data.xlsx'))

    X = data[:, 0:4]
    y = data[:, 4]

    tri_random = pickle.load(open('random_tri/4d.pickle', 'rb'))

    

if data_name == 'autompg': 
    data = pd.read_table('data/auto-mpg.data', delim_whitespace=True, header=None)
    data = data.replace('?', np.NaN)  # handle missing 
    data = data.dropna()

    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:-1].values

    tri_random = pickle.load(open('random_tri/7d.pickle', 'rb'))
    

if data_name == 'housing': 
    data = np.loadtxt('data/housing.txt', skiprows=0)

    X = data[:, 0:2]
    y = data[:, 2] / 1e3   # convert to k

    tri_random = pickle.load(open('random_tri/2d.pickle', 'rb'))



valid_mses_all = list()
train_mses_all = list()
test_mses_all = list ()
sparsity_all = list()
htv_all = list()
lip_all = list()
lip_ex_all = list()

for trial in range(n_trials): 
    print('trial:', trial)

    X_train, X_rem, y_train, y_rem = train_test_split(X, y,  test_size=(100-train_per)/100, random_state=2022+trial)
    X_test, X_valid, y_test, y_valid = train_test_split(X_rem, y_rem, test_size=0.5, random_state=2022+trial)

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_n = sc.transform(X_train)
    X_valid_n = sc.transform(X_valid)
    X_test_n = sc.transform(X_test)

    sc_y = StandardScaler()
    sc_y.fit(y_train.reshape(-1, 1))

    y_train_n = sc_y.transform(y_train.reshape(-1, 1)).ravel()
    y_valid_n = sc_y.transform(y_valid.reshape(-1, 1)).ravel()
    y_test_n = sc_y.transform(y_test.reshape(-1, 1)).ravel()

    data = Data()
    data.train['input'] = torch.from_numpy(X_train_n).float()
    data.train['values'] = torch.from_numpy(y_train_n).float()

    data.valid['input'] = torch.from_numpy(X_valid_n).float()
    data.valid['values'] = torch.from_numpy(y_valid_n).float()

    data.test['input'] = torch.from_numpy(X_test_n).float()
    data.test['values'] = torch.from_numpy(y_test_n).float()

    data.grid['input'] = torch.from_numpy(tri_random.grid_points).float()

    data.rpoints['input'] = torch.from_numpy(np.random.normal(0, 2, size=(10000, X.shape[1]))).float()

    print('train data size: ', X_train.shape, '\nvalidation data size: ', X_valid.shape, '\ntest data size: ', X_test.shape)

    if method_name == 'lr': 
        hyps = [0]

    if method_name == 'dhtv': 
        tri = MyDelaunay(X_train_n, y_train_n)
        tri.construct_forward_matrix()
        tri.construct_regularization_matrix()
        hyps = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    if method_name == 'dhtv_irr': 
        tri = tri_learn
        tri.update_data_points(X_train_n, y_train_n)
        print(tri.L.shape, tri.H.shape)
        hyps = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

    if method_name == 'rbf': 
        lmbdas =   [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        epss = [0.5, 0.7, 1, 3, 5]

    if method_name == 'nn2': 
        hyps =  [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        layers = 2
        neurons = 500
    
    if method_name == 'nn6': 
        hyps =  [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        layers = 6
        neurons = 200

    train_mses = list()
    test_mses = list()
    valid_mses= list()
    htv_vec = list()
    spar_vec = list()
    lip_vec = list()  # not for rbf
    lip_ex_vec = list()

    if method_name != 'rbf':
        for i in range(len(hyps)): 
            
            if method_name == 'lr': 
                reg = LinearRegression().fit(X_train_n, y_train_n)
                valid_val = reg.predict(X_valid_n)
                train_val = reg.predict(X_train_n)
                test_val = reg.predict(X_test_n)

                grid_vals = reg.predict(tri_random.grid_points)

                lip = np.linalg.norm(reg.coef_, 2)
                lip_vec.append(lip)

            if method_name == 'dhtv' or method_name == 'dhtv_irr':
                print('lmbda is ', hyps[i])
                if method_name == 'dhtv':
                    y_fista, _ = double_fista(tri.data_values, tri.H, tri.L, tri.lip_H, tri.lip_L, hyps[i], 1, 10000, device=device, verbose=False)
                if method_name == 'dhtv_irr':
                    y_fista, _ = double_fista(tri.data_values, tri.H, tri.L, tri.lip_H, tri.lip_L, hyps[i], 500, 300, device=device, verbose=True)

                fista_solution = y_fista.detach().cpu().numpy()

                valid_val = tri.evaluate(X_valid_n, fista_solution)
                train_val = tri.evaluate(X_train_n, fista_solution)
                test_val = tri.evaluate(X_test_n, fista_solution)

                sim = tri.find_simplex_valid(tri.valid_simplices_centers)
                grads = np.matmul(tri.final_T[sim], fista_solution[tri.simplices_sorted[sim]][:, :, None]).squeeze()
                grad_norms = np.linalg.norm(grads, 2, axis=1)
                lip_exact = np.max(grad_norms)
                lip_ex_vec.append(lip_exact)

                '''sim2 = tri.find_simplex_valid(data.rpoints['input'].cpu().numpy())
                sim2 = sim2[np.where(sim2 != -1)]
        
                grads = np.matmul(tri.final_T[sim2], fista_solution[tri.simplices_sorted[sim2]][:, :, None]).squeeze()
                grad_norms = np.linalg.norm(grads, 2, axis=1)
                lip_approx = np.max(grad_norms)
                
                lip_vec.append(lip_approx)'''

                grid_vals = tri.evaluate(tri_random.grid_points, fista_solution)
        

            if method_name == 'nn2' or method_name == 'nn6': 
                torch.set_grad_enabled(True)
                print('wd is ', hyps[i])
                nn_trainer = NNTrainer(data, num_epochs=1000, hidden=neurons, layer=layers, batch_size=1024, weight_decay = hyps[i], learning_rate=0.001, device=device, verbose=False)
                nn_trainer.train()

                valid_val = nn_trainer.net(data.valid['input'].to(device=nn_trainer.device, dtype=nn_trainer.net.dtype)).detach().cpu().numpy()
                train_val = nn_trainer.net(data.train['input'].to(device=nn_trainer.device, dtype=nn_trainer.net.dtype)).detach().cpu().numpy()
                test_val = nn_trainer.net(data.test['input'].to(device=nn_trainer.device, dtype=nn_trainer.net.dtype)).detach().cpu().numpy()

                grid_vals = nn_trainer.net(data.grid['input'].to(device=nn_trainer.device, dtype=nn_trainer.net.dtype)).detach().cpu().numpy()

                x = data.rpoints['input'].to(device=nn_trainer.device, dtype=nn_trainer.net.dtype).requires_grad_(requires_grad=True) 

                grads = list()
                for i, _ in enumerate(x):
                    input = x[i:i+1]
                    output = nn_trainer.net.forward(input)
                    grad = torch.autograd.grad(outputs=output, inputs=input, retain_graph=True)[0]
                    grads.append(grad.cpu().numpy().squeeze())
                  
                grads = np.array(grads)
                grad_norms = np.linalg.norm(grads, 2, axis=1)
                lip = np.max(grad_norms)
                lip_vec.append(lip)

            valid_mse = np.mean((y_valid.ravel() - sc_y.inverse_transform(valid_val.reshape(-1, 1)).ravel())**2)
            train_mse = np.mean((y_train.ravel() - sc_y.inverse_transform(train_val.reshape(-1, 1)).ravel())**2)
            test_mse = np.mean((y_test.ravel() - sc_y.inverse_transform(test_val.reshape(-1, 1)).ravel())**2)

            htv = np.sum(np.abs(tri_random.L.dot(grid_vals)))
            sparsity = np.where(np.abs(tri_random.L.dot(grid_vals)) < eps)[0].shape[0] / tri_random.L.shape[0]

            valid_mses.append(valid_mse)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            htv_vec.append(htv)
            spar_vec.append(sparsity)
            
            print('test mse ', test_mse)
            print('valid mse', valid_mse)
            
        valid_mses_all.append(valid_mses)
        train_mses_all.append(train_mses)
        test_mses_all.append(test_mses)
        htv_all.append(htv_vec)
        sparsity_all.append(spar_vec)
        lip_all.append(lip_vec)
        lip_ex_all.append(lip_ex_vec)


    if method_name == 'rbf':

        for i in range(len(lmbdas)):
            for j in range(len(epss)):
                rbf = RBF(data, eps=epss[j], lmbda=lmbdas[i])

                valid_val = rbf.evaluate(data.valid['input']).cpu().numpy()
                train_val = rbf.evaluate(data.train['input']).cpu().numpy()
                test_val = rbf.evaluate(data.test['input']).cpu().numpy()

                grid_vals = rbf.evaluate(data.grid['input']).cpu().numpy()

                valid_mse = np.mean((y_valid.ravel() - sc_y.inverse_transform(valid_val.reshape(-1, 1)).ravel())**2)
                train_mse = np.mean((y_train.ravel() - sc_y.inverse_transform(train_val.reshape(-1, 1)).ravel())**2)
                test_mse = np.mean((y_test.ravel() - sc_y.inverse_transform(test_val.reshape(-1, 1)).ravel())**2)

                htv = np.sum(np.abs(tri_random.L.dot(grid_vals)))
                sparsity = np.where(np.abs(tri_random.L.dot(grid_vals)) < eps)[0].shape[0] / tri_random.L.shape[0]
        

                valid_mses.append(valid_mse)
                train_mses.append(train_mse)
                test_mses.append(test_mse)
                htv_vec.append(htv)
                spar_vec.append(sparsity)

                print('test mse ', test_mse)
                print('valid mse', valid_mse)
                
    
        valid_mses_all.append(valid_mses)
        train_mses_all.append(train_mses)
        test_mses_all.append(test_mses)
        htv_all.append(htv_vec)
        sparsity_all.append(spar_vec)

                

np.savetxt(folder_name + '/test_mse'+str(train_per)+'.txt', test_mses_all)
np.savetxt(folder_name + '/valid_mse'+str(train_per)+'.txt', valid_mses_all)
np.savetxt(folder_name + '/train_mse'+str(train_per)+'.txt', train_mses_all)
np.savetxt(folder_name + '/htv'+str(train_per)+'.txt', htv_all)
np.savetxt(folder_name + '/sparsity'+str(train_per)+'.txt', sparsity_all)
if method_name != 'rbf': 
    np.savetxt(folder_name + '/lip'+str(train_per)+'.txt', lip_all)
if method_name == 'dhtv' or method_name == 'dhtv_irr':
    np.savetxt(folder_name + '/lip_exact'+str(train_per)+'.txt', lip_ex_all) 

