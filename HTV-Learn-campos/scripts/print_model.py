#!/usr/bin/env python3

import argparse
import json

from htvlearn.master_project import MasterProject
from htvlearn.nn_manager import NNManager
from htvlearn.rbf_manager import RBFManager
from htvlearn.rbf import RBF
from htvlearn.lattice import Lattice
from htvlearn.htv_manager import HTVManager
from htvlearn.htv_utils import (
    compute_snr,
    get_sparsity,
    silence_stdout
)


def print_model(args):
    """
    Args:
        args: arguments from argparser
    """
    ckpt = MasterProject.get_loaded_ckpt(args.ckpt_filename)
    params = ckpt['params']
    params['log_dir'] = '/'.join(args.ckpt_filename.split('/')[:-2])
    params['data']['log_dir'] = params['log_dir']
    params['restore'] = True

    print('\nLoading parameters from checkpoint : ',
          args.ckpt_filename,
          sep='\n')
    print('\nParameters : ', params, sep='\n')

    with silence_stdout():

        htv = None
        if params['method'] == 'neural_net':
            manager = NNManager(params, log=False)
            if ckpt['htv_log']:
                _, htv_model = NNManager.read_htv_log(ckpt['htv_log'])
                htv = htv_model[-1]

        elif params['method'] == 'rbf':
            params['device'] = 'cpu'
            manager = RBFManager(params, log=False)
            if ckpt['htv_log']:
                htv = RBFManager.read_htv_log(ckpt['htv_log'])

        elif params['method'] == 'htv':
            params['device'] = 'cpu'
            manager = HTVManager(params, log=False)
            if ckpt['htv_log']:
                htv = manager.read_htv_log(ckpt['htv_log'])[-1]

        test_mse, _ = manager.evaluate_results(mode='test')
        train_mse, _ = manager.evaluate_results(mode='train')

        data_obj = manager.data
        test_snr = compute_snr(data_obj.test['values'], test_mse)
        train_snr = compute_snr(data_obj.train['values'], train_mse)
        if params['method'] == "htv":
            X_mat = ckpt['lattice']['final']['X_mat']
            C_mat = ckpt['lattice']['final']['C_mat']
            lattice_obj = Lattice(X_mat=X_mat, C_mat=C_mat)
        else:
            lattice_obj = manager.evaluate_lattice()
        percentage_nonzero = get_sparsity(lattice_obj)

    print('\nTrain mse : {:.2E}'.format(train_mse))
    print('Test mse  : {:.2E}'.format(test_mse))
    print('Train snr : {:.2f} dB'.format(train_snr))
    print('Test snr  : {:.2f} dB'.format(test_snr))
    print('% nonzero : {:.2f}'.format(percentage_nonzero))

    if htv is not None:
        if isinstance(htv, dict):
            for key, val in htv.items():
                htv[key] = float('{:.2f}'.format(val[0]))
            print('HTV\t  :', json.dumps(htv, indent=4, sort_keys=False))
        else:
            print('HTV\t  : {:.2f}'.format(htv))

    print('Exact HTV : {:.2f}'.format(ckpt['exact_htv']))
    if params['method'] == 'rbf':
        print('sigma : {:.2E}'.format(
            RBF.get_sigma_from_eps(params["rbf"]["eps"])))


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Load parameters from checkpoint file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('ckpt_filename',
                        metavar='ckpt_filename [STR]',
                        type=str,
                        help='Checkpoint where model is saved.')
    args = parser.parse_args()

    print_model(args)
