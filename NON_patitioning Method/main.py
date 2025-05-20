import numpy as np
import time
from data import simulate_parameter, simulate_linear_sem_hz
from logger import *
import logging
import argparse

# import methods
from NOTEARS import notears_linear
from castle.metrics import MetricsDAG
from castle.algorithms import GOLEM, ICALiNGAM, DirectLiNGAM
from GOLEM_CIR import GOLEM_CIR
from DAG_GNN import DAG_GNN_main
from DAG_GNN_CIR import DAG_GNN_CIR_main

# ValueError: illegal value in 4-th argument of internal None

skeletons = ['alarm', 'andes', 'diabetes', 'link']
methods = ['GOLEM', 'GOLEM_CIR', 'DAG_GNN', 'DAG_GNN_CIR', 'NOTEARS', 'ICALiNGAM', 'DirectLiNGAM', 'GES']
parser = argparse.ArgumentParser()
parser.add_argument('--num_iters',
                    help='number of iterations;',
                    type=int,
                    default=1)
parser.add_argument('--skeleton',
                    help='use which skeleton;',
                    type=str,
                    choices=skeletons,
                    default='alarm')
parser.add_argument('--method',
                    help='use which method;',
                    type=str,
                    choices=methods,
                    default='ICALiNGAM')

args = parser.parse_args(args=sys.argv[1:])

num_iters = args.num_iters
skeleton = args.skeleton
method = args.method
tot_metrics = {}
output_dir = 'Results/{}/{}'.format(method, skeleton)
create_dir(output_dir)    # Create directory to save log files and outputs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
LogHelper.setup(log_path='{}/summary.log'.format(output_dir), level='INFO')
_logger = logging.getLogger(__name__)

for seed in range(num_iters):
    np.random.seed(seed)
    file = r'{}.txt'.format(skeleton)
    with open(file,encoding = 'utf-8') as f:
        B_true = np.loadtxt(f)
    W_true = simulate_parameter(B_true)
    if 'LiNGAM' in method: 
        dataset = simulate_linear_sem_hz(B=W_true, n=max(100, B_true.shape[0]+1), noise_type="uniform")
    else:
        dataset = simulate_linear_sem_hz(B=W_true, n=100, noise_type="uniform")

    st = time.time()

    # structure learning
    if method == 'GOLEM':
        alg = GOLEM(lambda_1=0.05, equal_variances=True, num_iter=3e+4)
        alg.learn(dataset)
        B_init = np.array(alg.causal_matrix)
        import torch
        alg = GOLEM(B_init=torch.from_numpy(B_init).double().clone().detach().requires_grad_(True), lambda_1=0.05, non_equal_variances=True, num_iter=4e+4)
        alg.learn(dataset)
        met = MetricsDAG(alg.causal_matrix, B_true)
    elif method == 'GOLEM_CIR': 
        B_est = GOLEM_CIR(dataset, W_true, 0.02, 5.0, 1.0)
        met = MetricsDAG(B_est!=0, B_true)
    elif method == 'DAG_GNN':
        G = DAG_GNN_main(dataset, B_true)
        est = G.copy()
        est[np.abs(est) < 0.3] = 0 
        est[np.abs(est) != 0] = 1 
        met = MetricsDAG(est, B_true)
    elif method == 'DAG_GNN_CIR':
        G = DAG_GNN_CIR_main(dataset, B_true)
        est = G.copy()
        est[np.abs(est) < 0.3] = 0 
        est[np.abs(est) != 0] = 1 
        met = MetricsDAG(est, B_true)
    elif method == 'NOTEARS':
        est = notears_linear(dataset, 0.02, 'l2')
        met = MetricsDAG(est!=0, B_true)
    elif method == 'ICALiNGAM':
        alg = ICALiNGAM()
        alg.learn(dataset)
        met = MetricsDAG(alg.causal_matrix, B_true)
    elif method == 'DirectLiNGAM':
        alg = DirectLiNGAM()
        alg.learn(dataset)
        met = MetricsDAG(alg.causal_matrix, B_true)

    metrics = met.metrics.copy()
    ed = time.time()
    metrics['time'] = ed-st
    for key in metrics.keys():
        if key not in tot_metrics.keys():
            tot_metrics[key] = 0
        tot_metrics[key] += metrics[key]
    _logger.info({'seed': seed})
    _logger.info(metrics)

for key in tot_metrics.keys():
    tot_metrics[key] /= num_iters
print(tot_metrics)
