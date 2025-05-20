from castle.metrics import MetricsDAG
from castle.algorithms import GOLEM
import numpy as np
from torch.autograd import Variable as V
import torch
import time
from data import simulate_parameter, simulate_linear_sem_hz

if __name__ == '__main__':
    num_iters = 1
    tot_metrics = {}
    for seed in range(num_iters):
        np.random.seed(seed)
        # file = r'alarm.txt'
        # with open(file,encoding = 'utf-8') as f:
        #     B_true = np.loadtxt(f)
        file = r'skeleton_andes.csv'
        with open(file,encoding = 'utf-8') as f:
            B_true = np.loadtxt(f,delimiter=',')
        W_true = simulate_parameter(B_true)
        dataset = simulate_linear_sem_hz(B=W_true, n=100, noise_type="uniform")

        st = time.time()
        # structure learning using equal_variances loss as initialization
        alg = GOLEM(lambda_1=0.05, equal_variances=True, num_iter=3e+4)
        alg.learn(dataset)

        # calculate accuracy
        met = MetricsDAG(alg.causal_matrix, B_true)
        print(met.metrics)
        B_init = np.array(alg.causal_matrix)

        # structure learning using non_equal_variances loss
        alg = GOLEM(B_init=torch.tensor(B_init, dtype=torch.float64, requires_grad=True), lambda_1=0.05, non_equal_variances=True, num_iter=4e+4)
        alg.learn(dataset)

        # calculate accuracy
        met = MetricsDAG(alg.causal_matrix, B_true)
        print(met.metrics)

        metrics = met.metrics.copy()
        ed = time.time()
        metrics['time'] = ed-st
        for key in metrics.keys():
            if key not in tot_metrics.keys():
                tot_metrics[key] = 0
            tot_metrics[key] += metrics[key]

    for key in tot_metrics.keys():
        tot_metrics[key] /= num_iters
    print(tot_metrics)