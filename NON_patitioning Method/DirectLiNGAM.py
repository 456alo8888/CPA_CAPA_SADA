from castle.metrics import MetricsDAG
from castle.algorithms import DirectLiNGAM
import numpy as np
import time 
from data import simulate_parameter, simulate_linear_sem_hz

if __name__ == '__main__':
    num_iters = 3
    tot_metrics = {}
    for seed in range(1, num_iters+1):
        np.random.seed(seed)
        # file = r'alarm.txt'
        # with open(file,encoding = 'utf-8') as f:
        #     B_true = np.loadtxt(f)
        file = r'skeleton_andes.csv'
        with open(file,encoding = 'utf-8') as f:
            B_true = np.loadtxt(f,delimiter=',')
        W_true = simulate_parameter(B_true)
        dataset = simulate_linear_sem_hz(B=W_true, n=224, noise_type="uniform")
        
        # ValueError: You are using LassoLarsIC in the case where the number of samples is smaller than the number of features. In this setting, getting a good estimate for the variance of the noise is not possible. Provide an estimate of the noise variance in the constructor.
        st = time.time()
        # structure learning
        alg = DirectLiNGAM()
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