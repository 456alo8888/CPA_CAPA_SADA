from castle.metrics import MetricsDAG
from castle.algorithms import ICALiNGAM
import numpy as np
import time
from data import simulate_parameter, simulate_linear_sem_hz
from logger import *
import logging

if __name__ == '__main__':
    num_iters = 1
    skeleton = 'alarm'
    tot_metrics = {}
    output_dir = 'Results/ICA-LiNGAM/{}'.format(skeleton)
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
        dataset = simulate_linear_sem_hz(B=W_true, n=224, noise_type="uniform")

        st = time.time()
        # structure learning
        alg = ICALiNGAM()
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
        _logger.info({'seed': seed})
        _logger.info(metrics)

    for key in tot_metrics.keys():
        tot_metrics[key] /= num_iters
    print(tot_metrics)