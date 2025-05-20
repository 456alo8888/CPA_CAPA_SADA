from castle.metrics import MetricsDAG
from castle.algorithms import GES
import numpy as np
import time
from data import simulate_parameter, simulate_linear_sem_hz
import hashlib
import sys
from causallearn.graph.SHD import SHD
from causallearn.utils.DAG2CPDAG import dag2cpdag
from causallearn.utils.TXT2GeneralGraph import txt2generalgraph
import unittest
from causallearn.search.ScoreBased.GES import ges

# https://arxiv.org/pdf/1802.01396.pdf
num_iters = 1
tot_metrics = {}
for seed in range(num_iters):
    np.random.seed(seed)
    file = r'andes.txt'
    with open(file,encoding = 'utf-8') as f:
        B_true = np.loadtxt(f)
    W_true = simulate_parameter(B_true)
    dataset = simulate_linear_sem_hz(B=W_true, n=100, noise_type="uniform")
    st = time.time()
    # structure learning
    # alg = GES(criterion='bic', method='scatter')
    # alg.learn(dataset)
    res_map = ges(dataset, score_func='local_score_BIC', maxP=1, parameters=None)
    # calculate accuracy
    print(res_map['G'].graph)
    est_graph = res_map['G'].graph.copy()
    est_graph = est_graph * (est_graph != -1) - est_graph * (est_graph == -1)
    print(est_graph)
    # for i in range(est_graph.shape[0]):
    #     for j in range(est_graph.shape[1]):
    #         if est_graph[i][j] == -1:
    #             est_graph[i][j] = (i>j)
    met = MetricsDAG(est_graph, B_true)
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