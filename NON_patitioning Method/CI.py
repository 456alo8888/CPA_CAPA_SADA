import numpy as np
import math
import scipy.stats as st
from itertools import combinations

def getCorr(data, _x, _y):     # data.shape: num_samples * num_nodes
    x = data[:, _x]
    y = data[:, _y]
    cov = np.mean(x * y) - np.mean(x) * np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    return cov/np.sqrt(var_x * var_y)
                      
#https://en.wikipedia.org/wiki/Partial_correlation
#By using dynamic programming
'''
Input:
    x : int
    y : int 
    Z : tuple
    data : data matrix( dim: num_samples * num_nodes )
    max_size : the max size of the conditional set
    
Output:
    isCI : Bool (True: x is independent with y given Z)

Parameter: 
    alpha : significance level (default 0.05/0.01)
'''
def PartialCorr_DP(data, max_size, x, y, Z, alpha):
    num_nodes = len(data)
    corr = np.zeros([num_nodes,num_nodes,max_size])
    vis = np.zeros([num_nodes,num_nodes,max_size],dtype=bool)      
    
    def getCorr_cond(x, y, z, k): # k表示当前即将处理到z集合中下标为k的数据
        if len(z) == 0:
            return getCorr(data, x, y)
        if vis[x][y][k] == True:
            return corr[x][y][k]
        if (k == len(Z)):
            return getCorr(data, x, y)
        vis[x][y][k] = True
        val_1 = getCorr_cond(x, Z[k], z, k+1)
        val_2 = getCorr_cond(Z[k], y, Z, k+1)
        val = getCorr_cond(x, y, Z, k+1)
        corr[x][y][k] = (val - val_1 * val_2) / (math.sqrt(1 - val_1 * val_1) * math.sqrt(1 - val_2 * val_2))
        #print('({},{},{}) {:.3f}'.format(i,j,k,corr[x][y][k]))
        return corr[x][y][k]

    pcc = getCorr_cond(x, y, Z, 0)
    zpcc = 0.5*math.log((1+pcc)/(1-pcc))
    A = math.sqrt(len(data) - len(Z) - 3) * math.fabs(zpcc)
    B = st.norm.ppf(1-alpha/2) # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)

    if A>B:
        return False
    else :
        return True
        
#https://en.wikipedia.org/wiki/Partial_correlation
#By using linear regression
'''
Input:
    x : int
    y : int 
    Z : tuple
    data : data matrix( dim: num_samples * num_nodes )
    
Output:
    isCI : Bool (True: x is independent with y given Z)

Parameter: 
    alpha : significance level (default 0.05/0.01)
'''
def PartialCorr_LR(data, x, y, Z, alpha): 
    data_x = data[:,x] # num_samples, 
    data_y = data[:,y] # num_samples, 
    data_Z = data[:,Z] # num_samples * |Z|
    num_samples = len(data)
    
    Z_nodes = len(Z) # length of Z
    if Z_nodes == 0 :
        pcc = getCorr(data, x, y)
    else :
        num_samples = len(data_Z) # number of data samples
        arr_one = (np.ones([num_samples]))
        data_Z = np.insert(data_Z, 0, arr_one, axis = 1) # insert an all-ones column in the left

        wx = np.linalg.lstsq(data_Z, data_x, rcond=None)[0] # wx is the answer of data_Z * X = data_x by using least square method
        wy = np.linalg.lstsq(data_Z, data_y, rcond=None)[0] # wy is the answer of data_Z * X = data_y by using least square method

        rx = data_x - data_Z @ wx # calc residual error of data_x
        ry = data_y - data_Z @ wy # calc residual error of data_y

        pcc = num_samples * (np.transpose(rx) @ ry) - np.sum(rx) * np.sum(ry)
        pcc /= math.sqrt(num_samples * (np.transpose(rx) @ rx) - np.sum(rx) * np.sum(rx))
        pcc /= math.sqrt(num_samples * (np.transpose(ry) @ ry) - np.sum(ry) * np.sum(ry))

    zpcc = 0.5*math.log((1+pcc)/(1-pcc))
    A = math.sqrt(num_samples - Z_nodes - 3) * math.fabs(zpcc)
    B = st.norm.ppf(1-alpha/2) # Inverse Cumulative Distribution Function of normal Gaussian (parameter : 1-alpha/2)

    if A>B:
        return False
    else :
        return True

def prepare_CI_table(data, alpha=0.05):
    num_nodes = data.shape[1]
    CI_table = np.zeros([num_nodes, num_nodes])
    for (x,y) in combinations(range(num_nodes),2):
        CI_table[x][y] = PartialCorr_LR(data, x, y, (), alpha)
        CI_table[y][x] = CI_table[x][y]
        # if CI_table[x][y]==False :
        #     for z in range(num_nodes):
        #         if z==x or z==y:
        #             continue
        #         if newCI_test(data_matrix, x, y, (z, ), 0.5)==True:
        #             CI_table[x][y] = True
        #             CI_table[y][x] = CI_table[x][y]
        #             break  
    return CI_table

if __name__ == '__main__':
    import time
    from data import simulate_parameter, simulate_linear_sem_hz

    num_iters = 10
    sum = 0
    for seed in range(num_iters):
        np.random.seed(seed)
        file = r'alarm.txt'
        with open(file,encoding = 'utf-8') as f:
            B_true = np.loadtxt(f)
        W_true = simulate_parameter(B_true)
        dataset = simulate_linear_sem_hz(B=W_true, n=100, noise_type="uniform")
        dataset -= np.mean(dataset, axis=0)
        # np.savetxt('data.csv', dataset.X, delimiter=',')
        CI_table = prepare_CI_table(dataset, alpha = 0.05)
        d = dataset.shape[1]

        ground_truth = np.ones([d, d]) - np.eye(d) - (B_true!=0)
        ground_truth = ground_truth * ground_truth.transpose()
        ground_truth = np.ones([d, d]) - np.eye(d) - ground_truth
        print('Type II error edges: ',np.sum(ground_truth * CI_table))
        sum += np.sum(ground_truth * CI_table)
    print(sum/num_iters)

