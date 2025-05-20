import numpy as np
import networkx as nx

# data simulation, simulate true causal dag and train_data.
def simulate_parameter(B, w_ranges=((-1.0, -0.5), (0.5, 1.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    S = 1
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem_hz(B, n, noise_type):
    """Simulate samples from linear SEM with specified type of noise.

    Args:
        B (numpy.ndarray): [d, d] weighted adjacency matrix of DAG.
        n (int): Number of samples.
        noise_type ('gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'): Type of noise.
        rs (numpy.random.RandomState): Random number generator.
            Default: np.random.RandomState(1).

    Returns:
        numpy.ndarray: [n, d] data matrix.
    """
    def _simulate_single_equation(X, B_i):
        """Simulate samples from linear SEM for the i-th node.

        Args:
            X (numpy.ndarray): [n, number of parents] data matrix.
            B_i (numpy.ndarray): [d,] weighted vector for the i-th node.

        Returns:
            numpy.ndarray: [n,] data matrix.
        """
        if noise_type == 'uniform':
            # Uniform noise
            N_i = np.random.uniform(low=-0.5, high=0.5, size=n)
        elif noise_type == 'gaussian_ev':
            # Gaussian noise with equal variances
            N_i = np.random.normal(scale=1.0, size=n)
        elif noise_type == 'gaussian_nv':
            # Gaussian noise with non-equal variances
            scale = np.random.uniform(low=1.0, high=2.0)
            N_i = np.random.normal(scale=scale, size=n)
        elif noise_type == 'exponential':
            # Exponential noise
            N_i = np.random.exponential(scale=1.0, size=n)
        elif noise_type == 'gumbel':
            # Gumbel noise
            N_i = np.random.gumbel(scale=1.0, size=n)
        else:
            raise ValueError("Unknown noise type.")
        return X @ B_i + N_i * np.random.uniform(low=0.5, high=1.0)

    d = B.shape[0]
    X = np.zeros([n, d])
    G = nx.DiGraph(B)
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    from sklearn.preprocessing import MinMaxScaler
    for i in ordered_vertices:
        parents = list(G.predecessors(i))
        if len(parents)!=0:
            X[:, i] = _simulate_single_equation(X[:, parents], B[parents, i])
            scaler = MinMaxScaler()
            scaler.fit(X[:, i].reshape(-1, 1))
            X[:, i] = scaler.transform(X[:, i].reshape(-1,1)).reshape(-1)
        else :
            X[:, i] = np.random.uniform(low=-0.5, high=0.5, size=n)
    return X