import logging
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class GolemModel:
    """Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, n, d, CI_table, lambda_1, lambda_2, lambda_3, equal_variances=True,
                 seed=1, B_init=None):
        """Initialize self.

        Args:
            n (int): Number of samples.
            d (int): Number of nodes.
            lambda_1 (float): Coefficient of L1 penalty.
            lambda_2 (float): Coefficient of DAG penalty.
            lambda_3 (float): Coefficient of CI penalty.
            equal_variances (bool): Whether to assume equal noise variances
                for likelibood objective. Default: True.
            seed (int): Random seed. Default: 1.
            B_init (numpy.ndarray or None): [d, d] weighted matrix for
                initialization. Set to None to disable. Default: None.
        """
        self.n = n
        self.d = d
        self.CI_table = CI_table
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.equal_variances = equal_variances
        self.B_init = B_init

        self._build()
        self._init_session()

    def _init_session(self):
        """Initialize tensorflow session."""
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True
            )
        ))

    def _build(self):
        """Build tensorflow graph."""
        tf.compat.v1.reset_default_graph()

        # Placeholders and variables
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[self.n, self.d])
        self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        if self.B_init is not None:
            self.B = tf.Variable(tf.convert_to_tensor(self.B_init, tf.float32))
        else:
            self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        self.B = self._preprocess(self.B)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.CI_penalty = self._compute_CI_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h + self.lambda_3 * self.CI_penalty

        # Optimizer
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.score)
        self._logger.debug("Finished building tensorflow graph.")

    def _preprocess(self, B):
        """Set the diagonals of B to zero.

        Args:
            B (tf.Tensor): [d, d] weighted matrix.

        Returns:
            tf.Tensor: [d, d] weighted matrix.
        """
        return tf.linalg.set_diag(B, tf.zeros(B.shape[0], dtype=tf.float32))

    def _compute_likelihood(self):
        """Compute (negative log) likelihood in the linear Gaussian case.

        Returns:
            tf.Tensor: Likelihood term (scalar-valued).
        """
        if self.equal_variances:    # Assuming equal noise variances
            return 0.5 * self.d * tf.math.log(
                tf.square(
                    tf.linalg.norm(self.X - self.X @ self.B)
                )
            ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]
        else:    # Assuming non-equal noise variances
            return 0.5 * tf.math.reduce_sum(
                tf.math.log(
                    tf.math.reduce_sum(
                        tf.square(self.X - self.X @ self.B), axis=0
                    )
                )
            ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]

    def _compute_L1_penalty(self):
        """Compute L1 penalty.

        Returns:
            tf.Tensor: L1 penalty term (scalar-valued).
        """
        return tf.norm(self.B, ord=1)

    def _compute_h(self):
        """Compute DAG penalty.

        Returns:
            tf.Tensor: DAG penalty term (scalar-valued).
        """
        return tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d
    
    def _compute_CI_penalty(self):
        """Compute CI penalty.

        Returns:
            tf.Tensor: CI penalty term (scalar-valued).
        """
        # return tf.reduce_sum(tf.math.maximum(tf.zeros_like(self.CI_table, tf.float32), (self.CI_table * tf.math.abs(self.B) - self.CI_table * 0.3)))
        # return tf.reduce_sum(tf.math.log(1 + tf.math.exp(self.CI_table * tf.math.abs(self.B) - self.CI_table * 0.3)))
        return tf.reduce_sum((self.B * self.CI_table) * (self.B * self.CI_table),[0,1])
    
class GolemTrainer:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""
    _logger = logging.getLogger(__name__)

    def __init__(self, learning_rate=1e-3):
        """Initialize self.

        Args:
            learning_rate (float): Learning rate of Adam optimizer.
                Default: 1e-3.
        """
        self.learning_rate = learning_rate

    def train(self, model, X, num_iter, checkpoint_iter=None, output_dir=None):
        """Training and checkpointing.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.
            num_iter (int): Number of iterations for training.
            checkpoint_iter (int): Number of iterations between each checkpoint.
                Set to None to disable. Default: None.
            output_dir (str): Output directory to save training outputs. Default: None.

        Returns:
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        model.sess.run(tf.compat.v1.global_variables_initializer())

        self._logger.info("Started training for {} iterations.".format(num_iter))
        for i in range(0, int(num_iter) + 1):
            if i == 0:    # Do not train here, only perform evaluation
                score, likelihood, h, B_est = self.eval_iter(model, X)
            else:    # Train
                score, likelihood, h, B_est = self.train_iter(model, X)

            if checkpoint_iter is not None and i % checkpoint_iter == 0:
                self.train_checkpoint(i, score, likelihood, h, B_est, output_dir)

        return B_est

    def eval_iter(self, model, X):
        """Evaluation for one iteration. Do not train here.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        score, likelihood, h, B_est \
            = model.sess.run([model.score, model.likelihood, model.h, model.B],
                             feed_dict={model.X: X,
                                        model.lr: self.learning_rate})

        return score, likelihood, h, B_est

    def train_iter(self, model, X):
        """Training for one iteration.

        Args:
            model (GolemModel object): GolemModel.
            X (numpy.ndarray): [n, d] data matrix.

        Returns:
            float: value of score function.
            float: value of likelihood function.
            float: value of DAG penalty.
            numpy.ndarray: [d, d] estimated weighted matrix.
        """
        _, score, likelihood, h, B_est \
            = model.sess.run([model.train_op, model.score, model.likelihood, model.h, model.B],
                             feed_dict={model.X: X,
                                        model.lr: self.learning_rate})

        return score, likelihood, h, B_est
    
    def train_checkpoint(self, i, score, likelihood, h, B_est, output_dir):
        """Log and save intermediate results/outputs.

        Args:
            i (int): i-th iteration of training.
            score (float): value of score function.
            likelihood (float): value of likelihood function.
            h (float): value of DAG penalty.
            B_est (numpy.ndarray): [d, d] estimated weighted matrix.
            output_dir (str): Output directory to save training outputs.
        """
        print(
            "[Iter {}] score {:.3E}, likelihood {:.3E}, h {:.3E}".format(
                i, score, likelihood, h
            )
        )

import networkx as nx

def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))

def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres


def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B

def golem(X, CI_table, lambda_1, lambda_2, lambda_3, equal_variances=True,
          num_iter=1e+5, learning_rate=1e-3, seed=1,
          checkpoint_iter=5000, output_dir=None, B_init=None):
    """Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

    Args:
        X (numpy.ndarray): [n, d] data matrix.
        CI_table (np.ndarray): [d, d] Conditional Independence table
        lambda_1 (float): Coefficient of L1 penalty.
        lambda_2 (float): Coefficient of DAG penalty.
        lambda_3 (float): Coefficient of CI penalty.
        equal_variances (bool): Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter (int): Number of iterations for training.
        learning_rate (float): Learning rate of Adam optimizer. Default: 1e-3.
        seed (int): Random seed. Default: 1.
        checkpoint_iter (int): Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        output_dir (str): Output directory to save training outputs.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.

    Returns:
        numpy.ndarray: [d, d] estimated weighted matrix.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """
    # Center the data
    X = X - X.mean(axis=0, keepdims=True)

    # Set up model
    n, d = X.shape
    model = GolemModel(n, d, CI_table, lambda_1, lambda_2, lambda_3, equal_variances, seed, B_init)

    # Training
    trainer = GolemTrainer(learning_rate)
    B_est = trainer.train(model, X, num_iter, checkpoint_iter, output_dir)

    return B_est    # Not thresholded yet

from CI import prepare_CI_table
import numpy as np

def GOLEM_CIR(X, B_true, lambda_1, lambda_2, lambda_3):
    X -= np.mean(X, axis=0)
    # np.savetxt('data.csv', dataset.X, delimiter=',')
    CI_table = prepare_CI_table(X, alpha = 0.05)
    d = X.shape[1]

    ground_truth = np.ones([d, d]) - np.eye(d) - (B_true!=0)
    ground_truth = ground_truth * ground_truth.transpose()
    ground_truth = np.ones([d, d]) - np.eye(d) - ground_truth
    print('Type II error edges: ',np.sum(ground_truth * CI_table))
    
    B_init = golem(X, CI_table, lambda_1, lambda_2, lambda_3, equal_variances=True, num_iter=3e+4)
    B_est = golem(X, CI_table, lambda_1, lambda_2, lambda_3, equal_variances=False, num_iter=3e+4, B_init=B_init)
    B_processed = postprocess(B_est, 0.3)
    return B_processed

from data import simulate_parameter, simulate_linear_sem_hz
import time
from castle.metrics import MetricsDAG

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
        B_est = GOLEM_CIR(dataset, W_true, 0.02, 5.0, 0.0)

        # calculate accuracy
        met = MetricsDAG(B_est!=0, B_true)
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
