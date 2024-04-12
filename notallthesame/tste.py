"""
t-distributed Stochastic Triplet Embedding (t-STE)
"""
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class TSTE:
    def __init__(self, 
            learning_rate: float = 2.0, 
            max_iter: int = 1000, 
            conv_tol: float = 1e-7, 
            log_iter: Optional[int] = 10
        ) -> None:

        self.lr = learning_rate   # Learning rate
        self.max_iter = max_iter  # Maximum number of iterations
        self.tol = conv_tol       # Convergence tolerance
        self.log_iter = log_iter  # Log every `log_iter` iterations


    def run(self, 
        judgements: NDArray, 
        num_dims: int = 2, 
        reg_L2: float = 0,   # Lambda
        tdist_dof: Optional[int] = None,  # Alpha
        p_log: bool = True, 
        seed: Optional[int] = None
            ) -> Tuple[NDArray, float, int, float]:
        
        assert judgements.ndim == 2 and judgements.shape[1] == 3, "Judgements must be a 2D array with 3 columns"
        assert judgements.min() == 0, "Triplet indices must be non-negative and start with zero"

        self.num_judgements = judgements.shape[0]  # Number of judgements
        self.num_items = judgements.max() + 1  # Number of items to embed
        self.num_dims = num_dims  # Number of dimensions
        self.reg_L2 = reg_L2  # L2 regularization
        self.tdist_dof = tdist_dof if tdist_dof is not None else num_dims - 1  # Degrees of freedom for t-distribution
        self.p_log = p_log  # Log probabilities

        rng = np.random.default_rng(seed)
        X = rng.normal(loc=0.0, scale=0.0001, size=(self.num_items, self.num_dims))  # Randomly initialize embedding

        best_embed = X.copy()
        best_error = np.inf
        best_viol = np.inf

        lr = self.lr  # Learning rate
        error = np.inf  # Current error
        iter = 0  # Iteration counter
        no_incr = 0  # Number of iterations without improvement

        while iter < self.max_iter and no_incr < 5:
            last_error = error  # Last error

            error, grad, num_viol = self.loss(X, judgements)

            # Gradient update
            X = X - (lr / self.num_judgements * self.num_items) * grad

            # Update learning rate
            if last_error > error + self.tol:
                no_incr = 0
                lr = lr * 1.01
            else:
                no_incr += 1
                lr = lr * .5

            iter += 1

            num_constr = num_viol / self.num_judgements
            if self.log_iter and iter % self.log_iter == 0:
                print(f"Iteration: {iter}, error: {error:.4f}, number of constraints: {num_constr}")

            if error < best_error:
                best_embed = X.copy()
                best_error = error
                best_viol = num_viol

        return best_embed, best_error, iter, best_viol


    def loss(self, X: NDArray, judgements: NDArray) -> Tuple[float, NDArray, int]:
        X = np.reshape(X, (self.num_items, self.num_dims))  # Enforce shape

        # Pairwise distances
        S = np.sum(X ** 2, axis=1, keepdims=True).repeat(self.num_items, axis=1)
        D = S + S.T - 2 * (X @ X.T)

        # Student-t kernel
        J = 1 + D / self.tdist_dof
        K = J ** (-1. * (self.tdist_dof + 1) / 2)

        # Probabilities of judgements
        K1 = K[judgements[:, 0], judgements[:, 1]]
        K2 = K[judgements[:, 0], judgements[:, 2]]
        P = K1 / (K1 + K2)

        # Compute loss
        # minval = np.finfo(P.dtype).min
        # Q = np.log(np.maximum(P, minval)) if self.p_log else P
        Q = np.log(P) if self.p_log else P
        error = -np.sum(Q) + self.reg_L2 * np.sum(X ** 2)

        # Compute gradient
        grad = np.zeros_like(X)  # [self.num_items, self.num_dims]

        t_0 = judgements[:, 0]
        t_1 = judgements[:, 1]
        t_2 = judgements[:, 2]
        t_idx = np.hstack([t_0, t_1, t_2])

        J = 1 / J
        p = 1 - P
        if not self.p_log:
            p = P * p
        j_1 = p * J[t_0, t_1]
        j_2 = p * J[t_0, t_2]

        for i in range(self.num_dims):
            x_0 = X[t_0, i]
            x_1 = X[t_1, i]
            x_2 = X[t_2, i]

            x_derivative = np.hstack([
                 j_1 * (x_0 - x_1) - j_2 * (x_0 - x_2),
                -j_1 * (x_0 - x_1),
                 j_2 * (x_0 - x_2)
            ]) * -(self.tdist_dof + 1) / self.tdist_dof

            for j in range(self.num_items):  # For each item
                t_mask = t_idx == j  # Judgements with jth item
                grad[j, i] = np.sum(x_derivative[t_mask])

        # Regularize gradient
        grad = -grad + 2 * self.reg_L2 * X

        # Number of violated constraints
        num_viol = np.sum(D[t_0, t_1] > D[t_0, t_2])

        return error, grad, num_viol
