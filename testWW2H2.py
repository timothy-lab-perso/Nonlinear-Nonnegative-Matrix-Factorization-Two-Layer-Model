# === Alternating Optimization for W, W2, H2 with Synthetic Tests (Improved Initialization + Momentum + Dual Axis Plot) ===

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp
from scipy.linalg import svd

# --- Subgradient Computation Functions ---
def compute_subgradients_H2(W, W2, H2, X):
    Z = np.dot(W2, H2)
    A = np.maximum(Z, 0)
    error = np.dot(W, A) - X
    indicator = (Z > 0).astype(float)
    grad_H2 = 2 * np.dot(W2.T, np.dot(W.T, error) * indicator)
    return grad_H2

def compute_subgradients_W2(W, W2, H2, X):
    Z = np.dot(W2, H2)
    A = np.maximum(Z, 0)
    error = np.dot(W, A) - X
    indicator = (Z > 0).astype(float)
    grad_W2 = 2 * np.dot((np.dot(W.T, error) * indicator), H2.T)
    return grad_W2

# --- BCD Pre-Initialization ---
def pre_initialize_WH(X, r1, max_iter=10):
    n, m = X.shape
    W = np.abs(np.random.randn(n, r1))
    H = np.abs(np.random.randn(r1, m))
    normX = np.linalg.norm(X, 'fro')
    for it in range(max_iter):
        for j in range(m):
            H[:, j], _ = sp.nnls(W, X[:, j])
        for i in range(n):
            W[i, :], _ = sp.nnls(H.T, X[i, :])
        loss = np.linalg.norm(X - W @ H, 'fro') / normX
        print(f"[Init BCD {it+1}] Relative loss: {loss:.6f}")
    return W, H

# --- SVD Initialization of W2 and H2 ---
def initialize_W2_H2_from_H(H, r2):
    U, S, VT = svd(H, full_matrices=False)
    S_root = np.sqrt(S[:r2])
    W2 = U[:, :r2] @ np.diag(S_root)
    H2 = np.diag(S_root) @ VT[:r2, :]
    return W2, H2

# --- Subgradient Descent on W2, H2 ---
def subgradient_descent_W2_H2(
    X, W, W2_init, H2_init,
    learning_rate=0.01, num_iterations=1000,
    step_type="line_search", plot_frequency=50,
    reset_check_interval=500, use_momentum=False,
    momentum_beta=0.9
):
    W2 = W2_init.copy()
    H2 = H2_init.copy()
    loss_history = []
    grad_norm_history = []
    normX = np.linalg.norm(X, 'fro')

    v_W2 = np.zeros_like(W2)
    v_H2 = np.zeros_like(H2)

    for it in range(num_iterations):
        grad_W2 = compute_subgradients_W2(W, W2, H2, X)
        grad_H2 = compute_subgradients_H2(W, W2, H2, X)
        grad_norm = np.sqrt(np.linalg.norm(grad_W2)**2 + np.linalg.norm(grad_H2)**2)

        if step_type == "constant":
            step_size = learning_rate
        elif step_type == "1_over_k":
            step_size = learning_rate / (it + 1)
        elif step_type == "1_over_sqrt_k":
            step_size = learning_rate / np.sqrt(it + 1)
        elif step_type == "line_search":


            step_size = 1.0
            alpha = 0.5
            beta = 1e-7
            Z_current = np.maximum(np.dot(W2, H2), 0)
            loss_current = np.linalg.norm(X - np.dot(W, Z_current), 'fro') / normX
            for _ in range(20):
                W2_temp = W2 - step_size * grad_W2
                H2_temp = H2 - step_size * grad_H2
                Z_temp = np.maximum(np.dot(W2_temp, H2_temp), 0)
                loss_temp = np.linalg.norm(X - np.dot(W, Z_temp), 'fro') / normX
                if loss_temp <= loss_current - beta * step_size * grad_norm**2:
                    break
                step_size *= alpha
        else:
            step_size = learning_rate

        if use_momentum:
            v_W2 = momentum_beta * v_W2 - step_size * grad_W2
            v_H2 = momentum_beta * v_H2 - step_size * grad_H2
            W2 += v_W2
            H2 += v_H2
        else:
            W2 -= step_size * grad_W2
            H2 -= step_size * grad_H2

        if it % plot_frequency == 0:
            Z = np.maximum(np.dot(W2, H2), 0)
            rel_loss = np.linalg.norm(X - np.dot(W, Z), 'fro') / normX
            loss_history.append(rel_loss)
            grad_norm_history.append(grad_norm)

    return W2, H2, grad_W2, grad_H2, loss_history, list(range(0, num_iterations, plot_frequency)), grad_norm_history

# --- Alternating Descent ---
def alternating_descent_W_W2_H2(
    X, W_init, W2_init, H2_init,
    num_outer_iterations=50,
    inner_iterations=100,
    learning_rate=0.01,
    step_type="line_search",
    plot_frequency=50,
    use_momentum=False,
    momentum_beta=0.9,
    verbose=True
):
    W = W_init.copy()
    W2 = W2_init.copy()
    H2 = H2_init.copy()

    loss_history = []
    grad_norm_history = []
    normX = np.linalg.norm(X, 'fro')

    for outer_iter in range(num_outer_iterations):
        W2, H2, grad_W2, grad_H2, inner_loss_hist, _, grad_norm_hist = subgradient_descent_W2_H2(
            X, W, W2, H2,
            learning_rate=learning_rate,
            num_iterations=inner_iterations,
            step_type=step_type,
            plot_frequency=plot_frequency,
            reset_check_interval=inner_iterations + 1,
            use_momentum=use_momentum,
            momentum_beta=momentum_beta
        )

        loss_history.extend(inner_loss_hist)
        grad_norm_history.extend(grad_norm_hist)

        #NNLS change on W
        Z = np.maximum(np.dot(W2, H2), 0)
        for i in range(X.shape[0]):
            W[i, :], _ = sp.nnls(Z.T, X[i, :])

        if verbose:
            print(f"[Outer Iter {outer_iter+1}] Final relative loss: {loss_history[-1]:.6f}")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(loss_history, 'b-', label="Relative Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Relative Loss", color='b')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(grad_norm_history, 'r-', label="Gradient Norm", alpha=0.4)
    ax2.set_ylabel("Gradient Norm", color='r')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.suptitle("Loss and Gradient Norm over Iterations (log-log)")
    fig.tight_layout()
    plt.grid(True)
    plt.show()

    print(f"\n[INFO] Momentum: {use_momentum}, Beta: {momentum_beta}")

    return W, W2, H2, loss_history, grad_norm_history

# --- Main Synthetic Test ---
if __name__ == "__main__":
    np.random.seed(0)

    n, m = 20, 20
    r1, r2 = 10, 5

    W1 = np.random.randn(n, r2)
    H1 = np.random.randn(r2, r1)
    W2_true = np.random.randn(r1, r2)
    H2_true = np.random.randn(r2, m)

    W_true = np.maximum(W1 @ H1, 0)
    Z_true = np.maximum(W2_true @ H2_true, 0)
    X = W_true @ Z_true

    print("\n--- BCD Initialization Phase ---")
    W_init, H_mid = pre_initialize_WH(X, r1, max_iter=20)
    W2_init, H2_init = initialize_W2_H2_from_H(H_mid, r2)

    print("\n--- Starting Alternating Optimization ---")
    alternating_descent_W_W2_H2(
        X, W_init, W2_init, H2_init,
        num_outer_iterations=500,
        inner_iterations=5,
        learning_rate=0.1,
        step_type="line_search",
        plot_frequency=10,
        use_momentum=True,
        momentum_beta=0.999
    )
