import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import os
import numpy as np
from numpy.linalg import norm
from scipy.linalg import qr
from joblib import Parallel, delayed
from itertools import product
import scipy.optimize as sp
import math

def compute_subgradients_H2(W, W_2, H_2, X):
    Z_2 = np.dot(W_2, H_2)
    max_Z_2 = np.maximum(Z_2, 0)
    error = np.dot(W, max_Z_2) - X
    indicator = (Z_2 > 0).astype(float)
    grad_H_2 = 2 * np.dot(W_2.T, (indicator * np.dot(W.T, error)))
    return grad_H_2

def compute_subgradients_W2(W, W_2, H_2, X):
    Z_2 = np.dot(W_2, H_2)
    ReLU_Z_2 = np.maximum(Z_2, 0)
    indicator = (Z_2 > 0).astype(float)
    error = np.dot(W, ReLU_Z_2) - X
    grad_W_2 = 2 * np.dot((np.dot(W.T, error) * indicator), H_2.T)
    return grad_W_2

def subgradient_descent_W2_H2(X, W, W_2_init, H_2_init,
                              learning_rate, num_iterations, step_type,
                              plot_frequency, reset_check_interval,
                              use_momentum=False, momentum_beta=0.9):
    """
    Perform subgradient descent on W_2 and H_2 with various step size strategies,
    optionally using momentum.
    """
    W_2 = W_2_init.copy()
    H_2 = H_2_init.copy()

    approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
    frob_norm_X = np.linalg.norm(X, 'fro')
    loss = np.linalg.norm(approx_X - X, 'fro') / frob_norm_X

    loss_history = [np.log(loss)]
    iteration_numbers = [0]

    grad_W2 = compute_subgradients_W2(W, W_2, H_2, X)
    grad_H2 = compute_subgradients_H2(W, W_2, H_2, X)

    grad_norm = np.sqrt(np.linalg.norm(grad_W2)**2 + np.linalg.norm(grad_H2)**2)

    grad_norm_history = [np.log(grad_norm)] ################""

    # Initialize momentum variables if enabled
    if use_momentum:
        v_W2 = np.zeros_like(W_2)
        v_H2 = np.zeros_like(H_2)



    for iteration in range(num_iterations):
        grad_W2 = compute_subgradients_W2(W, W_2, H_2, X)
        grad_H2 = compute_subgradients_H2(W, W_2, H_2, X)

        grad_norm = np.sqrt(np.linalg.norm(grad_W2)**2 + np.linalg.norm(grad_H2)**2)

        # Compute step size according to chosen step_type
        if step_type == "constant":
            step_size = learning_rate
        elif step_type == "1_over_k":
            step_size = learning_rate / (iteration + 1)
        elif step_type == "1_over_sqrt_k":
            step_size = learning_rate / np.sqrt(iteration + 1)
        elif step_type == "exponential":
            lambda_ = 0.95
            step_size = learning_rate * (lambda_ ** iteration)
        elif step_type == "polyak":
            approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
            error = X - approx_X
            if grad_norm != 0:
                step_size = np.linalg.norm(error) / (grad_norm * 4)
            else:
                step_size = learning_rate
        elif step_type == "line_search":
            # Backtracking line search along negative gradient direction
            alpha = 0.5 
            beta = 0.00001
            gamma=0.5
            
            approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
            loss = np.linalg.norm(approx_X - X, 'fro')
            #step_size = gamma * np.linalg.norm(approx_X,'fro') / grad_norm
            step_size=1
            #print(step_size)

            # Try decreasing step size until loss improves
            max_iter = 25
            for k in range(max_iter):
                W_2_temp = W_2 - step_size * grad_W2
                H_2_temp = H_2 - step_size * grad_H2
                approx_temp = np.dot(W, np.maximum(np.dot(W_2_temp, H_2_temp), 0))
                new_loss = np.linalg.norm(approx_temp - X, 'fro')
                if new_loss <= loss - beta * step_size * (grad_norm)**2:
                    break
                step_size *= alpha
                if k == max_iter-1:
                    print("Search failed")
        else:
            raise ValueError(f"Unknown step_type: {step_type}")

        # Update parameters with (optional) momentum
        if use_momentum:
            v_W2 = momentum_beta * v_W2 - step_size * grad_W2
            v_H2 = momentum_beta * v_H2 - step_size * grad_H2
            W_2 += v_W2
            H_2 += v_H2
        else:
            W_2 -= step_size * grad_W2
            H_2 -= step_size * grad_H2

        # Logging and progress display
        if iteration % plot_frequency == 0 or iteration == num_iterations - 1:
            approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
            frob_norm_X = np.linalg.norm(X, 'fro')
            loss = np.linalg.norm(approx_X - X, 'fro') / frob_norm_X
            if np.isnan(loss):
                print(f"Skipping iteration {iteration} due to NaN loss.")
                break
            if loss <= 0:
                raise ValueError("Loss is non-positive, cannot compute log")
            loss_history.append(np.log(loss))
            iteration_numbers.append(iteration)
            grad_norm_history.append(np.log(grad_norm))

            print(f"Iter {iteration}, log(Loss): {np.log(loss):.4f}, Step: {step_size:.3e}, Grad norm: {grad_norm:.3e}")

            # Check stagnation and reset logic for decaying step sizes
            if iteration >= reset_check_interval:
                current_idx = len(loss_history) - 1
                steps_ago_idx = current_idx - (reset_check_interval // plot_frequency)
                if steps_ago_idx >= 0:
                    if np.isclose(loss_history[current_idx], loss_history[steps_ago_idx], atol=1e-2):
                        print(f"No progress in log loss over last {reset_check_interval} iterations. Resetting step size.")
                        if step_type in ["1_over_k", "1_over_sqrt_k", "exponential"]:
                            iteration = 0  # Reset effective iteration count for decay

    grad_W2 = compute_subgradients_W2(W, W_2, H_2, X)
    grad_H2 = compute_subgradients_H2(W, W_2, H_2, X)

    return W_2, H_2, grad_W2, grad_H2, loss_history, iteration_numbers, grad_norm_history


#relu_nmd not working
def relu_nmd_init(X, W, r, alpha_init=1.0, alpha_max=1.5, delta_bar=0.9, mu_init=0.05, 
                  maxit=50, tol=1e-4, display=False, check_unbounded=True):
    """
    ReLU-based NMD initialization for W2 and H2.

    Parameters:
        X (ndarray): Input data matrix of shape (n, m)
        W (ndarray): First-layer factor matrix of shape (n, p)
        r (int): Target inner dimension
        alpha_init (float): Initial extrapolation weight
        alpha_max (float): Max extrapolation weight
        delta_bar (float): Tolerance for adapting alpha
        mu_init (float): Step for adapting alpha
        maxit (int): Max number of iterations
        tol (float): Tolerance for stopping
        display (bool): Whether to print debug info
        check_unbounded (bool): Whether to warn on unbounded growth

    Returns:
        W_2 (ndarray): Initialized W2 of shape (p, r)
        H_2 (ndarray): Initialized H2 of shape (r, m)
    """
    n, m = X.shape
    p = W.shape[1]

    # Precompute
    normX = np.linalg.norm(X, 'fro')
    idx = (X == 0)  # boolean mask for zero entries in X

    # Initialize Z, W_2, H_2
    Theta = W.T @ X  # shape (p, m), basic approximation
    W_2 = np.random.randn(p, r)
    H_2 = np.random.randn(r, m)
    Z = np.copy(X)
    S = np.linalg.norm(Z - W @ np.maximum(W_2 @ H_2, 0), 'fro')

    # Setup for extrapolation
    alpha = alpha_init
    mu = mu_init

    for i in range(maxit):
        # Indirect extrapolation step
        Theta_current = W_2 @ H_2
        Z_alpha = alpha * Z + (1 - alpha) * W @ np.maximum(Theta_current, 0)

        # QR decomposition to orthogonalize
        Q, _ = np.linalg.qr(Z_alpha @ H_2.T)
        W_2_new = Q[:, :r]
        H_2_new = W_2_new.T @ Z_alpha
        Theta_new = W_2_new @ H_2_new

        # Check for unbounded entries in approximation of X
        approx_X = W @ np.maximum(Theta_new, 0)
        if check_unbounded and np.max(np.abs(approx_X[idx])) > 1e10:
            if display:
                print("⚠️ Warning: Unbounded growth in zero entries of X")

        # Update Z
        Z_new = np.minimum(0, approx_X * idx) + X

        # Residual
        S_new = np.linalg.norm(Z_new - approx_X, 'fro')
        res_ratio = S_new / S

        if res_ratio < 1:
            # Accept update
            W_2 = W_2_new
            H_2 = H_2_new
            Z = Z_new
            Theta_current = Theta_new
            S = S_new
            if res_ratio > delta_bar:
                mu = max(mu, 0.25 * (alpha - 1))
                alpha = min(alpha + mu, alpha_max)
                if alpha == alpha_max:
                    alpha = 1.0
        else:
            alpha = 1.0  # reset

        if display:
            print(f"Iter {i+1:02d}: rel. residual = {S/normX:.4e}, alpha = {alpha:.3f}")

        # Stopping condition
        if S / normX < tol:
            if display:
                print(f"✅ Converged at iteration {i+1}, rel. error < {tol}")
            break

    return W_2, H_2

def initialize_W2_H2_svd_pinv(p, r, m, W_true=None, H_true=None):
        # Approximate initialization via SVD of X = W * ReLU(W_2_true * H_2_true)
        # We approximate ReLU(W_2_true H_2_true) by X_hat = W_2_init * H_2_init
        # Compute ReLU(Z_true) = ReLU(W_2_true * H_2_true)
        # Since we don't have direct access to Z_true here, approximate from X and W
        # Solve for M: X ≈ W M, with M ≈ ReLU(W_2 H_2)
        # We'll do SVD on W^T X to get warm start
        # This is a heuristic initialization
        
        # Compute pseudo-inverse of W
        W_pseudo_inv = np.linalg.pinv(W)
        M = np.dot(W_pseudo_inv, X)
        # Project M to non-negative part (ReLU)
        M = np.maximum(M, 0)
        U, Sigma, VT = svd(M, full_matrices=False)
        Sigma_r = np.diag(np.sqrt(Sigma[:r]))
        W_2_init = np.dot(U[:, :r], Sigma_r)
        H_2_init = np.dot(Sigma_r, VT[:r, :])
        return W_2_init,H_2_init


def initialize_W2_H2_svd_nnls(r, X, W, W_true=None, H_true=None):
    """
    Initialize W_2 and H_2 such that X ≈ W @ ReLU(W_2 @ H_2)
    Step 1: Solve NNLS W @ Theta ≈ X to get Theta ≥ 0
    Step 2: Apply SVD on Theta to get W_2 and H_2 such that Theta ≈ W_2 @ H_2
    """
    n, m = X.shape
    p = W.shape[1]

    # Step 1: NNLS to estimate Theta ≥ 0 such that W @ Theta ≈ X
    Theta_est = np.zeros((p, m))
    for j in range(m):
        Theta_est[:, j], _ = sp.nnls(W, X[:, j],maxiter =math.floor(1/10*n))  # W: (n, p), X[:, j]: (n,)        U, Sigma, VT = svd(H_est, full_matrices=False)
        
     # Step 2: SVD to factor Theta ≈ W_2 @ H_2
    U, S, VT = svd(Theta_est, full_matrices=False)
    W_2 = U[:, :r] @ np.diag(np.sqrt(S[:r]))  # Shape: (p, r)
    H_2 = np.diag(np.sqrt(S[:r])) @ VT[:r, :]  # Shape: (r, m)

    return W_2, H_2




def nnls_bcd_init(X, W, r, max_iter=100, tol=1e-4, verbose=True):
    """
    NNLS + Block Coordinate Descent to find W2, H2
    such that: X ≈ W @ relu(W2 @ H2)
    
    Input:
        X: (m x n)
        W: (m x k)
        r: rank (latent dimension) of H2
    Output:
        W2: (k x r)
        H2: (r x n)
    """
    m, n = X.shape
    k = W.shape[1]

    # === Step 1: Initial ReLU activation matrix H ===
    H = np.zeros((k, n))
    for j in range(n):
        H[:, j], _ = sp.nnls(W, X[:, j])

    # === Step 2: Initialize W2, H2 ===
    W2 = np.abs(np.random.randn(k, r))
    H2 = np.abs(np.random.randn(r, n))

    normX = np.linalg.norm(X, 'fro')
    errors = []

    for it in range(max_iter):
        # Step A: Compute Z = ReLU(W2 @ H2)
        Theta = W2 @ H2
        Z = np.maximum(0, Theta)

        # Step B: Fit W via NNLS: solve W H ≈ X
        for j in range(n):
            H[:, j], _ = sp.nnls(W, X[:, j])

        # Step C: Fit W2 via NNLS: W2 @ H2 ≈ H
        for i in range(k):
            W2[i, :], _ = sp.nnls(H2.T, H[i, :])

        # Step D: Fit H2 via NNLS: W2 @ H2 ≈ H
        for j in range(n):
            H2[:, j], _ = sp.nnls(W2, H[:, j])

        # Compute current approximation error
        relu_out = np.maximum(0, W2 @ H2svd)
        X_hat = W @ relu_out
        err = np.linalg.norm(X - X_hat, 'fro') / normX
        errors.append(err)

        if verbose:
            print(f"[{it+1:03d}] rel.error: {err:.6f}")

        if err < tol:
            break

    return W2, H2



def initialize_W2_H2(p, r, m, X, W, W_true=None, H_true=None, init_type="random"):
    """
    Initialize W_2 and H_2 using different strategies:
    - "random": uniform random [-1,1]
    - "noisy_true": true values plus small uniform noise
    - "svd-pinv": SVD warm start for ReLU(W_2_true H_2_true) as an approx of W^-1 * X
    - "relu_nmd": Placeholder for ReLU-NMD initializer (to be added)
    """
    if init_type == "random":
        W_2_init = np.random.uniform(-1, 1, size=(p, r))
        H_2_init = np.random.uniform(-1, 1, size=(r, m))
    elif init_type == "random_gaussian":
        W_2_init = np.random.randn(p, r)
        H_2_init = np.random.randn(r, m)
    elif init_type == "noisy_true":
        if W_true is None or H_true is None:
            raise ValueError("True factors must be provided for noisy_true init")
        W_2_init = W_true + np.random.uniform(-0.01, 0.01, size=(p, r))
        H_2_init = H_true + np.random.uniform(-0.01, 0.01, size=(r, m))
    elif init_type == "svd_pinv" or init_type == "svd_pinv_noisy":
        W_2_init,H_2_init = initialize_W2_H2_svd_pinv(p, r, m, W_true, H_true)
        if init_type == "svd_pinv_noisy":
            W_2_init += np.random.uniform(-0.1, 0.1, size=(p, r))
            H_2_init += np.random.uniform(-0.1, 0.1, size=(r, m))
    elif init_type == "svd_nnls" or init_type == "svd_nnls_noisy":
        W_2_init,H_2_init = initialize_W2_H2_svd_nnls(r, X, W, W_true, H_true)
        if init_type == "svd_nnls_noisy":
            W_2_init += np.random.uniform(-0.01, 0.01, size=(p, r))
            H_2_init += np.random.uniform(-0.01, 0.01, size=(r, m))
    elif init_type == "bcd_nnls":
        W_2_init,H_2_init = nnls_bcd_init(X, W, r)
    elif init_type == "relu_nmd":
         W_2_init,H_2_init = initialize_W2_H2_svd_nnls(r, X, W, W_true, H_true)
         #not workingggg
         W_2_init, H_2_init = relu_nmd_init(X, W, r, display=True)
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")
    return W_2_init, H_2_init

def fine_grained_loss_scan_W2H2(X, W, W2, H2, grad_W2, grad_H2, granularity=1e-15, range=1e-8, verbose=False):
    
    y_vals = np.arange(-range, range + granularity, granularity)
    losses = []
    search = []
    grad_norm = np.sqrt(np.linalg.norm(grad_W2)**2 + np.linalg.norm(grad_H2)**2)


    approx = W @ np.maximum(W2@ H2, 0)
    loss = np.linalg.norm(X - approx, 'fro')

    for delta in y_vals:
        H2_perturbed = H2 - delta*grad_H2
        W2_perturbed = W2 - delta*grad_W2
        approx = W @ np.maximum(W2_perturbed @ H2_perturbed, 0)
        new_loss = np.linalg.norm(X - approx, 'fro')
        losses.append(np.log(new_loss))

        search.append(np.log(loss - 1e-3 *delta * (grad_norm)**2))


        if verbose:
            print(f"Perturb H2[{i},{j}] by {delta:.3f} → loss = {loss:.6f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(y_vals, losses)
    plt.plot(y_vals, search)
    plt.title(f"Log Loss vs (- Step size * grad)")
    plt.xlabel(f"Perturbation (delta)")
    plt.ylabel("Loss (Frobenius norm squared)")
    plt.grid(True)
    plt.show()
    return

def run_multiple_initializations_W2_H2(
    X, W, W_true, H_true,
    rate, iterations, plot_frequency, n_init, step_type,
    reset_check_interval, use_momentum=False, momentum_beta=0.9,
    init_type="random", title_prefix="", output_dir="."
):
    """
    Run subgradient descent multiple times with different initializations,
    plot and save log loss and gradient norm curves.
    """

    p, r = W.shape[1], H_true.shape[0]
    print(2)
    m = H_true.shape[1]

    # Prepare plots: one for loss, one for grad norm
    plt.figure(figsize=(12, 6))
    plt_loss = plt.figure(figsize=(10, 6))
    ax_loss = plt_loss.add_subplot(1, 1, 1)
    plt_grad = plt.figure(figsize=(10, 6))
    ax_grad = plt_grad.add_subplot(1, 1, 1)

    # Create output directory if needed
    #os.makedirs(output_dir, exist_ok=True)

    for i in range(n_init):
        print(f"\n--- Run {i + 1}/{n_init} with step_type='{step_type}', init='{init_type}' ---")
        W_2_init, H_2_init = initialize_W2_H2(p, r, m, X, W, W_true, H_true, init_type)

        W_2_opt, H_2_opt, grad_W2_opt, grad_H2_opt, loss_history, iteration_numbers, grad_norm_history = subgradient_descent_W2_H2(
            X, W, W_2_init, H_2_init,
            learning_rate=rate,
            num_iterations=iterations,
            step_type=step_type,
            plot_frequency=plot_frequency,
            reset_check_interval=reset_check_interval,
            use_momentum=use_momentum,
            momentum_beta=momentum_beta
        )

    

        #label = f"Run {i+1}"

        ax_loss.plot(iteration_numbers, loss_history)
        ax_grad.plot(iteration_numbers, grad_norm_history)


    # Titles and labels with full parameters info
    base_title = (
        f"{title_prefix} "
        f"Init={init_type}, StepType={step_type}, "
        f"LR={rate}, Iter={iterations}, "
        "\n"
        f"Momentum={'On' if use_momentum else 'Off'}, "
        f"Beta={momentum_beta:.2f}, Runs={n_init}"
    )

    ax_loss.set_title(base_title + " - Log Loss vs Iterations")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Log Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_grad.set_title(base_title + " - Log Gradient Norm vs Iterations")
    ax_grad.set_xlabel("Iteration")
    ax_grad.set_ylabel("Log Gradient Norm")
    ax_grad.legend()
    ax_grad.grid(True)

    #plot variations of the loss over a variations of step sizes
    fine_grained_loss_scan_W2H2(X, W, W_2_opt, H_2_opt, grad_W2_opt, grad_H2_opt, granularity=1e-8, range=1e-4, verbose=False)

    # Save plots as PDFs
   # loss_pdf = os.path.join(output_dir, f"{title_prefix}_loss_{init_type}_{step_type}_runs{n_init}.pdf")
    #grad_pdf = os.path.join(output_dir, f"{title_prefix}_gradnorm_{init_type}_{step_type}_runs{n_init}.pdf")
   # plt_loss.savefig(loss_pdf)
    #plt_grad.savefig(grad_pdf)

    print(f"\nSaved loss plot to {loss_pdf}")
    print(f"Saved gradient norm plot to {grad_pdf}")

    # Show both plots
    plt_loss.show()
    plt_grad.show()

#dimesions:
n=20
m=20
r1=10
r2=5

#random seed
np.random.seed(0)

# Synthetic tests (random AND POSITIVE)

W = np.random.uniform(0, 1, size=(n, r1))       # fixed W matrix n x p, must be non-negative

W_true = np.random.uniform(-1, 1, size=(r1, r2)) # true W_2, r1=4 x r2, can be negative
H_true = np.random.uniform(-1, 1, size=(r2, m)) # true H_2, r x m, can be negative

X = np.dot(W, (np.maximum(np.dot(W_true,H_true), 0))) # data matrix   n*m IS POSITIVE by definition

# Synthetic tests (gaussian AND POSITIVE)

"""
W = np.abs(np.random.randn(n, r1))   # fixed W matrix   n*p   AND HAS TO BE POSITIVE
W_true = np.random.randn(r1, r2)  # true W_2    p*r
H_true = np.random.randn(r2, m)  # true H_2    r*m
X = np.dot(W, np.dot(np.maximum(W_true, 0),np.maximum(H_true, 0))) # data matrix   n*m IS POSITIVE by definition
"""

run_multiple_initializations_W2_H2(
    X=X,
    W=W,
    W_true=W_true,
    H_true=H_true,
    rate=0.01,
    #parm
    iterations=10000,
    plot_frequency=50,
    n_init=1,
    step_type="line_search",
    reset_check_interval=50000,
    use_momentum=False,
    momentum_beta=0.6,
    init_type="svd_nnls",
    title_prefix="MyExperiment",
    output_dir="my_plots"
)
