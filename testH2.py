import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

# Define the function to compute the gradient of F(W, W', H') with respect to H'
def compute_subgradients(W, W_2, H_2, X):
    Z_2 = np.dot(W_2, H_2)
    max_Z_2 = np.maximum(Z_2, 0)  # ReLU (element-wise max with 0)
    error = np.dot(W, max_Z_2) - X
    indicator = (max_Z_2 > 0).astype(float)
    grad_H_2 = 2 * np.dot(W_2.T, (indicator * np.dot(W.T, error)))
    return grad_H_2

# Define the subgradient descent update rule for H' with various step sizes
def subgradient_descent_for_H(X, W, W_2, H_2_init, learning_rate, num_iterations, step_type, plot_frequency):
    n, m = X.shape
    p, r = W_2.shape
    r, m_2 = H_2_init.shape
    H_2 = H_2_init.copy()  # Properly call the copy method

    approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
    frob_norm_X = np.linalg.norm(X, 'fro')
    loss = np.linalg.norm(approx_X - X, 'fro') / frob_norm_X

    loss_history = [np.log(loss)]
    iteration_numbers = [0]

    
    grad_H2 = compute_subgradients(W, W_2, H_2, X)

    grad_norm = (np.linalg.norm(grad_H2))

    grad_norms = [np.log(grad_norm)] ################""





    for iteration in range(num_iterations):
        grad_H_2 = compute_subgradients(W, W_2, H_2, X)

        # Compute and store the loss
        if iteration % plot_frequency == 0 and iteration>0:
            approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
            frob_norm_X = np.linalg.norm(X, 'fro')
            loss = np.linalg.norm(approx_X - X, 'fro') / frob_norm_X
            loss_history.append(np.log(loss))
            iteration_numbers.append(iteration)

            grad_norm = np.linalg.norm(grad_H_2)
            if grad_norm > 0 :
                grad_norms.append(np.log(grad_norm))
            else:
                grad_norms.append(-30)
            # Optionally: print progress at intervals
            print(f"Iteration {iteration}, Loss: {loss:.3e},Grad norm: {np.linalg.norm(grad_H_2, 'fro'):.3e}")

        # Determine the learning rate based on the chosen type
        if step_type == "constant":
            step_size = learning_rate
        elif step_type == "1_over_k":
            step_size = learning_rate / (iteration + 1)
        elif step_type == "1_over_sqrt_k":
            step_size = learning_rate / np.sqrt(iteration + 1)
        elif step_type == "exponential":
            lambda_ = 0.99  # Exponential decay factor
            step_size = learning_rate* (lambda_ ** iteration)
        elif step_type == "polyak":
            grad_norm = np.linalg.norm(grad_H_2)
            if grad_norm != 0:
                step_size = np.linalg.norm(X - np.dot(W, np.maximum(np.dot(W_2, H_2), 0))) / (grad_norm * 4)
            else:
                step_size = learning_rate  # fallback to a constant step
        elif step_type == "line_search":
            # Backtracking line search along negative gradient direction
            alpha = 0.5 
            beta = 0.001
            gamma=0.1
            
            approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
            loss = np.linalg.norm(approx_X - X, 'fro')
            grad_H2_dir = -grad_H2
            step_size = gamma * np.linalg.norm(approx_X,'fro') / grad_norm

            # Try decreasing step size until loss improves
            for _ in range(20):
                H_2_temp = H_2 + step_size * grad_H2_dir
                approx_temp = np.dot(W, np.maximum(np.dot(W_2, H_2_temp), 0))
                new_loss = np.linalg.norm(approx_temp - X, 'fro')
                if new_loss <= loss - beta * step_size * grad_norm:
                    break
                step_size *= alpha
        else:
            raise ValueError(f"Unknown step_type: {step_type}")

        # Update H' based on the subgradient and the learning rate
        H_2_new = H_2 - step_size * grad_H_2
        H_2 = H_2_new +0.6*(H_2 - H_2_new)

        

    # Compute and print final gradient norm
    final_grad = compute_subgradients(W, W_2, H_2, X)
    grad_norm = np.linalg.norm(final_grad)
    print(f"Final gradient norm for step_type='{step_type}': {grad_norm:.4f}")
    
    return H_2, loss_history, iteration_numbers, grad_norms

# New function to run all five subgradient descents and handle plotting
def run_all_subgradient_descents(X, W, W_2, H_2, rate , iterations, plot_frequency):
    # Perform subgradient descent with different step sizes
    results = {}
    step_types = ["constant", "1_over_k", "1_over_sqrt_k", "exponential", "polyak", "line_search"]

    for step_type in step_types:
        print(f"\nRunning Subgradient Descent with {step_type} step size...")
        # Ensure the same initial H_2 is used for each descent
        H_2_opt, loss, iteration_numbers, _= subgradient_descent_for_H(
            X, W, W_2, H_2, learning_rate=rate, num_iterations=iterations, step_type=step_type, plot_frequency=plot_frequency
        )
        results[step_type] = {
            'H_2_opt': H_2_opt,
            'loss': loss,
            'iteration_numbers':iteration_numbers
        }

    # Plot all loss histories on the same plot
    plt.figure(figsize=(10, 6))

    for step_type in step_types:
        plt.plot(results[step_type]['iteration_numbers']results[step_type]['loss'], label=f'{step_type} step', linestyle='-', marker='o')

    # Add labels and legend
    plt.title('Loss over Iterations for Different Step Sizes')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (Frobenius norm)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

  
    # Return the results dictionary containing all the information
    return results


def run_multiple_initializations_H(X, W, W_2, H_chosen, rate, iterations, plot_frequency, n_init, step_type):
    plt.figure(figsize=(10, 6))

    for i in range(n_init):  
        print(f"\n--- Run {i + 1}/{n_init} with step_type = '{step_type}' ---")
        # Random initialization around H_true for fair comparison
        H_2_init = H_chosen + np.random.uniform(-0.01, 0.01, size=H_true.shape)
        #H_2_init = H_chosen #for nnls bcd input
        H_2_opt, loss, iteration_numbers, grad_norms = subgradient_descent_for_H(
            X, W, W_2, H_2_init,
            learning_rate=rate,
            num_iterations=iterations,
            step_type=step_type,
            plot_frequency=plot_frequency
        )

        #plt.plot(iteration_numbers,loss)
        plt.plot(iteration_numbers,loss)

    #plt.title(f'H2 : Log(Loss) over Iterations for {n_init} Runs (step_type = {step_type})')
    plt.title(f'H2 : Log(Grad_norm) over Iterations for {n_init} Runs (step_type = {step_type})')
    plt.xlabel('Logged Iteration (plot frequency step)')
    plt.ylabel('log(Loss) (Frobenius norm)')
    plt.legend()
    plt.grid(True)
    plt.show()

 

def nnls_bcd_init(X, W, r, W2, max_iter=20, tol=1e-4, verbose=True):
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
    #W2 = np.abs(np.random.randn(k, r))
    H2 = np.abs(np.random.randn(r, n))

    normX = np.linalg.norm(X, 'fro')
    errors = []



#BCD
    
    for it in range(max_iter):
        Z = H.copy()
        # Step A: Compute Z = ReLU(W2 @ H2)
        Theta = W2 @ H2
        for i in range(k):
            for j in range(n):
                if Z[i,j] == 0:
                    Z[i,j] = np.minimum(0, Theta[i,j])


        # DOn't update W2

        #  Fit H2 via LS: W2 @ H2 ≈ H
            H2, _, _,_= np.linalg.lstsq(W2, Z)

        # Compute current approximation error
        relu_out = np.maximum(0, W2 @ H2)
        X_hat = W @ relu_out
        err = np.linalg.norm(X - X_hat, 'fro') / normX
        errors.append(err)

        if verbose:
            print(f"[{it+1:03d}] rel.error: {err:.6f}")

        if err < tol:
            break

    return H2
# Synthetic Test Setup


# Dimensions for synthetic data
n, m = 60, 30  # X is 10x10 matrix
p, r = 15, 10  # W' is 10x5 matrix, H' is 5x10 matrix

# Create synthetic X, W', and H'
W = np.abs(np.random.rand(n, p))  # Non-negative W
W_2 = np.random.uniform(-1, 1, size=(p, r))  # Random W'
H_true = np.random.uniform(-1, 1, size=(r, m))  # Ground truth H'

# Set X = W * max(W' H', 0)
X = np.dot(W, np.maximum(np.dot(W_2, H_true), 0))

# Initialize H_2 randomly
H_2= H_true+np.random.uniform(-0.01, 0.01, size=(r, m)) 

#or initialize with an approximate guess
#H_2 = nnls_bcd_init(X, W, r, W_2, max_iter=10, tol=1e-4, verbose=True)

#run
"""
run_all_subgradient_descents(X, W, W_2, H_2, 
                            rate=0.01 ,
                            iterations=10000,
                            plot_frequency=50)

"""




run_multiple_initializations_H(
    X, W, W_2, H_true,
    rate=0.01,
    iterations=5000,
    plot_frequency=50,
    n_init=100,
    step_type='line_search'
)