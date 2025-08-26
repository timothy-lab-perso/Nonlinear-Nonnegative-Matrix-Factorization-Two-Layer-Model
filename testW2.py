import numpy as np
import matplotlib.pyplot as plt

# Compute the gradient of F(W, W', H') with respect to W'
def compute_subgradients_W2(W, W_2, H_2, X):
    Z_2 = np.dot(W_2, H_2)
    max_Z_2 = np.maximum(Z_2, 0)  # ReLU
    error = np.dot(W, max_Z_2) - X
    indicator = (Z_2 > 0).astype(float)
    #indicator_0 = (Z_2 == 0).astype(float) # can be changed because several subgradients fit
    grad_W2 = 2 * np.dot((np.dot(W.T, error) * (indicator)), H_2.T)
    return grad_W2

# Subgradient descent update for W2
def subgradient_descent_for_W2(X, W, W_2, H_2, learning_rate, num_iterations, step_type, plot_frequency):
    new_W_2 = W_2.copy()
    loss_history = []
    iteration_numbers=[]
    grad_norms = []

    for iteration in range(num_iterations):
        grad_W2 = compute_subgradients_W2(W, new_W_2, H_2, X)
        grad_norm = np.linalg.norm(grad_W2)


        # Step size handling
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
            grad_norm = np.linalg.norm(grad_W2)
            if grad_norm != 0:
                approx_X = np.dot(W, np.maximum(np.dot(new_W_2, H_2), 0))
                step_size = np.linalg.norm(X - approx_X) / (grad_norm * 4)
        elif step_type =="line_search":
            step_size = learning_rate
            # Backtracking line search along negative gradient direction
            alpha = 0.5 
            beta = 0.001
            gamma=0.1
            
            approx_X = np.dot(W, np.maximum(np.dot(W_2, H_2), 0))
            loss = np.linalg.norm(approx_X - X, 'fro')
            grad_W2_dir = -grad_W2
            step_size = gamma * np.linalg.norm(approx_X,'fro') / grad_norm

            # Try decreasing step size until loss improves
            for _ in range(20):
                W_2_temp = W_2 + step_size * grad_W2_dir
                approx_temp = np.dot(W, np.maximum(np.dot(W_2_temp, H_2), 0))
                new_loss = np.linalg.norm(approx_temp - X, 'fro')
                if new_loss <= loss - beta * step_size * grad_norm:
                    break
                step_size *= alpha
        else:
            raise ValueError(f"Unknown step_type: {step_type}")
        
        #plot current values
        if iteration % plot_frequency == 0:
            approx_X = np.dot(W, np.maximum(np.dot(new_W_2, H_2), 0))
            loss = np.linalg.norm(approx_X - X, 'fro')
            if loss <= 0:
                raise ValueError("Loss is non-positive, cannot compute log")
            loss_history.append(np.log(loss))
            iteration_numbers.append(iteration)
            grad_norm = np.linalg.norm(grad_W2)
            if grad_norm > 0 :
                grad_norms.append(np.log(grad_norm))
            else:
                grad_norms.append(-30)
            print(f"Iteration {iteration}, log(Loss): {np.log(loss):.4f}, Grad norm: {np.linalg.norm(grad_W2):.4f}")
        #update after ploting
        new_W_2 -= step_size * grad_W2

       

    return new_W_2, loss_history, iteration_numbers,grad_norms

# Function to run all step size methods for W2
def run_all_subgradient_descents_W2(X, W, W_2, H_2, rate, iterations, plot_frequency):
    results = {}
    step_types = [#"constant",
                   "1_over_k",
                    "1_over_sqrt_k",
                    "exponential", 
                    #"polyak"
                    ]

    for step_type in step_types:
        print(f"\nRunning Subgradient Descent for W2 with {step_type} step size...")
        W_2_init = W_2.copy()  # Ensure same starting point
        W_2_opt, loss = subgradient_descent_for_W2(
            X, W, W_2_init, H_2,
            learning_rate=rate, num_iterations=iterations,
            step_type=step_type, plot_frequency=plot_frequency
        )
        results[step_type] = {
            'W_2_opt': W_2_opt,
            'loss': loss
        }

    plt.figure(figsize=(10, 6))
    for step_type in step_types:
        plt.plot(results[step_type]['loss'], label=f'{step_type} log loss', marker='o')

    plt.title('log(Loss) over Iterations for Different Step Sizes (Optimizing W2)')
    plt.xlabel('Logged Iteration (plot frequency step)')
    plt.ylabel('log(Loss) (Frobenius norm)')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.ylim(-30, 5)
    plt.show()

    return results

def run_multiple_initializations_W2(X, W, W_2_true,H_2, rate, iterations, plot_frequency, n_init, step_type):
    plt.figure(figsize=(10, 6))

    for i in range(n_init):
        print(f"\n--- Run {i + 1}/{n_init} with step_type = '{step_type}' ---")
        #W_2_init= np.random.uniform(-1, 1, size=W_2_true.shape)
        W_2_init = W_2_true + np.random.uniform(-0.01, 0.01, size=W_2_true.shape)

        W_2_opt, loss, iteration_numbers,grad_norms = subgradient_descent_for_W2(
            X, W, W_2_init, H_2,
            learning_rate=rate,
            num_iterations=iterations,
            step_type=step_type,
            plot_frequency=plot_frequency
        )

        #plt.plot(iteration_numbers,loss)
        plt.plot(iteration_numbers,grad_norms)

    #plt.title(f'W2 :Log(Loss) over Iterations for {n_init} Runs (step_type = {step_type})')
    plt.title(f'W2 : Log(Grad_norm) over Iterations for {n_init} Runs (step_type = {step_type})')
    plt.xlabel('Logged Iteration (plot frequency step)')
    plt.ylabel('log(Loss) (Frobenius norm)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Synthetic Data Setup
n, m = 20, 20
p, r = 10, 5
W = np.abs(np.random.uniform(0, 1, size=(n, p)))
#W = np.abs(np.random.randn(n, p))  #gaussian
W_2_true = np.random.uniform(-1, 1, size=(p, r))
#W_2_true = np.random.randn(p, r) #gaussian
H_true = np.random.uniform(-1, 1, size=(r, m))
#H_true = np.random.randn(r, m)   #gaussian
X = np.dot(W, np.maximum(np.dot(W_2_true, H_true), 0))

# Add noise to W_2 for initialization
W_2_init = W_2_true + np.random.uniform(-0.01, 0.01, size=W_2_true.shape)

# Run 
# Run the experiment on multiple initializations
run_multiple_initializations_W2(
    X, W, W_2_true, H_true,
    rate=0.01,
    iterations=5000,
    plot_frequency=50,
    n_init=50,
    step_type="line_search"
)

