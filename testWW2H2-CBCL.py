# === CBCL NNMF Comparison Script ===

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from PIL import Image
import scipy.optimize as sp

# === Load CBCL Dataset ===
def load_cbcl_faces(folder):
    files = sorted(glob(os.path.join(folder, "face*") + ".pgm"))
    images = [np.array(Image.open(f), dtype=np.float64).flatten() for f in files]
    return np.array(images).T  # Each column is an image

# === Plot Grid of Images ===
def plot_image_grid(matrix, img_shape, grid_shape, title, invert=False, save_path=None):
    fig, ax = plt.subplots()
    n_rows, n_cols = grid_shape
    h, w = img_shape
    canvas = np.ones((n_rows * (h + 1), n_cols * (w + 1))) * 255
    for idx in range(n_rows * n_cols):
        if idx >= matrix.shape[1]:
            break
        img = matrix[:, idx].reshape(img_shape)
        if invert:
            img = 255 - img
        row = idx // n_cols
        col = idx % n_cols
        canvas[row * (h + 1):(row + 1) * (h + 1) - 1, col * (w + 1):(col + 1) * (w + 1) - 1] = img
    ax.imshow(canvas, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# === Basic BCD NMF ===
def bcd_nmf(X, r, max_iter=50, phase_name=""): 
    print(f"\n--- Running {phase_name} BCD (Rank {r}) ---")
    n, m = X.shape
    W = np.abs(np.random.randn(n, r))
    H = np.abs(np.random.randn(r, m))
    normX = np.linalg.norm(X, 'fro')
    loss_history = [np.linalg.norm(X - W @ H, 'fro') / normX]
    print(f"[Init] Relative loss: {loss_history[-1]:.6f}")

    for it in range(max_iter):
        for j in range(m):
            H[:, j], _ = sp.nnls(W, X[:, j])
        for i in range(n):
            W[i, :], _ = sp.nnls(H.T, X[i, :])
        loss = np.linalg.norm(X - W @ H, 'fro') / normX
        loss_history.append(loss)
        print(f"[Iter {it+1}] Relative loss: {loss:.6f}")
    return W, H, loss_history

# === Run All ===
def run_all_comparisons(X, r1, r2):
    from copy import deepcopy
    #from nnmf_alternating_opt import pre_initialize_WH, initialize_W2_H2_from_H, alternating_descent_W_W2_H2

    print("\n=== 1st METHOD: BCD with rank = r1 ===")
    W1, H1, loss_bcd_r1 = bcd_nmf(deepcopy(X), r1, max_iter=60, phase_name="1st")

    print("\n=== 2nd METHOD: BCD with rank = r2 ===")
    W2, H2, loss_bcd_r2 = bcd_nmf(deepcopy(X), r2, max_iter=60, phase_name="2nd")

    print("\n=== 3rd METHOD: NNMF Alternating Opt ===")
    W_init, H_mid = pre_initialize_WH(X, r1, max_iter=5)
    W2_init, H2_init = initialize_W2_H2_from_H(H_mid, r2)
    W_nnmf, W2_nnmf, H2_nnmf, loss_nnmf, _ = alternating_descent_W_W2_H2(
        X, W_init, W2_init, H2_init,
        num_outer_iterations=60,
        inner_iterations=1,
        learning_rate=0.01,
        step_type="line_search",
        plot_frequency=1,
        use_momentum=True,
        momentum_beta=0.9,
        verbose=True
    )

    # === Plot Loss Comparison ===
    plt.figure(figsize=(10,6))
    plt.plot(loss_nnmf, label=f"NNMF Alternating Opt (r1={r1}, r2={r2}, inner=1, step=line_search)")
    plt.plot(loss_bcd_r1, label=f"Classic BCD (r={r1})")
    plt.plot(loss_bcd_r2, label=f"Classic BCD (r={r2})")
    plt.yscale('log')
    plt.xlabel("Iterations")
    plt.ylabel("Relative Loss (log scale)")
    plt.title("Comparison of NMF Methods on CBCL Dataset\n(NNMF: 60 outer iters, 1 inner iter, line_search, momentum 0.9)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_comparison_cbcl.png", dpi=300)
    plt.show()

    # === Reconstruction / Features ===
    n_faces_to_plot = min(100, X.shape[1])
    grid_size = int(np.sqrt(n_faces_to_plot))
    X_sample = X[:, :grid_size**2]

    X_recon_bcd_r1 = W1 @ H1[:, :grid_size**2]
    X_recon_bcd_r2 = W2 @ H2[:, :grid_size**2]
    Z_nnmf = np.maximum(W2_nnmf @ H2_nnmf[:, :grid_size**2], 0)
    X_recon_nnmf = W_nnmf @ Z_nnmf

    plot_image_grid(X_sample, img_shape=(19,19), grid_shape=(grid_size,grid_size), title="Original Images", save_path="original_cbcl.png")
    plot_image_grid(X_recon_bcd_r1, (19,19), (grid_size,grid_size), title=f"BCD Reconstruction r={r1}", save_path="bcd_r1_reconstruction.png")
    plot_image_grid(X_recon_bcd_r2, (19,19), (grid_size,grid_size), title=f"BCD Reconstruction r={r2}", save_path="bcd_r2_reconstruction.png")
    plot_image_grid(X_recon_nnmf, (19,19), (grid_size,grid_size), title=f"NNMF Reconstruction (r1={r1}, r2={r2})", save_path="nnmf_reconstruction.png")

    # === Plot Features (H) as Images ===
    plot_image_grid(H1[:, :grid_size**2], (10,10), (grid_size,grid_size), title=f"Features H (BCD r={r1})", save_path="features_bcd_r1.png")
    plot_image_grid(H2[:, :grid_size**2], (10,10), (grid_size,grid_size), title=f"Features H (BCD r={r2})", save_path="features_bcd_r2.png")
    plot_image_grid(np.maximum(W2_nnmf @ H2_nnmf[:, :grid_size**2], 0), (10,10), (grid_size,grid_size), title="Features H (NNMF via ReLU)", save_path="features_nnmf.png")

# === Main Execution ===
if __name__ == "__main__":
    X = load_cbcl_faces("./cbcl_faces")  # adapt path if needed
    print("Loaded CBCL data of shape:", X.shape)  # (361, ~2000)
    run_all_comparisons(X, r1=100, r2=20)