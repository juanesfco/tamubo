import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn_extra.cluster import KMedoids
import matplotlib as mpl
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore",category =RuntimeWarning)

def acquisition_function(probabilities, X, k):
    probabilities = np.array(probabilities)
    X = np.array(X)
    eps = 1e-12
    p = probabilities
    entropy = - (p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))
    top_n = max(1, int(np.ceil(0.01 * len(probabilities))))
    sorted_indices = np.argsort(entropy)[::-1]
    top_indices = sorted_indices[:top_n]

    if len(top_indices) < k:
        return top_indices.tolist(), entropy, top_indices, np.arange(len(top_indices))

    X_top = X[top_indices]
    kmedoids = KMedoids(n_clusters=k, metric='euclidean', random_state=0).fit(X_top)
    selected_indices = top_indices[kmedoids.medoid_indices_]
    return selected_indices.tolist(), entropy, top_indices, kmedoids.labels_

def xor_label(X):
    return np.where(X[:, 0] * X[:, 1] > 0, 0, 1)

# Create grid
grid_size = 200
x1, x2 = np.linspace(-2, 2, grid_size), np.linspace(-2, 2, grid_size)
xx, yy = np.meshgrid(x1, x2)
X_grid = np.c_[xx.ravel(), yy.ravel()]
y_grid = xor_label(X_grid)

# Initialize dataset
np.random.seed(42)
total_points = X_grid.shape[0]
n_initial, k_query, max_iter = 15, 5, 20
all_indices = np.arange(total_points)
initial_train_indices = np.random.choice(all_indices, size=n_initial, replace=False)
X_train, y_train = X_grid[initial_train_indices], y_grid[initial_train_indices]
candidate_mask = np.ones(total_points, dtype=bool)
candidate_mask[initial_train_indices] = False
X_pool, y_pool = X_grid[candidate_mask], y_grid[candidate_mask]

# Set up figure for animation
fig, ax = plt.subplots(figsize=(10, 8))
frames = []

def update(frame):
    global X_train, y_train, X_pool, y_pool, candidate_mask

    ax.clear()

    # Train Gaussian Process Classifier
    gpc = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=42)
    gpc.fit(X_train, y_train)
    prob_pool = gpc.predict_proba(X_pool)[:, 1]

    # Active learning acquisition
    selected_local_indices, entropy, top_indices, cluster_labels = acquisition_function(prob_pool, X_pool, k_query)
    X_query, y_query = X_pool[selected_local_indices], y_pool[selected_local_indices]

    # Update training data
    X_train = np.vstack([X_train, X_query])
    y_train = np.hstack([y_train, y_query])
    candidate_mask[np.where(candidate_mask)[0][selected_local_indices]] = False
    X_pool, y_pool = X_grid[candidate_mask], y_grid[candidate_mask]

    # Plot entropy of pool
    entropy = np.delete(entropy, selected_local_indices)
    pool_scatter = ax.scatter(X_pool[:, 0], X_pool[:, 1], c=entropy, cmap='viridis', s=20, edgecolor='none', alpha=0.7)
    #cbar = plt.colorbar(pool_scatter, ax=ax)
    #cbar.set_label('Shannon Entropy')

    # Cluster visualization
    cmap = mpl.colormaps['tab10']
    cluster_edgecolors = [cmap(label) for label in cluster_labels]
    X_pool_top = X_pool[top_indices]
    ax.scatter(X_pool_top[:, 0], X_pool_top[:, 1], facecolors='none', edgecolors=cluster_edgecolors, s=100, linewidths=2)

    # Plot training data
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=100, marker='o', edgecolor='k')

    # Highlight selected query points
    ax.scatter(X_query[:, 0], X_query[:, 1], color='k', marker='*', s=400)

    # Contour of decision boundary
    prob_grid = gpc.predict_proba(X_grid)[:, 1].reshape(xx.shape)
    ax.contour(xx, yy, prob_grid, levels=[0.5], colors='black', linewidths=2, linestyles='--')

    ax.set_title(f'Iteration {frame + 1}: Active Learning on XOR Problem')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    if len(X_pool) == 0:
        ani.event_source.stop()

# Create animation
ani = animation.FuncAnimation(fig, update, frames=max_iter, repeat=False)

# Save as GIF
ani.save('active_learning_xor.gif', writer='pillow', fps=2)

plt.show()
