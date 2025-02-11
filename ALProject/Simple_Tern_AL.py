import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltern
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
from warnings import simplefilter
from sklearn_extra.cluster import KMedoids

from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore",category =RuntimeWarning)

# ---------------------------
# Helper Functions
# ---------------------------
def create_candidate_grid():
    """
    Create a uniform grid in the ternary space with compositions given in 1% steps.
    Only those combinations where A+B+C = 1 (i.e. 100%) are kept.
    """
    data = []
    # Loop over possible integer percentages for A and B. Then C = 100 - A - B.
    for a in range(101):
        for b in range(101 - a):
            c = 100 - a - b
            data.append([a/100, b/100, c/100])
    return pd.DataFrame(data, columns=["A", "B", "C"])

def compute_density(row, noise_std=0.5):
    """
    Compute synthetic density for a given alloy composition (row: A, B, C).
    Density model: density = 10*A + 15*B + 5*C + noise.
    """
    noise = np.random.normal(0, noise_std)
    density = 10 * row["A"] + 15 * row["B"] + 5 * row["C"] + noise
    return density

def label_from_density(density, threshold=12):
    """Return 1 if density > threshold, else 0."""
    return int(density > threshold)

# ---------------------------
# Settings and Initialization
# ---------------------------
np.random.seed(42)
density_threshold = 12  # threshold for classification

# Create candidate grid DataFrame
df_candidates = create_candidate_grid()
df_candidates.reset_index(inplace=True)  # preserve candidate index in a column named "index"

# For convenience, compute the ground truth density and binary label for each candidate.
# (This simulates the true measurement when a candidate is queried.)
df_candidates['Density'] = df_candidates.apply(compute_density, axis=1)
df_candidates['Label'] = df_candidates['Density'].apply(lambda d: label_from_density(d, density_threshold))

# Set aside an initial training set (queried candidates)
n_initial = 1
initial_idx = np.random.choice(df_candidates.index, n_initial, replace=False)
IDXs = [initial_idx[0]]

# Create training DataFrame; these are the points that have been "queried"
df_train = df_candidates.loc[initial_idx, ["index", "A", "B", "C", "Label",'Density']].copy().reset_index(drop=True)

# Create a pool DataFrame (candidates not yet queried)
df_pool = df_candidates.drop(initial_idx).reset_index(drop=True)

# Define Gaussian Process Model (we use regression on the binary label)
kernel = C(1.0) * RBF(length_scale=0.2)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=42,normalize_y=True)

# Create output directory for iteration images
output_dir = "active_learning_iterations"
os.makedirs(output_dir, exist_ok=True)

# Number of active learning iterations
n_iterations = 4

# List to store Jaccard Similarities
Jaccard_Similarities = []

# Batch size
N = 5

# Random Iterations Method 4
r = 1000

# ---------------------------
# Active Learning Loop
# ---------------------------
for it in range(n_iterations):
    # Fit the GP model on the current training data.
    X_train = df_train[["A", "B", "C"]].values
    y_train = df_train["Density"].values
    gpr.fit(X_train, y_train)
    
    # Predict on the full candidate grid (for visualization).
    X_candidates = df_candidates[["A", "B", "C"]].values
    y_pred, y_std = gpr.predict(X_candidates, return_std=True)
    print(y_pred)
    
    # Compute probability that density > density_threshold using the GP prediction distribution.
    # The probability is 1 - CDF((threshold - mean)/std). Note: when y_std is very small, the probability
    # will be nearly 0 or 1.
    prob_above_threshold = 1 - norm.cdf((density_threshold - y_pred) / y_std)
    H = - prob_above_threshold*np.log2(prob_above_threshold) - (1 - prob_above_threshold)*np.log2(1 - prob_above_threshold)
    H = np.nan_to_num(H)
    
    ## To compare how fast each method found the solution (95% probability threshold)
    label_pred = (prob_above_threshold > 0.95).astype(int)
    jaccard_similarity = np.sum(np.logical_and(label_pred,df_candidates['Label'].values))/np.sum(np.logical_or(label_pred,df_candidates['Label'].values))
    Jaccard_Similarities.append(jaccard_similarity)
    # Select the next candidate(s) from the remaining pool. Using method:
    ## 0. Randomly selecting among the pool.
    ## 1. Sample the point at maximum posterior variance. 
    ## 2. Expected Information Gain (Maximizing entropy reduction)
    ## 3. Select top 50% of points with maximum posterior variance, 
    ##    cluster them in N batches and select the N medoids as candidates.
    ## 4. Label pool in N clusters, randomly select one candidate from each cluster,
    ##    repeat r times and calculate expected information gain for each batch of
    ##    candidates. Select batch with highest information gain.
    ## 5. Select top 1% of points with maximum shannons entropy, 
    ##    cluster them in N batches and select the N medoids as candidates.
    method = 5
    if method == 0:
        next_idx = [np.random.choice(df_pool.index)] # Leave it as it was for the sake of comparison.
    elif method == 1:
        next_idx_df_candidates = np.argmax(y_std**2)
        next_idx = [df_pool.loc[df_pool['index']==next_idx_df_candidates].index[0]]
    elif method == 2:
        Entropy_Reductions = []
        for i in range(len(X_candidates)):
            y_pred_i, y_var_i = y_pred[i], y_std[i]**2
            prior_entropy = np.log(y_var_i + 1e-6)
            kernel2 = C(1.0) * RBF(length_scale=0.2)
            gprnew = GaussianProcessRegressor(kernel=kernel2, random_state=40,normalize_y=True)
            gprnew.fit(np.concatenate((X_train, X_candidates[i].reshape(1, -1))),np.append(y_train,y_pred_i+0.1*np.random.randn()))
            y_pred_i_new, y_var_i_new = gprnew.predict(X_candidates[i].reshape(1, -1), return_cov=True)
            post_entropy = np.log(y_var_i_new + 1e-6)[0][0]
            entropy_reduction = abs(post_entropy - prior_entropy)
            Entropy_Reductions.append(entropy_reduction)
        df_Entropy_Reductions = pd.DataFrame({'index':df_candidates['index'].values, 'entropy_reductions':Entropy_Reductions})
        indexes_Drop = df_Entropy_Reductions[df_Entropy_Reductions['index'].isin(IDXs)].index
        df_Entropy_Reductions.drop(indexes_Drop,inplace=True)
        max_index = df_Entropy_Reductions['entropy_reductions'].idxmax()
        next_idx_df_candidates = df_Entropy_Reductions.loc[max_index,'index']
        IDXs.append(next_idx_df_candidates)
        next_idx = [df_pool.loc[df_pool['index']==next_idx_df_candidates].index[0]]
    elif method == 3:
        number_top_candidates = int(0.5*len(df_pool))
        next_idsx_df_candidates = np.argpartition(y_std**2, -number_top_candidates)[-number_top_candidates:]
        next_candidates_df_pool = df_pool.loc[df_pool['index'].isin(next_idsx_df_candidates)]
        next_candidates_df_pool_array = next_candidates_df_pool.loc[:,['A','B','C']].values
        kmedoids = KMedoids(n_clusters=N).fit(next_candidates_df_pool_array)
        next_candidates_id = kmedoids.medoid_indices_
        next_idx = next_candidates_df_pool.iloc[next_candidates_id,0].values
    elif method == 4:
        df_pool_array = df_pool.loc[:,['A','B','C']].values
        kmedoids = KMedoids(n_clusters=N).fit(df_pool_array)
        next_candidates_cluster_labels = kmedoids.labels_
        df_pool_clusters = df_pool.copy()
        df_pool_clusters['cluster'] = next_candidates_cluster_labels
        Candidate_Batch_Indexes = []
        Entropy_Reductions = []
        for i in range(r):
            candidate_batch_indexes = []
            for j in range(N):
                df_pool_cluster_j = df_pool_clusters[df_pool_clusters['cluster']==j]
                candidate_batch_indexes.append(df_pool_cluster_j.sample(1).iat[0,0])
            candidate_batch = df_pool[df_pool['index'].isin(candidate_batch_indexes)]
            X_candidate_batch =  candidate_batch.loc[:,['A','B','C']].values
            y_sample_i = gpr.sample_y(X_candidate_batch,100)
            y_pred_i = gpr.predict(X_candidate_batch)
            prior_entropy = np.sum(np.log(np.var(y_sample_i, axis=1)))
            kernel2 = C(1.0) * RBF(length_scale=0.2)
            gprnew = GaussianProcessRegressor(kernel=kernel2, random_state=40,normalize_y=True)
            gprnew.fit(np.concatenate((X_train, X_candidate_batch)),np.append(y_train,y_pred_i+0.1*np.random.randn(1,N)))
            y_sample_i_new = gprnew.sample_y(X_candidate_batch,100)
            post_entropy = np.sum(np.log(np.var(y_sample_i_new, axis=1)))
            entropy_reduction = abs(post_entropy - prior_entropy)

            Candidate_Batch_Indexes.append(candidate_batch_indexes)
            Entropy_Reductions.append(entropy_reduction)

        entropy_reduction_i_max = np.argmax(Entropy_Reductions)
        next_idx = Candidate_Batch_Indexes[entropy_reduction_i_max]
    elif method == 5:
        number_top_candidates = int(0.01*len(df_pool))
        next_idsx_df_candidates = np.argpartition(H, -number_top_candidates)[-number_top_candidates:]
        next_candidates_df_pool = df_pool.loc[df_pool['index'].isin(next_idsx_df_candidates)]
        next_candidates_df_pool_array = next_candidates_df_pool.loc[:,['A','B','C']].values
        kmedoids = KMedoids(n_clusters=N).fit(next_candidates_df_pool_array)
        next_candidates_id = kmedoids.medoid_indices_
        next_idx = next_candidates_df_pool.iloc[next_candidates_id,0].values

    ##
    next_point = df_pool.loc[next_idx, ["index", "A", "B", "C", "Label",'Density']].copy()
    
    # Add the queried point to the training set.
    df_train = pd.concat([df_train, next_point], ignore_index=True)
    
    # Remove the queried candidate from the pool.
    df_pool = df_pool.drop(next_idx).reset_index(drop=True)
    
    # ---------------------------
    # Plotting the Iteration Result
    # ---------------------------
    # Create a figure with two subplots side-by-side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'ternary'})
    
    # Left subplot: Predicted probability (from GP using the CDF) that density > threshold.
    sc1 = ax1.scatter(df_candidates["A"], df_candidates["B"], df_candidates["C"],
                      c=prob_above_threshold, cmap='seismic', s=10, marker='s', 
                      edgecolors='none', alpha=0.8, vmin=0, vmax=1)
    cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.8)
    cb1.set_label('P(density > 12)')
    ax1.scatter(df_train["A"], df_train["B"], df_train["C"],
                facecolors='none', edgecolors='k', s=80, linewidths=1.5, label="Queried Points")
    ax1.set_title(f"Iteration {it+1} - Probability")
    ax1.legend(loc="upper right")
    
    # Right subplot: Continuous density prediction (raw GP mean prediction).
    sc2 = ax2.scatter(df_candidates["A"], df_candidates["B"], df_candidates["C"],
                      c=y_pred, cmap='viridis', s=10, marker='s', 
                      edgecolors='none', alpha=0.8, vmin=5, vmax=15)
    cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.8)
    cb2.set_label('Predicted Density')
    ax2.scatter(df_train["A"], df_train["B"], df_train["C"],
                facecolors='none', edgecolors='k', s=80, linewidths=1.5, label="Queried Points")
    ax2.set_title(f"Iteration {it+1} - Density Prediction")
    ax2.legend(loc="upper right")
    
    plt.suptitle(f"Active Learning Iteration {it+1}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure as a single image for the iteration.
    filename = os.path.join(output_dir, f"iteration_{it+1}_method_{method}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

## Jaccard index figure
fig = plt.figure(figsize=(6, 5))
plt.plot(Jaccard_Similarities)
plt.xlabel('Iteration')
plt.ylabel('Jaccard Similarity')
plt.title(f"Using Method {method}")
filename = os.path.join(output_dir, f"jaccard_similarity_method_{method}.png")
plt.savefig(filename, dpi=150, bbox_inches='tight')
plt.close(fig)
##
print(f"Saved {n_iterations} iteration images in '{output_dir}/'.")