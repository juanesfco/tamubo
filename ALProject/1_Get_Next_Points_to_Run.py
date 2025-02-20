import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn_extra.cluster import KMedoids
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.animation as animation

iteration = 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def probability3labels(p):
    if p > 0.975:
        return 1
    elif p >= 0.025:
        return 0.5
    else:
        return 0
    
# Function to map probability values to RGBA
cmap = mpl.colormaps['bwr']
def colorMap(p):
    return(np.array(cmap(int(255*p))))

class GaussianProcessWithPrior:
    def __init__(self, kernel=None, normalize_y=False, n_restarts_optimizer=10):
        self.gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=normalize_y, n_restarts_optimizer=n_restarts_optimizer)
        self.train_prior = None  

    def fit(self, X, y, train_prior):
        """Fits the model to residuals (y - train_prior)."""
        if train_prior.shape != y.shape:
            raise ValueError("Shape mismatch between 'train_prior' and 'y'")
        self.train_prior = train_prior
        residuals = y - train_prior
        self.gpr.fit(X, residuals)

    def predict(self, X, predict_prior, return_std=True):
        """Predicts using the GP model with the prior added back."""
        if predict_prior.shape[0] != X.shape[0]:
            raise ValueError("Shape mismatch between 'predict_prior' and 'X'")

        mu, std = self.gpr.predict(X, return_std=True)
        mu_with_prior = mu + predict_prior

        if return_std:
            return sigmoid(mu_with_prior), std
        return sigmoid(mu_with_prior)
    

def aquisitionFunction(df_grid, X_labels, N, top):
    """ 
    Selects new points to be queried given a probability map by selecting the top top_percent%
    points with maximum shannon entropy and uses KMedoids to ensure the N points are well spread.
    Args:
        df_grid:    Pandas DataFrame with all the points to be considered and their probability.
                    Variables have columns with names listed in X_labels and probability column is
                    called 'Predicted_Probability'.
        X_labels:   List of labels corresponding to the names of the columns of df_grid with the variables.
        N:          batch size
        top:        Percentage to be considered when selecting maximum shannon entropy.
    Returns:
        df_grid_candidates_top: Pandas DataFrame with the candidates selected
        df_grid_candidates_medoids: Pandas DataFrame with the cluster medoids
        df_grid_candidates:         Pandas DataFrame with the top candidates
        df_grid:                    Pandas DataFrame with all candidates considered

    """
    p = df_grid['Predicted_Probability'].values
    H = - p*np.log2(p) - (1 - p)*np.log2(1 - p)
    H = np.nan_to_num(H)
    df_grid['H'] = H

    number_top_candidates = int(top/100*len(df_grid))
    id_top_candidates_df_grid = np.argpartition(H, -number_top_candidates)[-number_top_candidates:]
    df_grid_candidates = df_grid.loc[id_top_candidates_df_grid,:]
    df_grid_candidates_array = df_grid_candidates.loc[:,X_labels].values
    kmedoids = KMedoids(n_clusters=N).fit(df_grid_candidates_array)
    id_medoids_candidates = kmedoids.medoid_indices_
    cluster_labels = kmedoids.labels_
    cmap = mpl.colormaps['Set3']
    cluster_colors = [cmap(label) for label in cluster_labels]

    df_grid_candidates['cluster_color'] = cluster_colors
    df_grid_candidates_medoids = df_grid_candidates.iloc[id_medoids_candidates,:]

    idmax_cluster = df_grid_candidates.groupby('cluster_color')['H'].idxmax().values
    df_grid_candidates_top = df_grid_candidates.loc[idmax_cluster,:]
    return df_grid_candidates_top, df_grid_candidates_medoids, df_grid_candidates, df_grid

# Load data (Change iteration number)
combined_df = pd.read_csv(f"Data/combined_solidus_karma_results_iteration_{iteration-1}_3labels.csv", dtype={'Explanation': str})
combined_df['log10(G)'] = np.log10(combined_df['G'])
print("Data read and log10(G) column added.")

# Ensure Prior values are within (0,1) for logit transformation
#prior_safe = np.clip(combined_df["Prior"] / 100, 1e-6, 1 - 1e-6)

# Scale Prior to logit(45) - logit(55)
#prior_scaled = logit(prior_safe)
#print(prior_scaled)

prior_scaler = MinMaxScaler(feature_range=(logit(.4), logit(.6)))
combined_df["Prior_Scaled"] = prior_scaler.fit_transform(combined_df[["Prior"]])
#combined_df["Prior_Scaled"] = logit(.5)

#print(combined_df["Prior_Scaled"])

# Scale Truth (Karma results) between -5 and 5
truth_scaler = MinMaxScaler(feature_range=(-5, 5))
combined_df["Truth_Scaled"] = truth_scaler.fit_transform(combined_df[["Truth"]])

# Separate training and prediction sets
train_df = (combined_df[combined_df["Source"] == "Karma"]).copy()
#train_df = train_df.sample(n=10)

grid_df = (combined_df[combined_df["Source"] == "Grid"]).copy()

# MinMax Scale X data (R, G, C)
X_labels = ["R", "log10(G)", "C"]
X_scaler = MinMaxScaler(feature_range=(0,1))
grid_df.loc[:,X_labels] = X_scaler.fit_transform(grid_df[X_labels])
train_df.loc[:,X_labels] = X_scaler.transform(train_df[X_labels])
print("Data scaled and separated in training and production.")

# Set up GP with prior
X_train = train_df[X_labels].values
y_train = train_df["Truth_Scaled"].values
train_prior = train_df["Prior_Scaled"].values

gp = GaussianProcessWithPrior(kernel=Matern(nu=.5,length_scale=[1,1,1],length_scale_bounds=(.05,1)), n_restarts_optimizer=10,normalize_y=False)
gp.fit(X_train, y_train, train_prior)
print("GP Trained")
print("Used length scales:")
print(gp.gpr.kernel_.length_scale)

# Predict for grid data
X_grid = grid_df[X_labels].values
predict_prior = grid_df["Prior_Scaled"].values
grid_df["Predicted_Probability"], grid_df["Std_Dev"] = gp.predict(X_grid, predict_prior)
print("Posterior predictions done.")

# Label Probability
grid_df["Label_Probability"] = grid_df["Predicted_Probability"].apply(probability3labels)

# Select next points to be queried based on posterior results
grid_df_candidates_top, grid_df_candidates_medoids, grid_df_candidates, grid_df  = aquisitionFunction(grid_df.copy(),X_labels,10,2.5)
print("Next points selected.")

# Inverse MinMax Scale X data (R, G, C)
grid_df[X_labels] = X_scaler.inverse_transform(grid_df[X_labels])
train_df[X_labels] = X_scaler.inverse_transform(train_df[X_labels])
grid_df_candidates[X_labels] = X_scaler.inverse_transform(grid_df_candidates[X_labels])
grid_df_candidates_medoids.loc[:,X_labels] = X_scaler.inverse_transform(grid_df_candidates_medoids[X_labels])
grid_df_candidates_top.loc[:,X_labels] = X_scaler.inverse_transform(grid_df_candidates_top[X_labels])
print("Inverse scaling done.")

# Save grid_df_candidates_top, grid_df_candidates_medoids, grid_df and train_df (change iteration number)
grid_df_candidates_top.to_csv(f'Data/grid_df_candidates_top_iteration_{iteration}_3labels.csv', index=False)
grid_df_candidates_medoids.to_csv(f'Data/grid_df_candidates_medoids_iteration_{iteration}_3labels.csv', index=False)
grid_df.to_csv(f'Data/grid_df_iteration_{iteration}_3labels.csv', index=False)
train_df.to_csv(f'Data/train_df_iteration_{iteration}_3labels.csv', index=False)
print("Data saved as csv")

### ANIMATION: G vs R while C varies ###
fig, ax = plt.subplots(1,2,figsize=(12, 6),sharey=True,width_ratios=[4,5])

# Extract unique `C` values for animation frames
unique_C_values = np.sort(grid_df["C"].unique())
unique_C_values_interval = (unique_C_values[1]-unique_C_values[0])/2
#print(unique_C_values)

# Create a scatter plot that will be updated in the animation
sc0 = ax[0].scatter([], [], c=[], s=[], cmap="coolwarm",vmin=0,vmax=1)
sc1 = ax[1].scatter([], [], c=[], s=[], cmap="coolwarm",vmin=0,vmax=1)

# Labels and colorbar
ax[0].set_xlabel("R")
ax[1].set_xlabel("R")
ax[0].set_ylabel(r"$\log_{10}$G")
ax[1].set_ylabel(r"$\log_{10}$G")
cbar = plt.colorbar(sc1, ax=ax[1])
cbar.set_label("Probability Alloy is Planar")

# Function to update animation frame
def update(frame):
    ax[0].clear()
    ax[1].clear()
    
    C_value = unique_C_values[frame]
    C_value_min = C_value - unique_C_values_interval
    C_value_max = C_value + unique_C_values_interval

    grid_df_subset = grid_df[grid_df["C"] == C_value]
    grid_df_candidates_subset = grid_df_candidates[grid_df_candidates["C"] == C_value]
    train_df_subset = train_df[(train_df["C"] >= C_value_min) & (train_df["C"] < C_value_max)]
    grid_df_candidates_medoids_subset = grid_df_candidates_medoids[(grid_df_candidates_medoids["C"] > C_value_min) & (grid_df_candidates_medoids["C"] < C_value_max)]
    #grid_df_candidates_top_subset = grid_df_candidates_top[(grid_df_candidates_top["C"] > C_value_min) & (grid_df_candidates_top["C"] < C_value_max)]
    
    R = grid_df_subset["R"].values
    G = grid_df_subset["log10(G)"].values
    predictions = grid_df_subset["Predicted_Probability"].values
    labelProb = grid_df_subset["Label_Probability"].values
    #std_dev = grid_df_subset["Std_Dev"].values

    sc0 = ax[0].scatter(R, G, c=labelProb, cmap="seismic", edgecolor="None",marker='s',s=55,vmin=0,vmax=1)
    ax[0].scatter(train_df_subset["R"], train_df_subset["log10(G)"],
                facecolors='none', edgecolors='orange', s=80, linewidths=1.5, label="Queried Points")
    #ax[0].scatter(grid_df_candidates_top_subset["R"], grid_df_candidates_top_subset["log10(G)"],
    #            color='k', marker='*', s=80, label="Next Points")
    ax[0].scatter(grid_df_candidates_medoids_subset["R"], grid_df_candidates_medoids_subset["log10(G)"],
                color='k', marker='*', s=80, label="Next Points")
    ax[0].set_xlabel("R")
    ax[0].set_ylabel(r"$\log_{10}$G")
    
    sc1 = ax[1].scatter(R, G, c=predictions, cmap="seismic", edgecolor="None",marker='s',s=55,vmin=0,vmax=1)
    ax[1].scatter(train_df_subset["R"], train_df_subset["log10(G)"],
                facecolors='none', edgecolors='orange', s=80, linewidths=1.5, label="Queried Points")
    ax[1].scatter(grid_df_candidates_subset["R"], grid_df_candidates_subset["log10(G)"],
                color=grid_df_candidates_subset['cluster_color'], s=10, label="Clusters")
    #ax[1].scatter(grid_df_candidates_medoids_subset["R"], grid_df_candidates_medoids_subset["log10(G)"],
    #            color=grid_df_candidates_medoids_subset['cluster_color'], s=50, label="Cluster Medoid")
    #ax[1].scatter(grid_df_candidates_top_subset["R"], grid_df_candidates_top_subset["log10(G)"],
    #            color='k', marker='*', s=80, label="Next Points")
    ax[1].scatter(grid_df_candidates_medoids_subset["R"], grid_df_candidates_medoids_subset["log10(G)"],
                color='k', marker='*', s=80, label="Next Points")
    #ax[0].set_title(f"Post Prob After It 1 - C: {C_value:.4f} - Top 2.5%") # Change iteration number
    ax[1].set_xlabel("R")
    
    ax[1].legend(loc='upper right')
    
    fig.suptitle(f"Posterior Iteration {iteration} - C: {C_value:.4f}")
    fig.tight_layout()

    return sc0, sc1

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(unique_C_values), interval=5, blit=False)
ani.save(f'Figures/posterior_iteration_{iteration}_3labels_medoids.gif', fps=5) # Change iteration number
print("Animation saved.")

# Sort data by x, y, z for 3D plots
grid_df_sorted = grid_df.sort_values(by=['R','C','log10(G)'])

# Apply colormap function to all probability values and RGBA colors in a (50,50,50,4) array
#colorArray = np.stack(grid_df_sorted['Prior'].apply(colorMap).values).reshape(50,50,50,4)
colorArray = np.stack(grid_df_sorted['Predicted_Probability'].apply(colorMap).values).reshape(50,50,50,4)
#colorArray = np.stack(grid_df_sorted['Label_Probability'].apply(colorMap).values).reshape(50,50,50,4)

# Create copies and change A parameter (transparency) on desired planes
colorArray1 = colorArray.copy()
colorArray2 = colorArray.copy()
colorArray3 = colorArray.copy()

colorArray1[:,:,:,3] = 0.25
colorArray1[:,0,:,3] = 1
colorArray2[:,:,:,3] = 0.25
colorArray2[:,25,:,3] = 1
colorArray3[:,:,:,3] = 0.25
colorArray3[:,49,:,3] = 1

# Create voxel array, initialize with all true because all of them are going to be plotted
voxelarray = np.full((50, 50, 50), True)

# Create copies and remove desired voxels from each plot
voxelarray1 = voxelarray.copy()
voxelarray2 = voxelarray.copy()
voxelarray2[:,:25,:] = False
voxelarray3 = voxelarray.copy()
voxelarray3[:,:49,:] = False

# Create labels and select their respective ticks
numberRlabels = 4
numberClabels = 5
numberlogGlabels = 5

Rlabels = np.linspace(grid_df['R'].min(),grid_df['R'].max(),numberRlabels).round(3)
Rticks = np.linspace(0,50,numberRlabels)
Clabels = np.linspace(grid_df['C'].min(),grid_df['C'].max(),numberClabels).round(2)
Cticks = np.linspace(0,50,numberClabels)
logGlabels = np.linspace(grid_df['log10(G)'].min(),grid_df['log10(G)'].max(),numberlogGlabels).round(1)
logGticks = np.linspace(0,50,numberlogGlabels)

# Create figures and add subplots
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax4 = fig.add_subplot(1, 3, 3)

ax1.voxels(voxelarray1, facecolors=colorArray1)
ax1.set_xlabel('R')
ax1.set_ylabel('C')
ax1.set_zlabel(r'$\log_{10}$(G)')
ax1.set_xticks(ticks=Rticks,labels=Rlabels)
ax1.set_yticks(ticks=Cticks,labels=Clabels)
ax1.set_zticks(ticks=logGticks,labels=logGlabels)

ax2.voxels(voxelarray2, facecolors=colorArray2)
ax2.set_xlabel('R')
ax2.set_ylabel('C')
ax2.set_zlabel(r'$\log_{10}$(G)')
ax2.set_xticks(ticks=Rticks,labels=Rlabels)
ax2.set_yticks(ticks=Cticks,labels=Clabels)
ax2.set_zticks(ticks=logGticks,labels=logGlabels)

ax3.voxels(voxelarray1, facecolors=colorArray3)
ax3.set_xlabel('R')
ax3.set_ylabel('C')
ax3.set_zlabel(r'$\log_{10}$(G)')
ax3.set_xticks(ticks=Rticks,labels=Rlabels)
ax3.set_yticks(ticks=Cticks,labels=Clabels)
ax3.set_zticks(ticks=logGticks,labels=logGlabels)

ax4.axis('off')
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap='bwr', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm,ax=ax4,shrink=0.7,anchor=(3.2,0.5))
cbar.set_label('Probability Alloy is Planar')

fig.suptitle('Posterior Iteration 3')
fig.savefig(f'Figures/posterior_iteration_{iteration}_3labels.png')
print("Figure created and saved.")
print("Script finished. Next points to run:")
print(grid_df_candidates_medoids[["R", "G", "C"]])