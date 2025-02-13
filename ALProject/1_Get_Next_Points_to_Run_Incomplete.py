import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, DotProduct
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

# Load data
combined_df = pd.read_csv("combined_solidus_karma_results.csv")

# Ensure Prior values are within (0,1) for logit transformation
#prior_safe = np.clip(combined_df["Prior"] / 100, 1e-6, 1 - 1e-6)

# Scale Prior to logit(45) - logit(55)
#prior_scaled = logit(prior_safe)
#print(prior_scaled)

prior_scaler = MinMaxScaler(feature_range=(logit(.4), logit(.6)))
combined_df["Prior_Scaled"] = prior_scaler.fit_transform(combined_df[["Prior"]])
#combined_df["Prior_Scaled"] = logit(.5)

print(combined_df["Prior_Scaled"])

# Scale Truth (Karma results) between -5 and 5
truth_scaler = MinMaxScaler(feature_range=(-5, 5))
combined_df["Truth_Scaled"] = truth_scaler.fit_transform(combined_df[["Truth"]])

# Separate training and prediction sets
train_df = combined_df[combined_df["Source"] == "Karma"]
#train_df = train_df.sample(n=10)

grid_df = combined_df[combined_df["Source"] == "Grid"]

# MinMax Scale X data (R, G, C)
X_scaler = MinMaxScaler(feature_range=(0,1))
grid_df[["R", "G", "C"]] = X_scaler.fit_transform(grid_df[["R", "G", "C"]])
train_df[["R", "G", "C"]] = X_scaler.transform(train_df[["R", "G", "C"]])

# Set up GP with prior
X_train = train_df[["R", "G", "C"]].values
y_train = train_df["Truth_Scaled"].values
train_prior = train_df["Prior_Scaled"].values

gp = GaussianProcessWithPrior(kernel=Matern(nu=.5,length_scale=[1,1,1],length_scale_bounds=(.05,.1)), n_restarts_optimizer=10,normalize_y=False)
gp.fit(X_train, y_train, train_prior)

# Predict for grid data
X_grid = grid_df[["R", "G", "C"]].values
predict_prior = grid_df["Prior_Scaled"].values
grid_df["Predicted_Probability"], grid_df["Std_Dev"] = gp.predict(X_grid, predict_prior)

# Save grid_df and train_df
grid_df.to_csv('grid_df_after_initial_iteration.csv', index=False)
train_df.to_csv('train_df_after_initial_iteration.csv', index=False)

### ANIMATION: G vs R while C varies ###
fig, ax = plt.subplots(figsize=(8, 6))

# Extract unique `C` values for animation frames
unique_C_values = np.sort(grid_df["C"].unique())
print(unique_C_values)

# Create a scatter plot that will be updated in the animation
sc = ax.scatter([], [], c=[], s=[], cmap="coolwarm",vmin=0,vmax=1)

# Labels and colorbar
ax.set_xlabel("R")
ax.set_ylabel("G")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Predicted Probability")

# Function to update animation frame
def update(frame):
    ax.clear()
    
    C_value = unique_C_values[frame]
    subset = grid_df[grid_df["C"] == C_value]
    
    R = subset["R"].values
    G = subset["G"].values
    predictions = subset["Predicted_Probability"].values
    std_dev = subset["Std_Dev"].values
    
    sc = ax.scatter(R, G, c=predictions, cmap="coolwarm", edgecolor="None",marker='s',s=55,vmin=0,vmax=1)
    ax.set_title(f"Varying C: {C_value:.2f}")
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    
    return sc,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(unique_C_values), interval=5, blit=False)
ani.save('Posterior_G_vs_R_slice_C.gif', writer='imagemagick', fps=10)
# Save or display animation
# plt.show()
