import pandas as pd
import numpy as np
import itertools

# ---------------------------
# 1. Define the Karma Data
# ---------------------------
karma_data = [
    [0.15, 1e7, 0.012, 0, "Cellular"],
    [0.15, 1e7, 0.03,  0, "Cellular"],
    [0.15, 1e7, 0.06,  0, "Cellular"],
    [0.15, 1e7, 0.12,  0, "Cellular"],
    [0.15, 1e7, 0.3,   1, "Planar"  ],
    [0.15, 1e7, 0.6,   1, "Planar"  ],

    [0.15, 1e6, 0.012, 0, "Cellular (not run)"],
    [0.15, 1e6, 0.03,  0, "Cellular (not run)"],
    [0.15, 1e6, 0.06,  0, "Cellular"],
    [0.15, 1e6, 0.12,  0, "Cellular"],
    [0.15, 1e6, 0.3,   1, "Planar"  ],
    [0.15, 1e6, 0.6,   1, "Planar"  ],

    [0.15, 1e5, 0.012, 0, "Cellular (not run)"],
    [0.15, 1e5, 0.03,  0, "Cellular (not run)"],
    [0.15, 1e5, 0.06,  0, "Cellular (not run)"],
    [0.15, 1e5, 0.12,  0, "Cellular"],
    [0.15, 1e5, 0.3,   0.5, "Planar (ish)"], # Removed because of inconclusive simulation
    [0.15, 1e5, 0.6,   1, "Planar"],
    
    [0.17, 1e7, 0.012, 0, "Cellular"],
    [0.17, 1e7, 0.03,  0, "Cellular"],
    [0.17, 1e7, 0.06,  0, "Cellular"],
    [0.17, 1e7, 0.12,  0, "Cellular"],
    [0.17, 1e7, 0.3,   1, "Planar"  ],
    [0.17, 1e7, 0.6,   1, "Planar"  ],

    [0.17, 1e6, 0.012, 0, "Cellular"],
    [0.17, 1e6, 0.03,  0, "Cellular"],
    [0.17, 1e6, 0.06,  0, "Cellular"],
    [0.17, 1e6, 0.12,  0, "Cellular"],
    [0.17, 1e6, 0.3,   0, "Cellular (ish)"],
    [0.17, 1e6, 0.6,   1, "Planar"],

    [0.17, 1e5, 0.012, 0, "Cellular (not run)"],
    [0.17, 1e5, 0.03,  0, "Cellular (not run)"],
    [0.17, 1e5, 0.06,  0, "Cellular (not run)"],
    [0.17, 1e5, 0.12,  0, "Cellular"],
    [0.17, 1e5, 0.3,   0, "Cellular (ish)"],
    [0.17, 1e5, 0.6,   1, "Planar"],
    
    [0.19, 1e7, 0.012, 0, "Cellular"],
    [0.19, 1e7, 0.03,  0, "Cellular"],
    [0.19, 1e7, 0.06,  0, "Cellular"],
    [0.19, 1e7, 0.12,  0, "Cellular"],
    [0.19, 1e7, 0.3,   1, "Planar"],
    [0.19, 1e7, 0.6,   1, "Planar"],

    [0.19, 1e6, 0.012, 0, "Cellular (not run)"],
    [0.19, 1e6, 0.03,  0, "Cellular (not run)"],
    [0.19, 1e6, 0.06,  0, "Cellular"],
    [0.19, 1e6, 0.12,  0, "Cellular"],
    [0.19, 1e6, 0.3,   0, "Cellular (ish)"],
    [0.19, 1e6, 0.6,   1, "Planar"],

    [0.19, 1e5, 0.012, 0, "Cellular (not run)"],
    [0.19, 1e5, 0.03,  0, "Cellular (not run)"],
    [0.19, 1e5, 0.06,  0, "Cellular (not run)"],
    [0.19, 1e5, 0.12,  0, "Cellular"],
    [0.19, 1e5, 0.3,   0, "Cellular (ish)"],
    [0.19, 1e5, 0.6,   1, "Planar"],

    [0.16959, 3556000, 0.252, 0, "Cellular"], # iteration 1 results
    [0.17122, 409500,  0.408, 1, "Planar"], # iteration 1 results
    [0.15980, 790600,  0.312, 1, "Planar"], # iteration 1 results
    [0.18510, 868500,  0.396,  0.5, "inconclusive"], # iteration 1 results
    [0.17449, 1265000, 0.372, 1, "Planar"], # iteration 1 results
    [0.15408, 212100,  0.372, 1, "Planar"], # iteration 1 results
    [0.18347, 3556000, 0.276, 0, "Cellular"], # iteration 1 results
    [0.16469, 175800,  0.408, 1, "Planar"], # iteration 1 results
    [0.18265, 193100,  0.432, 1, "Planar"], # iteration 1 results
    [0.15653, 4292000, 0.228, 0, "Cellular"], # iteration 1 results
]

# Note: Renaming the fourth column to "Truth" for clarity.
karma_df = pd.DataFrame(karma_data, columns=["composition", "G", "R", "Truth", "Explanation"])

# ---------------------------
# 2. Define the findPlanar Function
# ---------------------------
def findPlanar(G, c_0, R):
    G_value = G
    C_value = c_0
    Tl = 1751  # K
    Gamma = 3.47e-7  # K*m
    W0 = 1.0e-9  # m
    ke = 0.791  # unitless
    Dl = 3e-9  # m^2/s
    me = -3.49 * 100  # K/wt%
    TM = Tl - me * c_0
    Ts = TM + me / ke * c_0
    V0_d = Dl / W0  # m/s
    mu0_k = 0.3  # m/s·K
    alpha = 0.645
    S = 5
    b = np.log(ke) / 2
    a1_0 = 2 * np.sqrt(2) / 3
    Vd = 0.356 * np.log(1 / ke) / (1 - ke) * V0_d

    def partition_coefficient(Vspace):
        return (ke + Vspace / Vd) / (1 + Vspace / Vd)
    
    def liquidus_slope(V):
        kV = partition_coefficient(V)
        term1 = 1 - kV
        term2 = kV + (1 - kV) * alpha
        return me * (term1 + term2 * np.log(kV / ke)) / (1 - ke)
    
    def solidus_temperature(Vspace):
        kV = partition_coefficient(Vspace)
        mV = liquidus_slope(Vspace)
        T_solidus = TM + ((mV / kV) * c_0) - (Vspace / mu0_k)
        return T_solidus
    
    # Create a vector of Vspace values and find the one that maximizes T_solidus
    Vspace = np.logspace(np.log10(0.005), np.log10(2), 250)
    T_solidus = solidus_temperature(Vspace)
    V_ab = Vspace[np.argmax(T_solidus)]
    
    # Determine the prior classification:
    # 1 (Planar) if the input R is larger than V_ab, otherwise 0 (Cellular)
    classification = 1 if R > V_ab else 0
    return G_value, c_0, R, V_ab, classification

# ---------------------------
# 3. Compute Prior for the Grid Data
# ---------------------------
# Define grid ranges (feel free to adjust the number of points)
C_range = np.linspace(0.15, 0.19, 50)     # e.g., 50 points for composition
R_range = np.linspace(0.012, 0.6, 50)      # 50 points for R
G_range = np.logspace(5, 7, 50)         # 50 points for G

# Generate all possible combinations
param_combinations = list(itertools.product(C_range, R_range, G_range))
# Randomly sample 10,000 points from the full grid
sampled_indices = np.random.choice(len(param_combinations), 10000, replace=False)
#selected_params = [param_combinations[i] for i in sampled_indices]
selected_params = param_combinations

grid_data = []
for c_0, R, G in selected_params:
    G_val, C_val, R_val, V_ab_val, prior_class = findPlanar(G, c_0, R)
    grid_data.append({
        "Source": "Grid",
        "G": G_val,
        "C": C_val,
        "R": R_val,
        "Vab": V_ab_val,
        "Prior": prior_class,
        "Truth": np.nan,         # No truth value for grid data
        "Explanation": ""
    })

grid_df = pd.DataFrame(grid_data)

# ---------------------------
# 4. Compute Prior for the Karma Data
# ---------------------------
karma_prior_data = []
for idx, row in karma_df.iterrows():
    c_0 = row["composition"]
    G_val = row["G"]
    R_val = row["R"]
    truth = row["Truth"]
    explanation = row["Explanation"]
    G_val, C_val, R_val, V_ab_val, prior_class = findPlanar(G_val, c_0, R_val)
    karma_prior_data.append({
        "Source": "Karma",
        "G": G_val,
        "C": C_val,
        "R": R_val,
        "Vab": V_ab_val,
        "Prior": prior_class,
        "Truth": truth,
        "Explanation": explanation
    })

karma_prior_df = pd.DataFrame(karma_prior_data)

# ---------------------------
# 5. Combine and Export to CSV
# ---------------------------
combined_df = pd.concat([grid_df, karma_prior_df], ignore_index=True)
combined_df.to_csv("combined_solidus_karma_results_iteration_1_3labels.csv", index=False)
print("Combined CSV file saved as 'combined_solidus_karma_results_iteration_1_3labels.csv'")
