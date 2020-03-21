#%%
# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from toolkit import Viz, Tools, PF

#%%
# Pre configure modules
sns.set()

#%%[markdown]
# ---
# # Exploratory Analysis

#%%
# Get data
data, target_col, cat_cols, num_cols = PF.import_data()
X, y = data[cat_cols + num_cols], data[target_col]
X, zdense_cols, f0_col = PF.full_zero_mark(X)
X, strt_col = PF.add_strata(X, y)
cntr_cols = [f0_col, strt_col]

#%%
# Undersample Data
Xrs, yrs = PF.strat_undersample(X, y)
data_rs = pd.concat([Xrs, yrs], axis=1)

#%%[markdown]
# ---
# ## Exploration - Categorical Data

#%%[markdown]
# ### Device Data

#%%
# Add features from device
def device_features(X, y):
    X['L1'] = X['device'].str[0:2]
    X['L2'] = X['device'].str[2:4]
    X['L12'] = X['device'].str[0:4]
    #X['L3'] = X['device'].str[4:6]
    #X['L4'] = X['device'].str[6:]

    dev_cols = [c for c in ['L1', 'L2', 'L12', 'L3', 'L4'] if c in X.columns]

    return X, dev_cols

X, dev_cols = device_features(X, y)

#%%
# Visualize Device features
Viz.count_plot(X, y, dev_cols, proportion=True)

#%%[markdown]
# ### Device Data Conclusions
# - L1 and L2 groups data into few groups.
# - L1 may not be a good predictor since target is equally distributed among groups
# - L2 can be a good predictor since target is not equally distributed among groups
# - L12 seems to be the best option since target is divided in a more unequall way among groups
# - Further subleveling will not be usefull since it will have many values (sparse).

#%%[markdown]
# ### Date Data

#%%
# Define function to add features from date
def date_features(X, y):
    X['Y'] = X['date'].dt.year
    X['Q'] = X['date'].dt.quarter
    X['M'] = X['date'].dt.month
    X['D'] = X['date'].dt.day
    X['MW'] = X['D'].div(8).apply(np.floor).add(1).astype(int)
    X['WD'] = X['date'].dt.weekday
    X['WE'] = (X['WD'] > 4).astype(int)

    date_cols = [c for c in ['Y', 'Q', 'M', 'D', 'MW', 'WD', 'WE'] if c in X.columns]

    return X, date_cols

X, date_cols = date_features(X, y)

#%%
# Visualize Date features
Viz.count_plot(X, y, date_cols, proportion=True)

#%%[markdown]
# ### Date Data Conclusions
# - Y is not necessary since data is only for one year - 2015
# - WE may not be good predictor since does not add much info compared to WD.
# - Q may not be a good predictor since target values are mostly equally ditributed per group.
# - M, D, MW, WD, WE may be good predictors since target is equally distributed among groups.

#%%[markdown]
# ---
# ## Exploration - Numeric Data

#%%
# Create distribution plot per metric and target
# Remove zeros or nothing as desired
Viz.kernel_plot(X, y, num_cols, remove_val=[None], rug_sample=2500)

# %%
# Create distribution plot per metric and target for oversampled data
# Remove zeros or nothing as desired
Viz.kernel_plot(Xrs, yrs, num_cols, remove_val=[0], rug_sample=2500)

#%%
# Check correlation between metrics
corr_matrix = Tools.get_corr(X, num_cols)
Viz.corr_plot(corr_matrix)

#%%
# Check correlation between metrics for Non Full Zero rows
# Remove zeros or nothing as desired
corr_matrix = Tools.get_corr(X[X[f0_col] == 0], num_cols)
Viz.corr_plot(corr_matrix)

#%%[markdown]
# ### Numeric Data Conclusions
# For non zero inflated metrics:
# - M1, M5, M6 may not be good predictors since distribution of target variables are highly overlapped.
#
#
# For zero inflated netrics:
# - M2 may be good predictor. Extreme values are more frequent for failure. Ok values are highly concentrated 
# - M3 may be a good preditor since failure values are concentrated in lower values than non target (more spread).
# - M4 may be a googd predictor. Failure curve contains ok curve. Failure values can be more extreme.
# - M7 and M8 are dupicated (Correlation of 1). Just one of them can be goo predictor based on same argument as M4.
# - M9 may not be a good predictor since distributions are mostly overlapped (contained for target value).
#
#
# Correlation:
# - M7 and M8 are dupicated (Correlation of 1).
# - M3 and M9 are moderately correlated. This should be treated for logistic regression based ensemble model.
# - For the rest of the metrics there is low correlation between data.
#
#
# Outliers:
# - Since near 70% of all data will have all measures equal to zero. Outliers may retain the relevant information.
# - Outliers wont be removed. In many metrics will guide the criteria to identify failure.


# %%
