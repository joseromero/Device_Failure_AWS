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

#%%
# Get data
data, target_col, cat_cols, num_cols = PF.import_data()
X, y = data[cat_cols + num_cols], data[target_col]

#%%[markdown]
# ---
# ## Target

#%%
# Calculate target proportions
target_prop = pd.concat([y.value_counts(), y.value_counts(normalize=True)], axis=1)
target_prop.columns = ['cnt', 'pct']
target_prop

#%%[markdown]
# ## Target Conclusions
# Imbalance
# - Target variable is highly unbalanced with a 999:1 ratio.
# - In order to prevent model to ignore failure patterns. Sampling strategy must be followed.
#
#
# Sampling Strategy
# - Oversampling may induce that variables appear to have lower variance than they do.
# - Undersampling on the other hand may induce that variables appear to have a higher variance than they do.
# - Mixing undersampling with bagging (ensemble models) may have a good result. Sampling for each estimator must follow a downsampling strategy. 
#
#
# Model Requirements
# - Best to use estimators that supports probability outcomes (logistic regression, decision trees) to keep a threshold control to optimize results.
# - Accuracy isn't a good metric to track for. F1-score and AUROC metrics will guide a better reduction of FN and FP.

#%%[markdown]
# ---
# ## Missing Values

#%%
# Check for zero value proportion by target
zero_pct = Tools.get_percentage(data[num_cols], y, values=[0])
mzero_pct = zero_pct.reset_index().melt(id_vars=target_col ,value_vars=num_cols, var_name='m', value_name='zero pct')
sns.barplot(x='m', y='zero pct', hue=target_col, data=mzero_pct, ax=Viz.get_figure(1))
plt.show()
zero_pct

#%%
# Check for non zero mean and std on all metrics separated by target
def nz_mean(c): return c[c > 0].mean()
def nz_std(c): return c[c > 0].std()

m_nzstats = data[num_cols + [target_col]].groupby(target_col)
m_nzstats = m_nzstats.agg([nz_mean, nz_std])
m_nzstats.stack()

#%%
# Check for mean and std on all metrics separated by target
m_stats = data[num_cols + [target_col]].groupby(target_col)
m_stats = m_stats.agg(['mean', 'std'])
m_stats.stack()

#%% 
# Check proportion when all measures are zero by target
zd_cols = ['m2', 'm3', 'm4', 'm7', 'm8', 'm9']
full_zero = pd.DataFrame(data[zd_cols].sum(axis=1) == 0, columns=['f0'])
zero_pct = Tools.get_percentage(full_zero, y, values=[True])
mzero_pct = zero_pct.reset_index().melt(id_vars=target_col ,value_vars='f0', var_name='m', value_name='zero pct')
sns.barplot(x='m', y='zero pct', hue=target_col, data=mzero_pct)
plt.show()
zero_pct

# %%
# Add full zero column to data
f0_col = 'f0'
X.insert(X.shape[1], 'f0', full_zero.astype(int))

#%%[markdown]
# ## Missing Values Conclusions
# Zero Dense Metrics
# - There are zero dense metrics (M2, M3, M4, M7, M8, M9). Where zero values represent more than 90% of metric's data in most cases and 78% for M9.
# - Zero dense metrics adds huge amount of noice in further analysis. It fully changes mean and std for instance
#
#
# Full Zeros
# - Near 68% of all data has only zero values in zero dense metrics.
# - It is important to mark such rows (F0) to guide a better sampling strategy in exploratory analysis. To reduce zero noice and better highlight patterns.
#
#
# Model Requirements
# - When ensembling the bagging model, full zero (F0) mark can help to perform a stratified undersampling. To control the amount of this zero noice in each estimator.

#%%[markdown]
# ---
# ## Undersampling

#%%
# Undersample data to desired proportions on strata
strt = pd.Series([0] * X.shape[0], index=X.index)
strt[(y == 0) & (X[f0_col] == 0)] = 1
strt[(y == 1)] = 2

strata_p = {0: 0.1, 1: 0.6, 2: 0.3}
random_state = 0

Xrs, strt_rs = Tools.stratified_undersample(X, strt, strata_p, random_state)
yrs = y.loc[Xrs.index]

#%%
# Get Undersample proportion for strata
strt_prop = pd.concat([strt_rs.value_counts(), strt_rs.value_counts(normalize=True)], axis=1)
strt_prop.columns = ['cnt', 'pct']
strt_prop

#%%
# Get Undersample proportion for target
target_prop = pd.concat([yrs.value_counts(), yrs.value_counts(normalize=True)], axis=1)
target_prop.columns = ['cnt', 'pct']
target_prop

#%%[markdown]
# ## Undersampling Conlusions
# For undersampling data, the following strata where defined.
# - Strata 0: Target = 0 and Full Zero Mark = 1 (Near to 70 % of data. Does not add much info since all metrics are 0)
# - Strata 1: Target = 0 and Full Zero Mark = 0 (Near to 30 % of data. Adds important info for non failure patterns)
# - Strata 2: Target = 1 (Near 0.1 % of data. Is all available info for failure data.)
# 
# 
# Undersampling strategy
# - Undersample will always have all Strata 2 data. This Strata determines full undersample size.
# - Will use a Strata 0, Strata 1 ratio (15:85). This is the same proportion seen in failure devices when full zero mark is present. This will ensure an homogeneous undersample.
# - If zero noice is still an obstacle, Strata 0 proportion may be reduced.
# - IMPORTANT: For data exploration, working with a single sample may bias conclusions since mcuh information for non failure devices is ommited. But it will help focus on failing devices patterns.
#
# Model Prerquisites
# - Sample undersampling strategy will be followed for estimators in bagging model.
# - A coherent number of estimators will compensate the undersampling effect on non failing devices info.
# - The Knowledge of Crowds principle will be leveraged by this sampling strategy.

# %%
