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
# # Data Preparation

#%%
# Get data
def reset_data():
    data, _, _, _ = PF.import_data()
    X, y = data[PF.cat_cols + PF.num_cols], data[PF.target_col]
    X = PF.add_features(X, y)

    return data, X, y

data, X, y = reset_data()

#%%[markdown]
# ## Metric Transformations

#%%
# Function to perform transformations by column
def transform(X, c_trans):
    for c, t in c_trans.items():
        if t == 'yj': xt = Tools.power_transform(X[[c]], standardize=True)
        elif t == 'qt': xt = Tools.quantile_transform(X[[c]])
        elif t == 'nrm': xt = Tools.standardize(X[[c]], which='normal')
        elif t == 'mnmx': xt = Tools.standardize(X[[c]], which='minmax')
        else: xt = X[[c]]

        X[c] = xt[c]

    return X

c_trans = {}

#%%[markdown]
# ## Base Metrics Transformations

# %%
# Visualize base metrics distributions
Viz.kernel_plot(X, y, c_plot=PF.num_cols, remove_val=[None], rug_sample=2500)

#%%
# Transform base metrics
data, X, y = reset_data()
c_trans['m'] = {'m1': 'qt', 'm2':'nrm', 'm3':'nrm', 
                'm4':'nrm', 'm5':'qt', 'm6':'qt',
                'm7':'nrm', 'm9':'nrm'}

X = transform(X, c_trans['m'])
data = pd.concat([X, y], axis=1)

# %%
# Visualize transformed metrics
Viz.kernel_plot(X, y, c_plot=c_trans['m'].keys(), remove_val=[None], rug_sample=2500)

# %% [markdown]
# ## Base Metrics Transformations - Conclusions
# Xyj Transformation seem to not add much value to data than a simple standarization.
# Xqt Transformation makes a good job approximating M1, M5, M6 to normal distributions. Overlap between target remains and even is powered.
# Xstd / Xmnmx shifts and scales data but leaves original distribution.
#
# Transformation Decisions:
# - Transformations will be required only for logistic regression models, since its ideal to persue normality in data.
# - M1, M5, M6 will be Quartile transformed.
# - M2, M3, M4, M7, M9 will be normal standardized so that everithing is in terms of normal standard deviations.

#%%[markdown]
# ## Change Metrics Transformations

# %%
# Visualize change metrics
Viz.kernel_plot(X, y, c_plot=PF.chng_cols, remove_val=[None], rug_sample=2500)

# %%
data, X, y = reset_data()
c_trans['ch'] = {'ch1': 'mnmx', 'ch2':'mnmx', 'ch4':'mnmx', 
                 'ch5':'mnmx', 'ch6':'mnmx', 'ch7':'mnmx'}

X = transform(X, c_trans['ch'])
data = pd.concat([X, y], axis=1)

# %%
# Visualize transformed change metrics
Viz.kernel_plot(X, y, c_plot=c_trans['ch'].keys(), remove_val=[None], rug_sample=2500)

# %% [markdown]
# ## Change Metrics Transformations - Conclusions
# Transformation Decisions:
# - Change percentage by itself is expected to be normalized, so transformations won't make much difference.
# - Each change metric uses different dimensions. A Max Min standardization will translate into a common dimensionality.

#%%[markdown]
# ## Rank Metrics Transformations

# %%
# Visualize rank metrics
Viz.kernel_plot(X, y, c_plot=PF.rnk_cols, remove_val=[None], rug_sample=2500)

# %%
data, X, y = reset_data()
c_trans['sr'] = {'sr2': 'none', 'sr3':'none', 'sr4':'none', 
                 'sr7':'none', 'sr9':'none', 'srmx':'none'}

X = transform(X, c_trans['sr'])
data = pd.concat([X, y], axis=1)

# %%
# Visualize transformed change metrics
Viz.kernel_plot(X, y, c_plot=c_trans['sr'].keys(), remove_val=[None], rug_sample=2500)

# %% [markdown]
# ## Rank Metrics Transformations - Conclusions
# Transformation Decisions:
# - No transformation or standarization is required. Data is spread between 0 and 1 already.

#%%[markdown]
# ---
# # Multicolinearity

#%%[markdown]
# ## Metrics Correlation

#%%
# Perform all transformations on data
c_trans['m'] = {'m1': 'qt', 'm2':'nrm', 'm3':'nrm', 'm4':'nrm', 
                'm5':'qt', 'm6':'qt', 'm7':'nrm', 'm9':'nrm'}
c_trans['ch'] = {'ch1': 'mnmx', 'ch2':'mnmx', 'ch4':'mnmx', 
                 'ch5':'mnmx', 'ch6':'mnmx', 'ch7':'mnmx'}
c_trans['sr'] = {'sr2': 'none', 'sr3':'none', 'sr4':'none', 
                 'sr7':'none', 'sr9':'none', 'srmx':'none'}

data, X, y = reset_data()
for k in c_trans.keys(): X = transform(X, c_trans[k])
data = pd.concat([X, y], axis=1)

#%%
# Check correlation between rank metrics and base metrics
corr_matrix = Tools.get_corr(X, PF.nnum_cols)
Viz.change_default(per_row=1)
Viz.corr_plot(corr_matrix)

#%%[markdown]
# ## Metrics Correlation - Conclusion
# - Correlation is minimum among metrics after transformations.
# - Few cases need to be assesed. PCA can remove all collinearity.

#%%[markdown]
# ## PCA (Principal Component Analysis)

# %%
# Test PCA on zero dense measures
c_pca = PF.nnum_cols
Xpca, pca_var = Tools.pca(X, c_pca)
pca_cumvar = np.cumsum(pca_var)

Viz.screeplot(pca_var, Xpca.columns)

pca_var = [{'var':pca_var[i], 'cumvar':pca_cumvar[i]} for i, _ in enumerate(Xpca.columns)]
pca_var = pd.DataFrame(pca_var, index=Xpca.columns)
pca_var.transpose()

#%%
# Check correlation between pca components
corr_matrix = Tools.get_corr(Xpca, Xpca.columns)
Viz.corr_plot(corr_matrix)

#%%
# Create distribution plot per pca and target
Viz.change_default()
Viz.kernel_plot(Xpca, y, c_plot=Xpca.columns, remove_val=[None], rug_sample=2500)

#%%[markdown]
# ## PCA - Conclusion
# - PCA decomposition will eliminate any multicolinearity present in data.
# - Although last 10 components explains close to 1% of variance, they will not be removed since target itself has a 999:1 unbalance.
# - Additionally in many of this last components, failure values are more easilly separable which will enhance model performance.

#%%[markdown]
# ## Category Encoding

#%%
cat_cols = [c for c in PF.ncat_cols if c not in PF.cat_cols]
Xenc, enc_cols = Tools.oh_encode(X[cat_cols], cat_cols)
Xenc.shape[1]

#%%[markdown]
# ## Category Encoding - Conclusion
# - One Hot Encoding was performed eliminating one value per category to avoy redundance. 
# - Dimensionality increased. If it is an issue for model performance, reduction techniques will be designed.