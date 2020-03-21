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
# # Derived Variables

#%%
# Get Data
data, target_col, cat_cols, num_cols = PF.import_data()
X, y = data[cat_cols + num_cols], data[target_col]

X, zdense_cols, f0_col = PF.full_zero_mark(X)
X, strt_col = PF.add_strata(X, y)
X, num_cols = PF.remove_duplicates(X)
data = pd.concat([X, y], axis=1)
cntr_cols = [f0_col, strt_col]

#%%[markdown]
# ---
# ## Time Based Metrics

#%%[markdown]
# ### Absolute Percentage Change

# %%
# Get absolute percentage change for metrics and day difference for date (elapsed)
def change_metrics(data):
    # Sort data 
    data_sort = data.sort_values(cat_cols)
    Xsort, ysort = data_sort[cat_cols + num_cols + cntr_cols], data_sort[target_col]

    # Create elapsed column
    elapsed = Xsort[cat_cols].groupby('device').diff().bfill()['date']

    # Create absolute change dataframe
    Xchng = Xsort[['device'] + num_cols].groupby('device').pct_change().fillna(0).abs()
    Xchng.columns = [c.replace('m', 'ch') for c in Xchng.columns]
    Xchng.insert(0, 'device', Xsort['device'])
    Xchng.insert(1, 'date', Xsort['date'])
    Xchng.insert(2, 'elapsed', elapsed.dt.days)
    Xchng.insert(Xchng.shape[1], cntr_cols[0], Xsort[cntr_cols[0]])
    Xchng.insert(Xchng.shape[1], cntr_cols[1], Xsort[cntr_cols[1]])

    chng_cols = [c for c in Xchng.columns if c not in cat_cols + cntr_cols]
    data_chng = pd.concat([Xchng, ysort], axis=1).sort_index()

    return data_chng, chng_cols

data_chng, chng_cols = change_metrics(data)
Xchng = data_chng[cat_cols + chng_cols + cntr_cols]

#%%
# Check for inf value proportion by target
inf_pct = Tools.get_percentage(Xchng[chng_cols], y, values=[np.inf])
minf_pct = inf_pct.reset_index().melt(id_vars=target_col ,value_vars=chng_cols, var_name='ch', value_name='inf pct')
sns.barplot(x='ch', y='inf pct', hue=target_col, data=minf_pct, ax=Viz.get_figure(1))
plt.show()
inf_pct

#%%
# Replace inf values and undersample
Xchng = Xchng.replace([np.inf], 0)
data_chng = pd.concat([Xchng, y], axis=1)
Xchng_rs, yrs = PF.strat_undersample(Xchng, y)

#%%
# Check for zero value proportion by target
zero_pct = Tools.get_percentage(Xchng[chng_cols], y, values=[0])
mzero_pct = zero_pct.reset_index().melt(id_vars=target_col ,value_vars=chng_cols, var_name='ch', value_name='zero pct')
sns.barplot(x='ch', y='zero pct', hue=target_col, data=mzero_pct, ax=Viz.get_figure(1))
plt.show()
zero_pct

# %%
# Create distribution plot per new metric and target
# Can remove zero values if desired
Viz.kernel_plot(Xchng, y, chng_cols, remove_val=[None], rug_sample=2500)

# %%
# Create distribution plot per new metric and target for undersample data
Viz.kernel_plot(Xchng_rs, yrs, chng_cols, remove_val=[None], rug_sample=2500)

#%%
# Check correlation between change metrics
corr_matrix = Tools.get_corr(Xchng, chng_cols)
Viz.corr_plot(corr_matrix)

#%%[markdown]
# ### Metrics Percentage Change - Conclusions
# Infinite Values
# - Note that for any metric, if for subsequent days, metric value changes from zero, the change metric will be +/- inf.
# - This elements were replaced by 0, since it ocurrence is really low.
# - It's important to analyze a switch-like variable that respresents this changes (from / to zero)
# - When checking +/- inf proportions, few cases may indicate failure but is not quite representative.  
#
#
# Absolute Percentage Change
# - Elapsed (days between subsequent metrics for each device), may not be a good predictor. Failure curve is contained by the ok curve. Perhaps extreme cases are more likely for failure (shifted mean and wider std).
# - CH3, CH9, are bad predictors. It seems that all values for failure are 0 or +/- inf, so no relevant info is added.
# - CH1 seems to be a good predictor. Failure values are really concentraded, while ok values are wide spread and centered appart of failure main concentration.
# - CH2, CH4, CH6 seem to be good predictors. Ok values will almost always be around 0. Failure curve has a wider spectrum (covering oks curve).
# - CH5, CH7 seem to be good predictors as well. Same explanation than CH2, CH4.
# - It is important to highlight that once again, 0 values will represent the mayority of data for values derived from zero intessive original columns. This means that curves are represented mainly by really few cases.
# - Because of this, oversampling can be of good use when building the model
#
#
# Correlation
# - There is not much correlation between change metrics.
# - Elapsed and CH5 show a moderate correlation (seems random because of metrics definition).

#%%[markdown]
# ### On / Off Switch

#%%
# Get on / off switch (when metric changes from 0 or to 0 in subsequent measures)
def switch_metrics(data):
    # Sort data 
    data_sort = data.sort_values(cat_cols)
    Xsort, ysort = data_sort[cat_cols + num_cols + cntr_cols], data_sort[target_col]

    # Create absolute change dataframe
    Xswtch = pd.concat([Xsort[['device']], (Xsort[num_cols] != 0).astype(int)], axis=1)
    Xswtch = Xswtch.groupby('device').diff().fillna(0).astype(int)
    Xswtch.columns = [c.replace('m', 'sw') for c in Xswtch.columns]
    
    Xswtch.insert(0, 'device', Xsort['device'])
    Xswtch.insert(1, 'date', Xsort['date'])
    Xswtch.insert(Xswtch.shape[1], cntr_cols[0], Xsort[cntr_cols[0]])
    Xswtch.insert(Xswtch.shape[1], cntr_cols[1], Xsort[cntr_cols[1]])

    swtch_cols = [c for c in Xswtch.columns if c not in cat_cols + cntr_cols]
    data_swtch = pd.concat([Xswtch, ysort], axis=1).sort_index()

    return data_swtch, swtch_cols

data_swtch, swtch_cols = switch_metrics(data)
Xswtch = data_swtch[cat_cols + swtch_cols + cntr_cols]

# %%
# Check for zero value proportion by target
Tools.get_percentage(Xswtch[swtch_cols], y, values=[0])

# %%
# Check for 1 / -1 value proportion by target
Tools.get_percentage(Xswtch[swtch_cols], y, values=[-1, 1])

# %%
# Check for 1 value proportion by target
Tools.get_percentage(Xswtch[swtch_cols], y, values=[1])

# %%
# Check for -1 value proportion by target
Tools.get_percentage(Xswtch[swtch_cols], y, values=[-1])

#%%[markdown]
# ### On / Off Switch - Conclusions
# Proportions
# - Definatelly this is not a good metric to use
# - Values are to few to be relevant (for failure and ok target)


#%%[markdown]
# ---
# ## Aggregate Metrics

#%%[markdown]
# ### Average / Maximum Standardized Rank

#%%
# Get rank metrics
def rank_metrics(data):
    # Create rank dataframe
    Xrnk = data[num_cols].rank(method='min').sub(1).div(data.shape[0])
    #Xrnk = Tools.standardize(data[num_cols], which='minmax')
    Xrnk.columns = [c.replace('m', 'sr') for c in Xrnk.columns]
    Xrnk['srav'] = Xrnk.mean(axis=1)
    Xrnk['srmx'] = Xrnk.max(axis=1)

    Xrnk.insert(0, 'device', data['device'])
    Xrnk.insert(1, 'date', data['date'])
    Xrnk.insert(Xrnk.shape[1], cntr_cols[0], data[cntr_cols[0]])
    Xrnk.insert(Xrnk.shape[1], cntr_cols[1], data[cntr_cols[1]])

    rnk_cols = [c for c in Xrnk.columns if c not in cat_cols + cntr_cols]
    data_rnk = pd.concat([Xrnk, y], axis=1).sort_index()

    return data_rnk, rnk_cols

data_rnk, rnk_cols = rank_metrics(data)
Xrnk = data_rnk[cat_cols + rnk_cols + cntr_cols]
Xrnk_rs, yrs = PF.strat_undersample(Xrnk, y)

# %%
# Check for zero value proportion by target
Tools.get_percentage(Xrnk[rnk_cols], y, values=[0])

# %%
# Create distribution plot per new metric and target
# Can remove zero if requiered (zero percentiles are for zero values)
Viz.kernel_plot(Xrnk, y, rnk_cols, remove_val=[0], rug_sample=2500)

# %%
# Create distribution plot per new metric and target on undersample data
Viz.kernel_plot(Xrnk_rs, yrs, rnk_cols, remove_val=[None], rug_sample=2500)

#%%
# Check correlation between rank metrics
corr_matrix = Tools.get_corr(Xrnk, rnk_cols)
Viz.corr_plot(corr_matrix)

#%%
# Check correlation between rank metrics and base metrics
corr_matrix = Tools.get_corr(pd.concat([Xrnk, X[num_cols]], axis=1), num_cols + rnk_cols)
Viz.corr_plot(corr_matrix)

#%%[markdown]
# ### Average / Maximum Standardized Rank
# Metric Standardized Rank
# - It is spected that metric rank will follow a similar behaviour than raw metric.
# - SR1, SR5, SR6 seems to not add much additional information.
# - SR2, SR3, SR4, SR7 show a wider spectrum can be useful to identify failure (although is similar behaviour compared to original metrics)
#
#
# Average / Max Standardized Rank
# - One common conclusion among metrics is that failure can be identified in extreme values.
# - It seems that average / max percentile make a good job accumulating such extreme values.
#
# 
# Correlation
# - Note that SR1/M1, SR5/M5, SR6/M6 are highly correlated. Perhaps SR1, SR5, SR6 are redundant.
# - Same applies to SRAV and SRMX.
# - For logistic regression based models. Perhaps PCA can help reduce collineality as well as reduce unecessary features.

# %%
