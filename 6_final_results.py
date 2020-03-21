#%%
# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from toolkit import Viz, Tools, PF

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#%%
# Pre configure modules
sns.set()
random_state = 42

#%%[markdown]
# ---
# # Test Results

#%%
# Get data
def reset_data():
    data, _, _, _ = PF.import_data()
    X, y = data[PF.cat_cols + PF.num_cols], data[PF.target_col]
    X = PF.add_features(X, y)
    X = PF.prepare_features(X, num_transform=False)

    strt = X[PF.strt_col]
    X.drop(PF.cntr_cols, axis=1, inplace=True)

    return data, X, y, strt

data, X, y, strt = reset_data()

X_train, X_test, strt_train, strt_test = train_test_split(X, strt, test_size=.2, random_state=random_state)
y_train, y_test = y[X_train.index], y[X_test.index]

#%%
# Get Stratified Bagg + Boost Model
bbb = PF.best_strat_bagg_boost(X_train, strt_train)
strt_proba = bbb.predict_proba(X_test)
y_proba = list(map(lambda v: v[2], strt_proba))

#%%
# Visualize results on test data 
bbb_mtrcs, bbb_thrsh, bbb_matrix = PF.get_metrics(y_test, y_proba, plot=True)

# %%
thresholds = [0.8, 0.85, 0.9, bbb_thrsh, 0.95, 0.965]

Viz.change_default(per_row=3)
ax = Viz.get_figure(6)

for i, t in enumerate(thresholds):
    y_pred = list(map(lambda v: 1 if v >= t else 0, y_proba))
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, vmin= 0, vmax=100, square=False, annot=True, fmt='d', ax=ax[i])
    ax[i].set_title(f'treshold: {round(t, 2)}')

Viz.change_default()
plt.show()


# %%
