#%%
# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from toolkit import Viz, Tools, PF

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score

from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


#%%
# Pre configure modules
sns.set()
random_state = 42

#%%[markdown]
# ---
# # Model Definition

#%%
# Get data
def reset_data(num_transform=True):
    data, _, _, _ = PF.import_data()
    X, y = data[PF.cat_cols + PF.num_cols], data[PF.target_col]
    X = PF.add_features(X, y)
    X = PF.prepare_features(X, num_transform=True)

    strt = X[PF.strt_col]
    X.drop(PF.cntr_cols, axis=1, inplace=True)

    return data, X, y, strt

data, X, y, strt = reset_data(num_transform=True)


#%%[markdown]
# ## Train / Test Split

# %%
# Split data
X_train, X_test, strt_train, strt_test = train_test_split(X, strt, test_size=.2, random_state=random_state)
y_train, y_test = y[X_train.index], y[X_test.index]

# %%
# Check the proportion of target within train / test sets.
trgt_prop, trgt_cnt = pd.DataFrame(), y.value_counts()
trgt_prop['train cnt'] = y_train.value_counts()
trgt_prop['train pct'] = trgt_prop['train cnt'].div(trgt_cnt)
trgt_prop['test cnt'] = y_test.value_counts()
trgt_prop['test pct'] = trgt_prop['test cnt'].div(trgt_cnt)
trgt_prop

# %%
# Check the proportion of strata within train / test sets.
strt_prop, strt_cnt = pd.DataFrame(), strt.value_counts()
strt_prop['train cnt'] = strt_train.value_counts()
strt_prop['train pct'] = strt_prop['train cnt'].div(strt_cnt)
strt_prop['test cnt'] = strt_test.value_counts()
strt_prop['test pct'] = strt_prop['test cnt'].div(strt_cnt)
strt_prop

# %% [markdown]
# ## Train / Test Split - Conclusions
# - Data is split evenly between train and test sets. Strata and target proportions follow split ratio as well.
# - A 80/20 ratio will be used.

#%%[markdown]
# ## Ensemble Model

#%%[markdown]
# ### Number of Estimators

#%%
# Get strata undersample proportions and counts
strata_p = PF.strata_p
n_vals = Tools.strat_undersample_counts(strt_train, strata_p, random_state)

#%%
# Compare for Strata 1
t_vals = strt.value_counts()
proxy_estimators = t_vals.loc[1] / n_vals[1]
proxy_estimators = int(round(proxy_estimators, 0))
proxy_estimators

#%%[markdown]
# ### Number of Estimators - Conclusions
# - When stratified undersampling, all values of Strata 2 will be kept in each subsample, Strata 0 and Strata 1 will be undersampled.
# - Strata 0 provides few to no valuable information since are full zero instances on base zero dense columns.
# - Strata 1 provide valuable information for ok target. This should be the main focus to estimate the number of predictors.
# - Because of the high imbalance in target variable, bootstraping should be avoided to guarantee that most original data is used.
# - The proxy estimator is the proportion between the number of Strata 1 elements in a subsample with all the available ones.
# - The proxy estimator is approx 220 estimators (if all subsample where different).
# - 1000 estimators will be used per ensemble model. This will help in the inclusion of most original data (considering that samples can have similar instances).


#%%[markdown]
# ## Ensemble Candidates

#%%
# Create results variable
results = {}

#%%[markdown]
# ### Bagging with Logistic Regression

#%%
# Get preprocessed data and split
data, X, y, strt = reset_data(num_transform=True)
X_train, X_test, strt_train, strt_test = train_test_split(X, strt, test_size=.2, random_state=random_state)
y_train, y_test = y[X_train.index], y[X_test.index]

#%%[markdown]
# ### Candidate 1: Stratifed Undersampling per Estimator
# Ensemble Characteristics
# - All estimators will have a subsample based on stratified undersampling strategy.
# - The new target variable to use in training will be strata. Multiclass (3 possible values) problem.
# - Probability will be requested for threshold analysis.
# - Probability for Strata 0 and Strata 1 will be added into Target 0 (No Failure), Strata 2 will be Target 1 (Failure)
#
#
# Logistic Regression Characteristics
# - No hyperparamether limitations for now. We are evaluating best ensemble strategy for now.

#%%
# Build Candidate 1
strata_p = PF.strata_p
n_vals = Tools.strat_undersample_counts(strt_train, strata_p, random_state)

def candidate1(X_train, strt_train, strata_p, n_vals):
    logreg = LogisticRegression(class_weight=strata_p)

    bbagg = BalancedBaggingClassifier(base_estimator=logreg, sampling_strategy=n_vals,
                                      n_estimators=proxy_estimators, bootstrap=False, n_jobs=10,
                                      random_state=random_state)
    bbagg.fit(X_train, strt_train)
    return bbagg

model1 = candidate1(X_train, strt_train, strata_p, n_vals)
strt_proba1 = model1.predict_proba(X_train)
y_proba1 = list(map(lambda v: [v[0] + v[1], v[2]], strt_proba1))

results[1] = {'name':'StrtBaggLog', 'y_proba': y_proba1}

# %%
# ### Candidate 2: Undersampling per Estimator
# Ensemble Characteristics
# - All estimators will have a traditional undersampling strategy. 
# - The proportion per sample will be 30% failure, 70% non failure.s
# - The problem will remain as binary.
# - Probability will be requested for threshold analysis.
#
#
# Logistic Regression Characteristics
# - No hyperparamether limitations for now. We are evaluating best ensemble strategy for now.

#%%
# Build Candidate 2
p = 0.3
r = p/(1-p)

target_p = {0:1-p, 1:p}

def candidate2(X_train, y_train, target_p, r):
    logreg = LogisticRegression(class_weight=target_p)

    bbagg = BalancedBaggingClassifier(base_estimator=logreg, sampling_strategy=r,
                                      n_estimators=proxy_estimators, bootstrap=False, n_jobs=10,
                                      random_state=random_state)

    bbagg.fit(X_train, y_train)
    return bbagg

model2 = candidate2(X_train, y_train, target_p, r)
y_proba2 = model2.predict_proba(X_train)

results[2] = {'name':'UndBaggLog', 'y_proba': y_proba2}

#%%[markdown]
# ### Random Forest

# %%
# Get preprocessed data (no numeric transformation needed) and split
data, X, y, strt = reset_data(num_transform=False)
X_train, X_test, strt_train, strt_test = train_test_split(X, strt, test_size=.2, random_state=random_state)
y_train, y_test = y[X_train.index], y[X_test.index]

#%%[markdown]
# ### Candidate 3: Stratifed Undersampling per Estimator
# Ensemble Characteristics
# - All estimators will have a subsample based on stratified undersampling strategy.
# - The new target variable to use in training will be strata. Multiclass (3 possible values) problem.
# - Probability will be requested for threshold analysis.
# - Probability for Strata 0 and Strata 1 will be added into Target 0 (No Failure), Strata 2 will be Target 1 (Failure)
#
#
# Decision Tree
# - No hyperparamether limitations for now. We are evaluating best ensemble strategy for now.

#%%
# Build Candidate 3
strata_p = PF.strata_p
n_vals = Tools.strat_undersample_counts(strt_train, strata_p, random_state)

def candidate3(X_train, strt_train, strata_p, n_vals):
    brf = BalancedRandomForestClassifier(class_weight=strata_p, sampling_strategy=n_vals,
                                        n_estimators=proxy_estimators, bootstrap=False, n_jobs=10,
                                        random_state=random_state)

    brf.fit(X_train, strt_train)
    return brf

model3 = candidate3(X_train, strt_train, strata_p, n_vals)
strt_proba3 = model3.predict_proba(X_train)
y_proba3 = list(map(lambda v: [v[0] + v[1], v[2]], strt_proba3))

results[3] = {'name':'StrtRandFor', 'y_proba': y_proba3}

# %%
# ### Candidate 4: Undersampling per Estimator
# Ensemble Characteristics
# - All estimators will have a traditional undersampling strategy. 
# - The proportion per sample will be 30% failure, 70% non failure.
# - The problem will remain as binary.
# - Probability will be requested for threshold analysis.
#
#
# Decision Tree
# - No hyperparamether limitations for now. We are evaluating best ensemble strategy for now.

# %%
# Build Candidate 4
p = 0.3
r = p/(1-p)

target_p = {0:1-p, 1:p}

def candidate4(X_train, y_train, target_p, r):
    brf = BalancedRandomForestClassifier(class_weight=target_p, sampling_strategy=r,
                                        n_estimators=proxy_estimators, bootstrap=False, n_jobs=10,
                                        random_state=random_state)
                                    
    brf.fit(X_train, y_train)
    return brf

model4 = candidate4(X_train, y_train, target_p, r)
y_proba4 = model4.predict_proba(X_train)

results[4] = {'name':'UndRandFor', 'y_proba': y_proba4}

#%%[markdown]
# ### Bagging + Boosting

# %%
# Get preprocessed data (no numeric transformation needed) and split
data, X, y, strt = reset_data(num_transform=False)
X_train, X_test, strt_train, strt_test = train_test_split(X, strt, test_size=.2, random_state=random_state)
y_train, y_test = y[X_train.index], y[X_test.index]

#%%[markdown]
# ### Candidate 5: Stratifed Undersampling per Estimator
# Ensemble Characteristics
# - All estimators will have a subsample based on stratified undersampling strategy.
# - The new target variable to use in training will be strata. Multiclass (3 possible values) problem.
# - Probability will be requested for threshold analysis.
# - Probability for Strata 0 and Strata 1 will be added into Target 0 (No Failure), Strata 2 will be Target 1 (Failure)
#
#
# AdaBoost
# - Decision Trees among Adaboost wont be limited to 1 level (stump)
# - Number of estimates within each Adaboost will be limited to 10 to avoy long training times.
# - No additional hyperparamether limitations for now. We are evaluating best ensemble strategy for now.

#%%
# Build Candidate 5
strata_p = PF.strata_p
n_vals = Tools.strat_undersample_counts(strt_train, strata_p, random_state)

def candidate5(X_train, strt_train, strata_p, n_vals):
    tree = DecisionTreeClassifier(class_weight=strata_p)
    
    boost = AdaBoostClassifier(base_estimator=tree, n_estimators=10)

    bbagg = BalancedBaggingClassifier(base_estimator=boost, sampling_strategy=n_vals,
                                      n_estimators=proxy_estimators, bootstrap=False, n_jobs=10,
                                      random_state=random_state)
    bbagg.fit(X_train, strt_train)
    return bbagg

model5 = candidate5(X_train, strt_train, strata_p, n_vals)
strt_proba5 = model5.predict_proba(X_train)
y_proba5 = list(map(lambda v: [v[0] + v[1], v[2]], strt_proba5))

results[5] = {'name':'StrtBaggBst', 'y_proba': y_proba5}

# %%
# ### Candidate 6: Undersampling per Estimator
# Ensemble Characteristics
# - All estimators will have a traditional undersampling strategy. 
# - The proportion per sample will be 30% failure, 70% non failure.
# - The problem will remain as binary.
# - Probability will be requested for threshold analysis.
#
#
# AdaBoost
# - Decision Trees among Adaboost wont be limited to 1 level (stump)
# - Number of estimates within each Adaboost will be limited to 10 to avoy long training times.
# - No additional hyperparamether limitations for now. We are evaluating best ensemble strategy for now.

# %%
# Build Candidate 6
p = 0.3
r = p/(1-p)

target_p = {0:1-p, 1:p}

def candidate6(X_train, y_train, target_p, r):
    tree = DecisionTreeClassifier(class_weight=target_p)
    
    boost = AdaBoostClassifier(base_estimator=tree, n_estimators=10)

    bbagg = BalancedBaggingClassifier(base_estimator=boost, sampling_strategy=r,
                                      n_estimators=proxy_estimators, bootstrap=False, n_jobs=10,
                                      random_state=random_state)
    bbagg.fit(X_train, y_train)
    
    return bbagg

model6 = candidate6(X_train, y_train, target_p, r)
y_proba6 = model6.predict_proba(X_train)

results[6] = {'name':'UndBaggBst', 'y_proba': y_proba6}

#%%[markdown]
# ### Result Analysis

# %%
# Calculate Scores and Confusion Matrixes
threshold = 0.95

def get_scores(threshold, results):
    conf_matrixes, scores = {}, []
    for k, r in results.items(): 
        y_proba = r['y_proba']
        y_pred = list(map(lambda v: 1 if v[1] >= threshold else 0, y_proba))
        
        f1 = f1_score(y_train, y_pred)
        recall = recall_score(y_train, y_pred)
        precision = precision_score(y_train, y_pred)
        auroc = roc_auc_score(y_train, y_proba)

        conf_matrix = confusion_matrix(y_train, y_pred)
        conf_matrix = pd.DataFrame(conf_matrix, index=y_train.unique(), columns=y_train.unique())

        scores.append({'id':k, 'name': r['name'], 'F1-score': f1, 'Recall': recall, 'Precision': precision, 'AUROC':auroc})
        conf_matrixes[k] = conf_matrix

    scores = pd.DataFrame(scores).set_index('id')

    return scores, conf_matrixes

scores, conf_matrixes = get_scores(threshold, results)

# %%
# Visualize Scores per Candidate
m_scores = scores.melt(id_vars=['name'], var_name='metric', value_name='score')
sns.barplot(data=m_scores, x='name', y='score', hue='metric', ax=Viz.get_figure(1))
plt.show()

# %%
# Visualize Confusion Matrixes
Viz.change_default(per_row=3)
ax = Viz.get_figure(6)

for i, conf_matrix in conf_matrixes.items():
    sns.heatmap(conf_matrix, vmin= 0, vmax=200, square=True, annot=True, fmt='d', ax=ax[i-1])
    ax[i-1].set_title(results[i]['name'])

plt.show()

#%%[markdown]
# ### Result Analysis - Conclusions
# - Threshold used is 95% of probability of failure. This will narrow predictions giving a best score for each ensemble.
# - No hyperparamether was tuned to evaluate ensemble strategy only.
# - Accuracy won´t be used as decision score. 
# - Stratified Undersampling shows better results than Undersampling. This is true for all bagging strategies (no matter estimators' algorithm.)
# - Logistic Regression based models don't show good results in this run.
# - Stratified Random Forest and Bagg + Boost models show best behaviour. They ilustrate the Recall / Precision tradeoff.
# - Stratified Random Forest favors precision.
# - Stratifies Bagg + Boost favors recall.
# - Both models will be tuned, but from this point is imortant to understand the impact of False Pasitives and False Negatives to better select a final model.