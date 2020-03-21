#%%
# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

#%%
# Pre configure modules
sns.set()

#%%
# Define functions
class Fun():
    # Function to import data
    @classmethod
    def import_data(cls):
        # Import data, remove duplicates and set index
        data = pd.read_csv('data/device_failure.csv', parse_dates=['date'], encoding='ISO-8859-1')
        data.drop_duplicates(inplace=True)
        data.set_index(['device', 'date'], inplace=True)

        # Rename / Reorder columns, asssign target and features
        data.columns = [c.replace('attribute', 'm') if c != 'failure' else c for c in data.columns]
        target_col =  'failure'
        feature_cols = sorted(list(set(data.columns) - set([target_col])))
        data = data[feature_cols + [target_col]]

        return data, target_col, feature_cols

    #Define function to filter zero values for specified cols
    @classmethod
    def filter_zeros(cls, c_check, X):
        query = ' | '.join([f"({c} > 0)" for c in c_check])
        return X.query(query)
    
    # Function to get outliers according to IQR or Z score.
    @classmethod
    def outlier_limits(cls, d, method='IQR'):
        if method == 'IQR':
            q1, q3 = d.quantile(.25), d.quantile(.75)
            iqr = q3 - q1
            llim, ulim = q1 - (1.5 * iqr), q3 + (1.5 * iqr)
        else:
            mean, std = d.mean(), d.std()
            llim, ulim = mean-(3*std), mean+(3*std)
        
        return llim, ulim

    # Function to plot deatiled view for a column   
    @classmethod 
    def detail_plot(cls, d, outliers=True):
        _, axs = plt.subplots(2, 1, figsize=(20, 7), sharex=True, gridspec_kw={'height_ratios': [6, 1]})

        sns.distplot(d, axlabel=False, ax=axs[0])
        
        if outliers:
            mean, std = d.mean(), d.std()
            llim, ulim = mean-(3*std), mean+(3*std)
        
            kde_x, kde_y = axs[0].lines[0].get_data()
            _, _ = axs[0].axvline(x=llim,color='red'), axs[0].axvline(x=ulim,color='red')
            axs[0].fill_between(kde_x, kde_y, where=(kde_x<llim) | (kde_x>ulim) , interpolate=True, color='#EF9A9A')

        sns.boxplot(d, ax=axs[1])
        plt.show()

    # Function to power transform data
    @classmethod
    def power_transform(cls, X_train, X_test, standardize=True):
        # Perform power transform
        yj = PowerTransformer(method='yeo-johnson', standardize=standardize)
        yj_train = yj.fit_transform(X_train)
        yj_test = yj.transform(X_test)
        lmbdas = yj.lambdas_
        
        yj_train = pd.DataFrame(yj_train, index=X_train.index, columns=X_train.columns)
        yj_test = pd.DataFrame(yj_test, index=X_test.index, columns=X_test.columns)

        return yj_train, yj_test, lmbdas

    # Function to quantile transform data
    @classmethod
    def quantile_transform(cls, X_train, X_test):
        # Perform quantile transform
        rng = np.random.RandomState(304)
        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=rng)
        qt_train = qt.fit_transform(X_train)
        qt_test = qt.transform(X_test)

        qt_train = pd.DataFrame(qt_train, index=X_train.index, columns=X_train.columns)
        qt_test = pd.DataFrame(qt_test, index=X_test.index, columns=X_test.columns)

        return qt_train, qt_test

    # Function to standardize data
    @classmethod
    def standardize(cls, X_train, X_test):
        std = StandardScaler()
        std_train = std.fit_transform(X_train)
        std_test = std.transform(X_test)
        stats = {'mean': std.mean_, 'std':std.scale_}

        std_train = pd.DataFrame(std_train, index=X_train.index, columns=X_train.columns)
        std_test = pd.DataFrame(std_test, index=X_test.index, columns=X_test.columns)

        return std_train, std_test, stats

    # Function to plot correlation heatmap
    @classmethod
    def corr_plot(cls, corr_matrix):
        aprox_corr = corr_matrix.round(2)
        abs_corr = aprox_corr.abs()

        mask = np.zeros_like(aprox_corr)
        mask[np.triu_indices_from(mask)] = True
        
        with sns.axes_style("white"):
            _, axs = plt.subplots(1, 2, figsize=(20, 7))
            sns.heatmap(aprox_corr, mask=mask, vmin= -1, vmax=1, square=True, annot=True, ax=axs[0])
            sns.heatmap(abs_corr, mask=mask, vmin= 0, vmax=1, square=True, annot=True, cmap="Blues", ax=axs[1])

        plt.show()

    # Function to count plot for category data
    @classmethod
    def count_plot(cls, c_plot, X, y=None, per_row=2):
        n_plots = len(c_plot)
        n_cols = per_row if n_plots > 1 else 1
        n_rows = 1 if n_plots <= per_row else int(round(0.49 + (n_plots / per_row), 0))
        
        _, ax = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        ax = ax.flatten()
        for i, c in enumerate(c_plot): sns.countplot(X[c], hue=y, ax=ax[i])
        plt.show()

    # Oversample data
    @classmethod
    def oversample(cls, X, y, p, method='random'):
        r, random_state = p/(1-p), 0

        if method == 'smote': os = SMOTE(r, random_state=random_state)
        else: os = RandomOverSampler(r, random_state=random_state)
        
        X_resampled, y_resampled = os.fit_sample(X, y)

        return X_resampled, y_resampled

    @classmethod
    # Function to get binary classification confusion matrix and report
    def binary_class_report(cls, y, y_pred):
        conf_matrix = pd.DataFrame(confusion_matrix(y, y_pred), index=[0, 1], columns=[0, 1])
        report = classification_report(y, y_pred, output_dict=True)

        true_rate_metrics = pd.DataFrame([report['0'], report['1'], report['macro avg'], report['weighted avg']], index=[0,1,'avg','wavg'])
        true_rate_metrics

        return conf_matrix, true_rate_metrics

    @classmethod
    # Function to plot curves like learning curves / recall precision curves, etc.
    def curve_plot(cls, data, x_name, color_name, value_name, separate_name=None, filter_query=None):
        plot_data = data.query(filter_query) if filter_query != None else data

        n_plots, n_cols, n_rows, n_rows, per_row = 1, 1, 1, 1, 2
        if separate_name != None:
            plot_values = plot_data[separate_name].unique()
            n_plots = len(plot_values)
            n_cols = per_row if n_plots > 1 else 1
            n_rows = 1 if n_plots <= per_row else int(round(0.49+(n_plots/per_row), 0))

        _, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*5))
        ax = ax.flatten()
        if n_plots > 1:
            for i, c in enumerate(plot_values):
                subplot_data = plot_data.query(f"{separate_name} == '{c}'")
                sns.lineplot(x=x_name, y=value_name, hue=color_name, style=color_name, 
                            markers=True, ci=None, data=subplot_data, ax=ax[i]).set_title(c)
        else:
            sns.lineplot(x=x_name, y=value_name, hue=color_name, style=color_name, 
                        markers=True, ci=None, data=plot_data, ax=ax)
        
        plt.show()

#%%
# Separate Training and Testing Data
data, target_col, feature_cols = Fun.import_data()
X_train, X_test, y_train, y_test = train_test_split(data[feature_cols], data[target_col], test_size=.3, random_state=42)

# Note that the split is performed randomly.
# Important to evaluate if it should be performed time based.

#%%
# Check target proportion
n_train = y_train.size
target_counts = y_train.value_counts()

idx_ok = y_train[y_train == 0].index
idx_fail = y_train[y_train == 1].index

target_proportions = pd.DataFrame({'counts':target_counts})
target_proportions.insert(1, 'proportion', target_proportions['counts'].div(n_train).mul(100).round(2))
target_proportions

# Note that over / under sampling technique is needed so that target proportion doesn't affect model performance

#%%
# Check for missing values (no measure m = 0)
zero_train_pct = (X_train == 0).sum().div(X_train.count()).mul(100).round(2)
zero_test_pct = (X_test == 0).sum().div(X_test.count()).mul(100).round(2)
zero_pct = pd.DataFrame({'train':zero_train_pct, 'test':zero_test_pct})

# Separate features
zero_absent = ['m1', 'm5', 'm6']
zero_dense = [c for c in feature_cols if c not in zero_absent]

zero_pct.transpose()

# Note that there are many zero dense variables 
# 0 values conform 78% of all data in m9 and over 90% in m2, m3, m4, m7, m8.
# Because of the high percentage on option can be discard this columns but it will remove over 50% of available features
# To eliminate columns, perhaps the best way is by regularizing or feature elimination techniques while building a model
# Further analysis will be made on this measures (if they are correlated removing 0 noice)

#%%
# Check if target proportions and zero porportions are somehow related.
def check_zero_target(c, X, idx_fail, idx_ok):
    f, o = X.loc[idx_fail, c], X.loc[idx_ok, c]
    
    fz = round(100 * f[f==0].count() / f.count(), 2)
    fnz = 100 - fz

    oz = round(100 * o[o==0].count() / o.count(), 2)
    onz = 100 - oz

    return {'fail_zero': fz, 'fail_nonzero': fnz, 'ok_zero':oz, 'ok_nonzero':onz}

zero_target = list(map(lambda c: check_zero_target(c, X_train, idx_fail, idx_ok), zero_dense))
zero_target = pd.DataFrame(zero_target, index=zero_dense)
zero_target

# Note that target proportion and zero proportions are not related (data is distributed between zero and non zero values)
# m3 seems to not add much information since 90% of fail / ok data is in zero values.
# Perhaps over sampling is a better option since under sampling increases the risk of loosing zero dense columns info.

#%%
# Get Power / Quantile Transformed and Standardized data
yj_train, yj_test, lmbdas = Fun.power_transform(X_train, X_test, standardize=True)
qt_train, qt_test = Fun.quantile_transform(X_train, X_test)

# Get Just Standardized data
std_train, std_test, stats = Fun.standardize(X_train, X_test)

#%%
# Visualize specific transformed data for a column
def check_measure(c, X_train, yj_train, qt_train, std_train, lmbdas, stats, which=['X', 'yj', 'qt', 'std']):
    loc = X_train.columns.get_loc(c)
    chars = ''

    if 'X' in which: Fun.detail_plot(X_train[c], outliers=False)
    if 'qt' in which:Fun.detail_plot(qt_train[c], outliers=True)
    if 'yj' in which: 
        Fun.detail_plot(yj_train[c], outliers=True)
        chars = f'YJ Lambda: {lmbdas[loc]} '
    if 'std' in which: 
        Fun.detail_plot(std_train[c], outliers=True)
        chars = f"{chars}STD Mean: {stats['mean'][loc]} STD Std: {stats['std'][loc]}"

    print(chars)

c, which = 'm1', ['qt']
check_measure(c, X_train, yj_train, qt_train, std_train, lmbdas, stats, which)

#%%
# Transform Train / Test Data
def X_transform(transforms, X, yj, qt, std):
    
    Xt = X.copy()
    td = {'X':X, 'yj':yj, 'qt':qt, 'std':std}
    for c, t in transforms.items(): Xt[c] = td[t][c]

    return Xt

transforms = {'m1': 'qt', 'm2':'qt', 'm3':'qt', 
              'm4':'qt', 'm5':'qt', 'm6':'qt',
              'm7':'qt', 'm8':'qt', 'm9':'qt'}

Xt_train = X_transform(transforms, X_train, yj_train, qt_train, std_train)
Xt_test = X_transform(transforms, X_test, yj_test, qt_test, std_test)

# %%
# Check detailed view of correlation between measures
def check_correlation(c_check, X_train, Xt_train, remove_zero=False, pairplot=False, heatmap=True):
    
    if remove_zero:
        no_zero_idx = Fun.filter_zeros(c_check, X_train).index
        corr_matrix = Xt_train.loc[no_zero_idx, c_check].corr()
        if pairplot: sns.pairplot(Xt_train.loc[no_zero_idx, c_check])
    
    else: 
        corr_matrix = Xt_train.loc[:, c_check].corr()
        if pairplot: sns.pairplot(Xt_train.loc[:, c_check])

    if heatmap: Fun.corr_plot(corr_matrix)
    plt.show()

    return corr_matrix

c_check = feature_cols
corr_matrix = check_correlation(c_check, X_train, Xt_train, remove_zero=False, pairplot=False, heatmap=True)

# Note that columns with no zeros arent correlated with any other column
# Note that M7 and M8 are fully correlated (Redundant Info) - Remove one
# When analyzing correlation between zero dense columns:
#   - M2 and M4 are highly correlated (0.23 with zeros, -0.7 without zeros)
#   - M3 and M9 are moderately correlated (0.37 with zeros, 0.4 without zeros) 

#%%
# Check zero dense measures
def analize_zero_dense(c, X_train, X_test):
    Xnz_train, Xnz_test = Fun.filter_zeros([c] ,X_train), Fun.filter_zeros([c] ,X_test)
    QTnz_train, _ = Fun.quantile_transform(Xnz_train, Xnz_test)

    which = ['qt']
    check_measure(c, Xnz_train, None, QTnz_train, None, None, None, which)

    c_check = feature_cols
    _ = check_correlation(c_check, Xnz_train, QTnz_train, remove_zero=False, pairplot=False, heatmap=True)

c = 'm9' 
analize_zero_dense(c, X_train, X_test)

# When analizing per zero dense measure: By filtering only rows with values > 0 for that column
# Didn't find any additional relationship between columns as the previous ones (in terms of correlation)
# Although transformation (quantile) resulted in a better distribution for certain cases.

# %%
# Drop redundant columns
if 'm8' in Xt_train.columns: Xt_train.drop('m8', axis=1, inplace=True)
if 'm8' in Xt_test.columns: Xt_test.drop('m8', axis=1, inplace=True)

if 'm8' in feature_cols: feature_cols.remove('m8')
if 'm8' in zero_dense: zero_dense.remove('m8')

# %%
# Define Function to replace outliers based on specified method
def replace_outliers(c_replace, X_train, X_test, method='Z'):
    Xol_train, Xol_test = X_train.copy(), X_test.copy()

    for c in c_replace:
        llim, ulim = Fun.outlier_limits(X_train[c], method)
        Xol_train.loc[Xol_train[c] > ulim, c] = ulim
        Xol_train.loc[Xol_train[c] < llim, c] = llim

        Xol_test.loc[Xol_test[c] > ulim, c] = ulim
        Xol_test.loc[Xol_test[c] < llim, c] = llim

    return Xol_train, Xol_test

c_replace = zero_absent
method = 'Z'
Xol_train, Xol_test = replace_outliers(c_replace, Xt_train, Xt_test, method)

#%%
# Check that outliers were replaced
c, which = 'm1', ['X']
check_measure(c, Xol_train, None, None, None, None, None, which)

# %%
# Test PCA on zero dense measures
def transform_pca(pca_cols, X_train, X_test, screeplot=True):
    n_components = len(pca_cols)
    pca = PCA(n_components = n_components)
    pca.fit(X_train[pca_cols])

    pca_variance = pca.explained_variance_ratio_
    pca_labels = [f'pc{i + 1}' for i in range(0, n_components)]

    def plot_screeplot(pca_variance, pca_labels):
        pca_cumvariance = np.cumsum(pca_variance)
        
        fig = plt.figure(figsize=(20, 7))
        ax1 = fig.add_subplot()
        ax1.bar(pca_labels, pca_variance)
        ax2 = ax1.twinx()
        ax2.plot(pca_labels, pca_cumvariance, color="r")
        ax2.grid(False)
        plt.show()

    if screeplot: plot_screeplot(pca_variance, pca_labels)
    
    Xpca_train, Xpca_test = pca.transform(X_train[pca_cols]), pca.transform(X_test[pca_cols])
    Xpca_train = pd.DataFrame(Xpca_train, index=X_train.index ,columns=pca_labels)
    Xpca_test = pd.DataFrame(Xpca_test, index=X_test.index ,columns=pca_labels)

    not_pca_cols = [c for c in X_train.columns if c not in pca_cols]
    Xpca_train = pd.concat([X_train[not_pca_cols], Xpca_train], axis=1)
    Xpca_test = pd.concat([X_test[not_pca_cols], Xpca_test], axis=1)

    return Xpca_train, Xpca_test

pca_cols = zero_dense
Xpca_train, Xpca_test = transform_pca(pca_cols, Xol_train, Xol_test, screeplot=True)

pca_features = Xpca_train.columns
c_check = pca_features
corr_matrix = check_correlation(c_check, X_train, Xpca_train, remove_zero=False, pairplot=False, heatmap=True)

# Note that leaving no zero dense measures without PCA transforming leaves fully non correlated dataset
# You can also PCA transform all dataset if required.
# We could delete last 2 components to have a 90% of explained variability
# As we have few features, maybe we could leave them (as we guaranteed that any correlation is eliminated)

# %%
# Check Categorical Data
cat_features = ['device', 'date']
Xcat_train = Xpca_train.reset_index()[cat_features].set_index(Xpca_train.index)
Xcat_test = Xpca_test.reset_index()[cat_features].set_index(Xpca_test.index)

#%%
# Define function to add features from device column
def device_features(X, y, plot=True, plot_y=True):
    initial_cols = X.columns

    X['L1'] = X['device'].str[0:2]
    X['L2'] = X['device'].str[2:4]
    #X['L3'] = X['device'].str[4:6]
    #X['L4'] = X['device'].str[6:]

    if plot:
        y = None if not plot_y else y
        c_plot = [c for c in X.columns if c not in initial_cols]
        Fun.count_plot(c_plot, X, y, per_row=2)

    return X

Xcat_train = device_features(Xcat_train, y_train, plot=True, plot_y=False)
Xcat_test = device_features(Xcat_test, y_test, plot=False)

# Note that only 2 sublevels were defined by device id
# May have a business meaning like location, or some sort of device category.
# This two levels groups data homogeneously and have few values.
# Note that for l2, one value has almost no data. This wont add much info to model.

# %%
# Define function to add features from date column
def date_features(X, y, plot=True, plot_y=True):
    initial_cols = X.columns

    #X['year'] = X['date'].dt.year
    X['Q'] = X['date'].dt.quarter
    X['M'] = X['date'].dt.month
    X['D'] = X['date'].dt.day
    X['WD'] = X['date'].dt.weekday
    X['WE'] = (X['WD'] > 4).astype(int)

    if plot:
        y = None if not plot_y else y
        c_plot = [c for c in X.columns if c not in initial_cols]
        Fun.count_plot(c_plot, X, y, per_row=2)

    return X

Xcat_train = date_features(Xcat_train, y_train, plot=True, plot_y=False)
Xcat_test = date_features(Xcat_test, y_test, plot=False)

# Note that all data is form one single year
# Year feature is not necessary, and perhaps the same can be said for quarter and month
# Quarter and month will be left for now (check to remove with regularization or feature elimination)

# %%
# Process Categorical Variables
def process_categorical(c_encode, c_leave, X_train, X_test):

    enc = OneHotEncoder(drop='first', sparse=False)
    Xenc_train = enc.fit_transform(X_train[c_encode])
    Xenc_test = enc.transform(X_test[c_encode])

    Xenc_train = pd.DataFrame(Xenc_train, index=X_train.index, columns=enc.get_feature_names(input_features=c_encode))
    Xenc_test = pd.DataFrame(Xenc_test, index=X_test.index, columns=enc.get_feature_names(input_features=c_encode))

    Xf_train = pd.concat([X_train[c_leave], Xenc_train.astype(int)], axis=1)
    Xf_test = pd.concat([X_test[c_leave], Xenc_test.astype(int)], axis=1)

    return Xf_train, Xf_test

c_encode, c_leave = ['L1', 'L2', 'Q', 'M', 'D', 'WD'], ['WE']
Xcat_train, Xcat_test = process_categorical(c_encode, c_leave, Xcat_train, Xcat_test)
Xcat_train.head()

# %%
# Merge Input Data / Remove intermediate data
def merge_input(Xcat_train, Xcat_test, Xnum_train, Xnum_test): 
    
    Xin_train = pd.concat([Xcat_train, Xnum_train], axis=1)
    Xin_test = pd.concat([Xcat_test, Xnum_test], axis=1)

    return Xin_train, Xin_test

X_train, X_test = merge_input(Xcat_train, Xcat_test, Xpca_train, Xpca_test)
feature_cols = X_train.columns

del yj_test, yj_train, qt_train, qt_test, std_train, std_test
del Xt_train, Xt_test, Xol_train, Xol_test, Xpca_train, Xpca_test, Xcat_train, Xcat_test

# %%
# Test for oversample proportion and method
def test_oversample(X, y, model, proportions, methods=['random', 'smote']):
    results = []
    for m in methods:
        for p in proportions:
            Xos_train, yos_train = Fun.oversample(X, y, p, m)
            model.fit(Xos_train, yos_train)
            y_pred = model.predict(X)

            _, class_metrics = Fun.binary_class_report(y, y_pred)
            class_metrics.insert(0, 'target', class_metrics.index.astype(str))
            class_metrics.insert(0, 'proportion', p)
            class_metrics.insert(0, 'method', m)
            
            results.append(class_metrics)

    results = pd.concat(results).reset_index(drop=True)
    
    return results

proportions, methods = np.arange(0.05, 0.55, 0.05), ['random', 'smote']
target = 1
model = LogisticRegression()
oversample_results = test_oversample(X_train, y_train, model, proportions, methods)

# Transform (melt oversample results)
id_vars = ['method', 'proportion', 'target']
value_vars = ['precision', 'recall', 'f1-score', 'support']
melt_oversample_results = pd.melt(oversample_results, id_vars=id_vars, value_vars=value_vars, var_name='metric')

#%%
# Plot learning curve for proportion / precision on both methods
filter_query= "(target == '0') | (target == '1')"    
Fun.curve_plot(oversample_results, 'proportion', 'method', 'precision', 'target', filter_query)

# Note that precision will be critical to assess.
# Both oversampling methods dont show much difference when evaluating precision
# Precision decreases as sample proportion increases

#%%
# Plot precision / recall curve for proportion  on both methods
filter_query= "target == '1' & ((metric == 'precision') | (metric == 'recall'))"  
Fun.curve_plot(melt_oversample_results, 'proportion', 'metric', 'value', 'method', filter_query)

# Recall increses with sample proportion 
# Recall is significantly better with random over sampling

# %%
