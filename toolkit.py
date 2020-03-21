# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class Viz():
    
    per_row = 2

    @classmethod
    # Change default values
    def change_default(cls, per_row=2):
        cls.per_row = per_row

    @classmethod
    # Function to build base template to place charts
    def get_figure(cls, n_plots):
        n_cols = cls.per_row if n_plots > 1 else 1
        n_rows = 1 if n_plots <= cls.per_row else int(round(0.49 + (n_plots / cls.per_row), 0))
        
        _, ax = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        ax = ax.flatten() if n_plots > 1 else ax

        return ax

    # Function to get readable string for number
    @classmethod
    def fnum(cls, n):
        n0, is_int = n, isinstance(n, int)
        magnitude = 0
        while abs(n) >= 1000:
            magnitude += 1
            n /= 1000.0
        
        if is_int & (n0 < 1000): return f'{n0}'
        else: return '%.2f%s' % (n, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    # Function to plot aproximate distribution of a numeric data
    @classmethod
    def kernel_plot(cls, X, y, c_plot, remove_val=[np.nan], rug_sample=1000): 
        y_vals = y.unique() if y is not None else []
        
        n_plots = len(c_plot)
        ax = cls.get_figure(n_plots)
        
        for i, c in enumerate(c_plot): 
            axi = ax[i] if n_plots > 1 else ax
            n = []
            if len(y_vals) > 1: 
                for j, v in enumerate(y_vals):
                    x = X.loc[y == v, c]
                    x = x[~x.isin(remove_val)]
                    name = f'{v} ({cls.fnum(x.mean())}, {cls.fnum(x.std())})' 
                    x.name = name
                    n.append(x.size)
                    plot = sns.kdeplot(x, shade=True, ax=axi)
                    
                    if rug_sample > 0:
                        color = plot.get_lines()[j].get_c()
                        s = x.sample(n=rug_sample) if x.size > rug_sample else x
                        s.name = name
                        sns.rugplot(s, ax=axi, color=color)
            else: 
                x = X.loc[~X[c].isin(remove_val), c]
                n.append(x.size)
                plot = sns.kdeplot(x, shade=True, ax=axi)

                if rug_sample > 0:
                    s = x.sample(n=rug_sample) if x.size > rug_sample else x
                    sns.rugplot(s, ax=axi)

            plot.set_title(f'{c} - {str(remove_val)}, n={cls.fnum(sum(n))} {str(list(map(cls.fnum, n)))}')
        
        plt.show()

    # Function to plot correlation heatmap
    @classmethod
    def corr_plot(cls, corr_matrix):
        aprox_corr = corr_matrix.round(2).mul(100).astype(int)
        abs_corr = aprox_corr.abs()

        mask = np.zeros_like(aprox_corr)
        mask[np.triu_indices_from(mask)] = True
        
        with sns.axes_style("white"):
            ax = cls.get_figure(2)
            sns.heatmap(aprox_corr, mask=mask, vmin= -100, vmax=100, square=False, annot=True, fmt='d', ax=ax[0])
            sns.heatmap(abs_corr, mask=mask, vmin= 0, vmax=100, square=False, annot=True, fmt='d', cmap="Blues", ax=ax[1])

        plt.show()

    # Function to count plot for category data
    @classmethod
    def count_plot(cls, X, y, c_plot, proportion=True):
        y = pd.Series([1] * X.shape[0], name='total') if y is None else y
        y_vals = y.unique()
        
        n_plots = len(c_plot)
        ax = cls.get_figure(n_plots)

        if len(y_vals) > 0:
            Xcnt = X[c_plot]
            Xcnt.insert(0, 'id', Xcnt.index)
            
            cnt_data = pd.concat([Xcnt, y], axis=1)
            cnt_data = cnt_data.groupby(c_plot + [y.name]).agg('count').unstack().fillna(0)
            cnt_data.columns = y_vals
            
            cnt_name = 'cnt'
            if proportion: 
                ycnt, cnt_name = {}, 'pct'
                for v in y_vals: ycnt[v] = (y == v).sum()
                cnt_data = cnt_data.div(ycnt).replace([-np.inf, np.inf], 0)
            cnt_name = 'pct' if proportion else 'cnt'
            
            cnt_data = cnt_data.reset_index().melt(id_vars=c_plot ,value_vars=y_vals, var_name=y.name, value_name=cnt_name)

        for i, c in enumerate(c_plot): 
            axi = ax[i] if n_plots > 1 else ax
            sns.barplot(x=c, y=cnt_name, hue=y.name, data=cnt_data, ci=None,  estimator=np.sum, ax=axi)
        
        plt.show()

    @classmethod
    def screeplot(cls, pca_var, pca_labels):
        pca_cumvar = np.cumsum(pca_var)
        
        ax1 = cls.get_figure(1)
        ax1.bar(pca_labels, pca_var)
        ax2 = ax1.twinx()
        ax2.plot(pca_labels, pca_cumvar, color="r")
        ax2.grid(False)
        plt.show()

class Tools():
    #Define function to filter zero values for specified cols
    @classmethod
    def filter_val(cls, X, c_check, vals=[None]):
        filters = []
        for c in c_check:
            for v in vals:
                if v is not None:
                    str_v = f"'{v}'" if isinstance(v, str) else f"{v}"
                    filters.append(f"({c} != {str_v})")
                    
        if len(filters) > 0: return X.query(' | '.join(filters))
        else: return X
    
    #Define function to calculate percentage of specified values
    @classmethod
    def get_percentage(cls, X, y, values=[0]):
        if y is None: y = pd.Series(['total'] * X.shape[0], name='total', index=X.index)
        v_pct = pd.concat([X.isin(values), y], axis=1)
        v_pct = v_pct.groupby(y.name)
        v_pct = v_pct.agg(lambda x: sum(x) / len(x)).mul(100).round(2)
        return v_pct
    
    # Find target proportions
    @classmethod
    def get_target_prop(cls, X, y):
        target_proportion = list(zip(y.value_counts(),
                                    y.value_counts(normalize=True).mul(100).round(2)))
        target_proportion = pd.DataFrame(target_proportion, columns=['count', 'proportion'])
        return target_proportion

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

    # Undersample data
    @classmethod
    def undersample(cls, X, y, p, random_state=0):
        r = p/(1-p)
        us = RandomUnderSampler(r, random_state=random_state)
        
        X_resampled, y_resampled = us.fit_sample(X, y)

        return X_resampled, y_resampled

    # Get number of instances per strata for a stritified undersample.
    @classmethod
    def strat_undersample_counts(cls, strt, strata_p, random_state=0):
        strt_counts = strt.value_counts()
        strt_min = strt_counts.idxmin()
        n_min, p_min = strt_counts[strt_min], strata_p[strt_min]
        
        n_vals = {}
        for v in strt_counts.index:
            n, p = strt_counts[v], strata_p[v]
            
            n = int(min(p * n_min / p_min, n))
            n_vals[v] = n

        return n_vals
    
    # Undersample data
    @classmethod
    def stratified_undersample(cls, X, y, strata_p, random_state=0):
        n_vals = cls.strat_undersample_counts(y, strata_p, random_state)
        
        us = RandomUnderSampler(n_vals, random_state=random_state)
        Xrs, yrs = us.fit_resample(X.reset_index(), y)
        Xrs = Xrs.set_index('index')
        
        return Xrs, yrs

    # Oversample data
    @classmethod
    def oversample(cls, X, y, p, method='random', random_state=0):
        r = p/(1-p)

        if method == 'smote': os = SMOTE(r, random_state=random_state)
        else: os = RandomOverSampler(r, random_state=random_state)
        
        X_resampled, y_resampled = os.fit_sample(X, y)

        return X_resampled, y_resampled

    # Function to power transform data
    @classmethod
    def power_transform(cls, X, standardize=True):
        yj = PowerTransformer(method='yeo-johnson', standardize=standardize)
        Xyj = yj.fit_transform(X)
        #lmbdas = yj.lambdas_
        
        Xyj = pd.DataFrame(Xyj, index=X.index, columns=X.columns)

        return Xyj

    # Function to quantile transform data
    @classmethod
    def quantile_transform(cls, X):
        rng = np.random.RandomState(304)
        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=rng)
        Xqt = qt.fit_transform(X)

        Xqt = pd.DataFrame(Xqt, index=X.index, columns=X.columns)

        return Xqt

    # Function to standardize data
    @classmethod
    def standardize(cls, X, which='minmax'):
        if which == 'normal':
            std = StandardScaler()
            Xstd = std.fit_transform(X)
        else:
            std = MinMaxScaler()
            Xstd = std.fit_transform(X)

        Xstd = pd.DataFrame(Xstd, index=X.index, columns=X.columns)

        return Xstd

    # Check detailed view of correlation between measures
    @classmethod
    def get_corr(cls, X, c_check, remove_val=[None]):  
        Xcorr = cls.filter_val(X, c_check, remove_val)
        corr_matrix = Xcorr.loc[:, c_check].corr()

        return corr_matrix

    # PCA Transform
    @classmethod
    def pca(cls, X, c_pca):
        n_comp = len(c_pca)
        pca = PCA(n_components=n_comp)
        Xpca = pca.fit_transform(X[c_pca])

        pca_var = pca.explained_variance_ratio_
        pca_labels = [f'pc{i + 1}' for i in range(0, n_comp)]

        Xpca = pd.DataFrame(Xpca, index=X.index, columns=pca_labels)

        return Xpca, pca_var

    # One Hot Encode
    @classmethod
    def oh_encode(cls, X, c_enc):
        enc = OneHotEncoder(drop='first', sparse=False)
        Xenc = enc.fit_transform(X)
        enc_cols = enc.get_feature_names(input_features=c_enc)
        
        Xenc = pd.DataFrame(Xenc, index=X.index, columns=enc_cols)

        return Xenc, enc_cols


class PF():

    strata_p = {0: 0.05, 1: 0.8, 2: 0.15} #{0: 0.1, 1: 0.6, 2: 0.3} {0: 0.1, 1: 0.7, 2: 0.2} {0: 0.1, 1: 0.75, 2: 0.15}
    random_state = 42
    
    # Function to import data
    @classmethod
    def import_data(cls):
        # Base columns
        cls.cat_cols = ['device', 'date']
        cls.num_cols = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9']
        cls.target_col = 'failure'

        # Import data, remove duplicates and set index
        data = pd.read_csv('data/device_failure.csv', parse_dates=['date'], encoding='ISO-8859-1')
        data.drop_duplicates(inplace=True)

        # Rename / Reorder columns, asssign target and features
        data.columns = [c.replace('attribute', 'm') if c != 'failure' else c for c in data.columns]
        data = data[cls.cat_cols + cls.num_cols + [cls.target_col]]

        return data, cls.target_col, cls.cat_cols, cls.num_cols

    # Function to add full zero mark
    @classmethod
    def full_zero_mark(cls, X):
        cls.zdense_cols = [c for c in ['m2', 'm3', 'm4', 'm7', 'm8', 'm9'] if c in X.columns]
        cls.f0_col = 'f0'
        full_zero = pd.DataFrame(X[cls.zdense_cols].sum(axis=1) == 0, columns=[cls.f0_col])
        X.insert(X.shape[1], cls.f0_col, full_zero.astype(int))

        return X, cls.zdense_cols, cls.f0_col
    
    @classmethod
    def add_strata(cls, X, y):
        strt = pd.Series([0] * X.shape[0], index=X.index)
        strt[(y == 0) & (X[cls.f0_col] == 0)] = 1
        strt[(y == 1)] = 2

        X.insert(X.shape[1], 'strt', strt)
        cls.strt_col = 'strt'

        return X, cls.strt_col
    
    @classmethod
    def remove_duplicates(cls, X):
        X = X.drop('m8', axis=1)
        cls.num_cols.remove('m8')

        return X, cls.num_cols
    
    # Function to add device features
    @classmethod
    def device_features(cls, X):
        X['L1'] = X['device'].str[0:2]
        X['L2'] = X['device'].str[2:4]
        X['L12'] = X['device'].str[0:4]
        #X['L3'] = X['device'].str[4:6]
        #X['L4'] = X['device'].str[6:]

        cls.dev_cols = [c for c in ['L1', 'L2', 'L12', 'L3', 'L4'] if c in X.columns]

        return X, cls.dev_cols

    # Function to add features from date
    @classmethod
    def date_features(cls, X):
        #X['Y'] = X['date'].dt.year
        X['Q'] = X['date'].dt.quarter
        X['M'] = X['date'].dt.month
        X['D'] = X['date'].dt.day
        X['MW'] = X['D'].div(8).apply(np.floor).add(1).astype(int)
        X['WD'] = X['date'].dt.weekday
        X['WE'] = (X['WD'] > 4).astype(int)

        cls.date_cols = [c for c in ['Y', 'Q', 'M', 'D', 'MW', 'WD', 'WE'] if c in X.columns]

        return X, cls.date_cols

    # Function to add change features
    @classmethod
    def change_metrics(cls, X):
        # Sort data 
        Xsort = X.sort_values(cls.cat_cols)

        # Create elapsed column
        elapsed = Xsort[cls.cat_cols].groupby('device').diff().bfill()['date']

        # Create absolute change dataframe
        Xchng = Xsort[['device'] + cls.num_cols].groupby('device').pct_change().fillna(0).abs()
        Xchng = Xchng.replace([np.inf], 0)
        Xchng.columns = [c.replace('m', 'ch') for c in Xchng.columns]
        Xchng.insert(0, 'elapsed', elapsed.dt.days)

        drop_cols = ['elapsed', 'ch3', 'ch9']
        if len(drop_cols) > 0: Xchng.drop(drop_cols, axis=1, inplace=True)

        cls.chng_cols = Xchng.columns.tolist()
        X = pd.concat([X, Xchng.sort_index()], axis=1)

        return X, cls.chng_cols

    # Function to add rank features
    @classmethod
    def rank_metrics(cls, X):
        Xrnk = X[cls.num_cols].rank(method='min').sub(1).div(X.shape[0])
        Xrnk.columns = [c.replace('m', 'sr') for c in Xrnk.columns]
        Xrnk['srav'] = Xrnk.mean(axis=1)
        Xrnk['srmx'] = Xrnk.max(axis=1)

        drop_cols = ['sr1', 'sr5', 'sr6', 'srav']
        if len(drop_cols) > 0: Xrnk.drop(drop_cols, axis=1, inplace=True)

        cls.rnk_cols = Xrnk.columns.tolist()
        X = pd.concat([X, Xrnk], axis=1)

        return X, cls.rnk_cols

    # Function to add all features
    @classmethod
    def add_features(cls, X, y):
        X, _ = cls.remove_duplicates(X)
        X, _ = cls.device_features(X)
        X, _ = cls.date_features(X)
        X, _ = cls.change_metrics(X)
        X, _ = cls.rank_metrics(X)
        X, _, _ = cls.full_zero_mark(X)
        X, _ = cls.add_strata(X, y)

        cls.ncat_cols = cls.cat_cols + cls.dev_cols + cls.date_cols
        cls.nnum_cols = cls.num_cols + cls.chng_cols + cls.rnk_cols
        cls.cntr_cols = [cls.f0_col, cls.strt_col]

        return X

    # Function to transform all numeric features
    @classmethod
    def transform(cls, X):
        cls.c_trans = {}
        cls.c_trans['m'] = {'m1': 'qt', 'm2':'nrm', 'm3':'nrm', 'm4':'nrm', 
                            'm5':'qt', 'm6':'qt', 'm7':'nrm', 'm9':'nrm'}
        cls.c_trans['ch'] = {'ch1': 'mnmx', 'ch2':'mnmx', 'ch4':'mnmx', 
                             'ch5':'mnmx', 'ch6':'mnmx', 'ch7':'mnmx'}
        cls.c_trans['sr'] = {'sr2': 'none', 'sr3':'none', 'sr4':'none', 
                             'sr7':'none', 'sr9':'none', 'srmx':'none'}

        for _, trans in cls.c_trans.items():
            for c, t in trans.items():
                if t == 'yj': xt = Tools.power_transform(X[[c]], standardize=True)
                elif t == 'qt': xt = Tools.quantile_transform(X[[c]])
                elif t == 'nrm': xt = Tools.standardize(X[[c]], which='normal')
                elif t == 'mnmx': xt = Tools.standardize(X[[c]], which='minmax')
                else: xt = X[[c]]

                X[c] = xt[c]

        return X
    
    # Function to pca transform all numeric features
    @classmethod
    def pca(cls, X):
        c_pca = cls.nnum_cols
        Xpca, cls.pca_cols = Tools.pca(X, c_pca)
        return Xpca
    
    # Function to one hot encode all categorical features
    @classmethod
    def encode(cls, X):
        cat_cols = [c for c in PF.ncat_cols if c not in PF.cat_cols]
        Xenc, cls.enc_cols = Tools.oh_encode(X[cat_cols], cat_cols)

        return Xenc
    
    # Function to prepare all features
    @classmethod
    def prepare_features(cls, X, num_transform=True):
        if num_transform:
            Xnum = cls.transform(X)
            Xnum = cls.pca(Xnum)
        else: Xnum = X[cls.nnum_cols]

        Xcat = cls.encode(X)
        Xprep = pd.concat([Xcat, Xnum, X[cls.cntr_cols]], axis=1)

        return Xprep
    
    @classmethod
    def get_nvals(cls, strt, strata_p, cv=False):
        n_vals = Tools.strat_undersample_counts(strt, strata_p, cls.random_state)
        if cv: n_vals = {k:int(v * 0.8) for k, v in n_vals.items()}

        return n_vals
    
    # Function to stratified undersample
    @classmethod
    def strat_undersample(cls, X, y, strata_p=strata_p, random_state=random_state):
        Xrs, _ = Tools.stratified_undersample(X, X[cls.strt_col], strata_p, random_state)
        yrs = y.loc[Xrs.index]

        return Xrs, yrs
    
    # Best Stratified Bagg + Boost Model
    @classmethod
    def best_strat_bagg_boost(cls, X_train, strt_train):
        
        n_vals = Tools.strat_undersample_counts(strt_train, cls.strata_p, cls.random_state)

        tree = DecisionTreeClassifier(class_weight='balanced', max_depth=30, max_features=0.5)

        boost = AdaBoostClassifier(base_estimator=tree, n_estimators=10)

        bbagg = BalancedBaggingClassifier(base_estimator=boost, n_estimators=300, sampling_strategy=n_vals,
                                          bootstrap=False, random_state=cls.random_state, n_jobs=10)
        
        bbagg.fit(X_train, strt_train)

        return bbagg

    # Define function to get Precision / Recall curve, Best Threshold and F1 Confusion Matrix
    @classmethod
    def get_metrics(cls, y_true, y_proba, plot=True):
        prc, rcl, thrsh = precision_recall_curve(y_true, y_proba) 
        f1 = 2 * ((prc * rcl) / (prc + rcl))
        pr_data = pd.DataFrame([thrsh, prc, rcl, f1], index=['threshold', 'precision', 'recall', 'f1']).transpose()

        id_best = pr_data['f1'].idxmax()
        thrsh_best = pr_data.loc[id_best, 'threshold']

        y_pred = list(map(lambda v: 1 if v >= thrsh_best else 0, y_proba))
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrix = pd.DataFrame(conf_matrix, index=y_true.unique(), columns=y_true.unique())

        if plot:
            m_pr_data = pr_data.melt(id_vars=['threshold'], var_name='metric')

            ax = Viz.get_figure(4)
            sns.lineplot(data=pr_data, x='precision', y='recall', ax=ax[0])
            sns.lineplot(data=m_pr_data.query("metric != 'f1'"), x='threshold', y='value', hue='metric', ax=ax[1])
            sns.lineplot(data=pr_data, x='threshold', y='f1', ax=ax[2])
            sns.heatmap(conf_matrix, vmin= 0, vmax=100, square=False, annot=True, fmt='d', ax=ax[3])
            plt.show()

        return pr_data, thrsh_best, confusion_matrix