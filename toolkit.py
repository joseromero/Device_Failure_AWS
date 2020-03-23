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

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class Viz():
    """ Class for easy access to frequently used vizualisation tools

        Class Atributes
        ---------------

        per_row: Number of charts per row when ensembling a multichart figure
    """
    per_row = 2

    @classmethod
    def change_default(cls, per_row=2):
        """ Changes default class atributes. In particular the number of charts per row.

            Parameters
            -----------
            per_row: New value for number of charts per row in a multichart figure
        """
        cls.per_row = per_row

    @classmethod
    def get_figure(cls, n_plots):
        """ Returns pyplot axis array / object with grid layout to place charts
            The grid layout is based on number of plots and default number of charts per row.

            Parameters
            ----------
            n_plots: Number of carts that need to be placed in final figure

            Returns
            -------
            ax: Pyplot axis object (when n_plots is 1) or array (when greater) to place charts.
        """
        n_cols = cls.per_row if n_plots > 1 else 1
        n_rows = 1 if n_plots <= cls.per_row else int(round(0.49 + (n_plots / cls.per_row), 0))
        
        _, ax = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        ax = ax.flatten() if n_plots > 1 else ax

        return ax

    @classmethod
    def fnum(cls, n):
        """ Turns number into readable abreviated string. Useful on annotating charts.

            Parameters
            ----------
            n: Number that needs to be transformed

            Returns
            -------
            n_str: String in readable abreviated format for the specified number
        """
        n0, is_int = n, isinstance(n, int)
        magnitude = 0
        while abs(n) >= 1000:
            magnitude += 1
            n /= 1000.0
        
        if is_int & (n0 < 1000): return f'{n0}'
        else: return '%.2f%s' % (n, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    @classmethod
    def kernel_plot(cls, X, y, c_plot, remove_val=[np.nan], rug_sample=1000):
        """ Plots the KDE (Kernel Distribution Estimate) for specified metric columns. One chart per metric column.

            Separates on distinct y values if required. 
            Removes a set of values before plotting.
            Adds rug plot based on subsample of data.

            Parameters
            ----------
            X: DataFrame that includes metric columns.

            y: None if any distinction is needed. Series alligned with X rows that represents separation criteria.

            c_plot: List of columns that need to be plotted. One chart will be displayed per column in this list.

            remove_val: List of values to remove from column data before plotting. Usefull to reduce noice.

            rug_sample: Number of data subsampled to generate rugplot.
        """
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
                        if len(plot.get_lines()) > j:
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

    @classmethod
    def corr_plot(cls, corr_matrix):
        """ Plots correlation heatmap. Plot will remove upper diagonal values for better readability.

            Two heatmaps will be displayed. One with corr values and the othe with its absolute values.
            Values will be displayed as 100 * corr_value

            Parameters
            ----------
            corr_matrix: DataFrame representing the confusion matrix to plot.
        """
        aprox_corr = corr_matrix.round(2).mul(100).astype(int)
        abs_corr = aprox_corr.abs()

        mask = np.zeros_like(aprox_corr)
        mask[np.triu_indices_from(mask)] = True
        
        with sns.axes_style("white"):
            ax = cls.get_figure(2)
            sns.heatmap(aprox_corr, mask=mask, vmin= -100, vmax=100, square=False, annot=True, fmt='d', ax=ax[0])
            sns.heatmap(abs_corr, mask=mask, vmin= 0, vmax=100, square=False, annot=True, fmt='d', cmap="Blues", ax=ax[1])

        plt.show()

    @classmethod
    def count_plot(cls, X, y, c_plot, proportion=True):
        """ Plots count plot for categorical data.

            Separates bars on y distinct values.
            Plots as absolute count or relative (to y value) proportion.

            Parameters
            ----------
            X: DataFrame that includes categorical columns.

            y: None if any distinction is needed. Series alligned with X rows that represents separation criteria.

            c_plot: List of columns that need to be plotted. One chart will be displayed per column in this list.

            proportion: If plot shows relative proportion or not.
        """
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
                ycnt = pd.Series(ycnt)
                cnt_data = cnt_data.div(ycnt).replace([-np.inf, np.inf], 0)
            cnt_name = 'pct' if proportion else 'cnt'
            
            cnt_data = cnt_data.reset_index().melt(id_vars=c_plot ,value_vars=y_vals, var_name=y.name, value_name=cnt_name)

        for i, c in enumerate(c_plot): 
            axi = ax[i] if n_plots > 1 else ax
            sns.barplot(x=c, y=cnt_name, hue=y.name, data=cnt_data, ci=None,  estimator=np.sum, ax=axi)
        
        plt.show()

    @classmethod
    def screeplot(cls, pca_var, pca_labels):
        """ Plots screeplot for PCA Analysis. Will display explained variance and cummuative explained variance.

            Parameters
            ----------
            pca_var: Iterable with variance explained per component found in PCA

            pca_labels: Iterable with label per component found in PCA
        """
        pca_cumvar = np.cumsum(pca_var)
        
        ax1 = cls.get_figure(1)
        ax1.bar(pca_labels, pca_var)
        ax2 = ax1.twinx()
        ax2.plot(pca_labels, pca_cumvar, color="r")
        ax2.grid(False)
        plt.show()

class Tools():
    """ Class for easy access to preparation and analysis tools
    """
    
    @classmethod
    def filter_val(cls, X, c_check, vals=[None]):
        """ Filters all of specified values on specified columns

            Parameters
            ----------
            X: DataFrame where data fill be filtered

            c_check: List of columns where data will be filtered

            vals: List of values to filter

            Returns
            -------
            X_filter: Same X DataFrame with filtered values.
        """
        filters = []
        for c in c_check:
            for v in vals:
                if v is not None:
                    str_v = f"'{v}'" if isinstance(v, str) else f"{v}"
                    filters.append(f"({c} != {str_v})")
                    
        if len(filters) > 0: return X.query(' | '.join(filters))
        else: return X
    
    @classmethod
    def get_percentage(cls, X, y, values=[0]):
        """ Calculates percentage of specified values in all columns of X, separated by y values.

            Parameters
            ----------
            X: DataFrame where percentage calculation will be performed

            y: None if no tistinction is wanted. Series of values aligned with X rows to separate percentage calculation.

            values: List of values to look for when calculating percentage.

            Returns
            -------
            v_pct: DataFrame with same columns as X, with the percentage of values per column.
        """
        if y is None: y = pd.Series(['total'] * X.shape[0], name='total', index=X.index)
        v_pct = pd.concat([X.isin(values), y], axis=1)
        v_pct = v_pct.groupby(y.name)
        v_pct = v_pct.agg(lambda x: sum(x) / len(x)).mul(100).round(2)
        return v_pct
    
    @classmethod
    def get_target_prop(cls, y):
        """ Calculates target proportion.

            Parameters
            ----------
            y: Series with target values.

            Returns
            -------
            targ_prop: DataFrame with target counts and target proportions
        """
        target_proportion = list(zip(y.value_counts(),
                                    y.value_counts(normalize=True).mul(100).round(2)))
        target_proportion = pd.DataFrame(target_proportion, columns=['count', 'proportion'])
        return target_proportion

    @classmethod
    def strat_undersample_counts(cls, strt, strata_p, cv=False):
        """ Calculates number of instances per strata for a stratified undersample.

            Considers if undersample will be used on cross validation.

            Parameters
            ----------
            strt: Series with strata value for each instance

            strata_p: Dict with percentage on final undersample per strata (key)

            cv: If undersample will be used in cross validation

            Returns
            -------
            n_vals: Dict with number of intances per strata (key)
        """
        strt_counts = strt.value_counts()
        strt_min = strt_counts.idxmin()
        n_min, p_min = strt_counts[strt_min], strata_p[strt_min]
        
        n_vals = {}
        for v in strt_counts.index:
            n, p = strt_counts[v], strata_p[v]
            
            n = int(min(p * n_min / p_min, n))
            n_vals[v] = n

        if cv: n_vals = {k:int(v * 0.8) for k, v in n_vals.items()}

        return n_vals
    
    @classmethod
    def stratified_undersample(cls, X, strt, strata_p, random_state=0):
        """ Gets a stratified under sample for X based on strata proportions.

            Parameters
            ----------
            X: DataFrame that will be undersampled

            strt: Series with strata values (aligned with X).

            strata_p: Dict with percentage on final undersample per strata (key)

            random_state: Random seed

            Returns
            -------
            Xrs: Stratified undersample of X.

            strt_rs: Stratified undersample of strt. Alligned with Xrs
        """
        n_vals = cls.strat_undersample_counts(strt, strata_p)
        
        us = RandomUnderSampler(n_vals, random_state=random_state)
        Xrs, strt_rs = us.fit_resample(X.reset_index(), strt)
        Xrs = Xrs.set_index('index')
        
        return Xrs, strt_rs

    @classmethod
    def power_transform(cls, X, standardize=True):
        """ Performs power transform on all columns of X

            Uses yeo-johnson technique. Standardizes data if required.

            Parameters
            ----------
            X: DataFrame that will be power transformed.

            standardize: If standardization is required or not.

            Returns
            -------
            Xyj: Transformed DataFrame
        """
        yj = PowerTransformer(method='yeo-johnson', standardize=standardize)
        Xyj = yj.fit_transform(X)
        #lmbdas = yj.lambdas_
        
        Xyj = pd.DataFrame(Xyj, index=X.index, columns=X.columns)

        return Xyj

    @classmethod
    def quantile_transform(cls, X):
        """ Performs quantile transform on all columns of X

            Parameters
            ----------
            X: DataFrame that will be power transformed.

            Returns
            -------
            Xqt: Transformed DataFrame
        """
        rng = np.random.RandomState(304)
        qt = QuantileTransformer(n_quantiles=500, output_distribution='normal', random_state=rng)
        Xqt = qt.fit_transform(X)

        Xqt = pd.DataFrame(Xqt, index=X.index, columns=X.columns)

        return Xqt

    @classmethod
    def standardize(cls, X, which='minmax'):
        """ Standardizes data based on min-max or normal strategy

            Parameters
            ----------
            X: DataFrame that will be transformed

            which: Either 'minmax' or 'normal' to specify method.

            Returns
            -------
            Xstd: Transformed DataFrame
        """
        if which == 'normal':
            std = StandardScaler()
            Xstd = std.fit_transform(X)
        else:
            std = MinMaxScaler()
            Xstd = std.fit_transform(X)

        Xstd = pd.DataFrame(Xstd, index=X.index, columns=X.columns)

        return Xstd

    @classmethod
    def get_corr(cls, X, c_check, remove_val=[None]):
        """ Calculates correlation matrix on specified columns of X.
            
            Parameters
            ----------
            X: DataFrame where correlation will be calculated.

            c_check: List of columns of X to use on correlation matrix.

            remove_val: List of values that need to be removed before calculating correlation.

            Returns
            -------
            corr_matrix: Numpy matrix - Correlation matrix
        """
        Xcorr = cls.filter_val(X, c_check, remove_val)
        corr_matrix = Xcorr.loc[:, c_check].corr()

        return corr_matrix

    @classmethod
    def pca(cls, X, c_pca):
        """ Executes PCA on X for specified columns

            The number of components will be the same number of columns in c_pca.
            Will return explained variance per component as well.

            Parameters
            ----------
            X: DataFrame that will be transformed

            c_pca: List of columns to use in PCA

            Returns
            -------
            Xpca: DataFrame with new PCA components

            pca_var: Array with explained variance per component.
        """
        n_comp = len(c_pca)
        pca = PCA(n_components=n_comp)
        Xpca = pca.fit_transform(X[c_pca])

        pca_var = pca.explained_variance_ratio_
        pca_labels = [f'pc{i + 1}' for i in range(0, n_comp)]

        Xpca = pd.DataFrame(Xpca, index=X.index, columns=pca_labels)

        return Xpca, pca_var

    @classmethod
    def oh_encode(cls, X, c_enc):
        """ One Hot Encodes categorical data

            Parameters
            ----------
            X: DataFrame where transformation will performed

            c_enc: List of columns that will be encoded.

            Returns
            -------
            X_enc: DataFrame with encoded categorical data.

            enc_cols: List of encoded column names
        """
        enc = OneHotEncoder(drop='first', sparse=False)
        Xenc = enc.fit_transform(X)
        enc_cols = enc.get_feature_names(input_features=c_enc)
        
        Xenc = pd.DataFrame(Xenc, index=X.index, columns=enc_cols)

        return Xenc, enc_cols

class PF():
    """ Class for Problem Specific Functions (PF). 

        Useful functions of analysis proccess are accumulated in this class for further steps use.

        Class Attributes
        ----------------
        strata_p: Initial proportion values for stratified subsampling. Used throughout first 5 steps.

        random_state: Random seed for replicability.

        cat_cols: List of categorical column names. Available after import_data

        num_cols: List of numerical column names. Available after import_data. Modified after remove_duplicates.

        target_col: Name of target column. Available after import_data

        zdense_col: List of zero dense column names. Available after full_zero_mark

        f0_col: Name of the zero mark column. Available after full_zero_mark

        strt_col: Name of strata column. Available after add_strata

        dev_cols: List of names for device derived columns. Availale after device_features

        date_cols: List of names for date derived columns. Availale after date_features

        chng_cols: List of names for change columns. Availale after change_features

        rank_cols: List of names for rank columns. Availale after rank_features

        ncat_cols: List of names for new categorical columns. Available after add_features

        nnum_cols: List of names for new categorical columns after adding new features. Available after add_features

        ncat_cols: List of names for new numeric columns after adding new features. Available after add_features

        cntr_cols: List of names for control columns (full zero and strata). Available after add_features

        c_trans: Dict of transformation performed on each new numeric feature (key). Available after transform

        pca_cols: List of names for PCA component columns. Available after pca.

        enc_cols: List of names for encoded columns. Available after encode.
    """

    strata_p = {0: 0.1, 1: 0.6, 2: 0.3}
    random_state = 42
    
    @classmethod
    def import_data(cls):
        """ Imports original data.

            Renames columns, and stores columns per type (categorical, numeric, target)
            Removes duplicate rows if any.

            Returns
            -------
            data: DataFrame with X and y with renamed columns

            cat_cols: List of categorical column names

            num_cols: List of numerical column names

            target_col: Name of target column
        """
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

    @classmethod
    def full_zero_mark(cls, X):
        """ Adds column representing if instance has only zeros in zero dense columns.

            Stores a list of zero dense column names. Also the f0_col (zero mark column) name.

            Parameters
            ----------
            X: DataFrame where zero mark column will be added

            Returns
            -------
            Xz: Original DataFrame with the added zero mark column.

            zdense_col: List of zero dense column names.

            f0_col: Name of the zero mark column.
        """
        cls.zdense_cols = [c for c in ['m2', 'm3', 'm4', 'm7', 'm8', 'm9'] if c in X.columns]
        cls.f0_col = 'f0'
        full_zero = pd.DataFrame(X[cls.zdense_cols].sum(axis=1) == 0, columns=[cls.f0_col])
        X.insert(X.shape[1], cls.f0_col, full_zero.astype(int))

        return X, cls.zdense_cols, cls.f0_col
    
    @classmethod
    def add_strata(cls, X, y):
        """ Adds strata column to X. Strata is calculated based on zero mark and target value.

            Stores name for strata column

            Parameters
            ----------
            X: DataFrame where strata column will be added.

            y: Series with target values.

            Returns
            -------
            Xstrt: Dataframe with added strata column

            strt_col: Name of strata column
        """
        strt = pd.Series([0] * X.shape[0], index=X.index)
        strt[(y == 0) & (X[cls.f0_col] == 0)] = 1
        strt[(y == 1)] = 2

        X.insert(X.shape[1], 'strt', strt)
        cls.strt_col = 'strt'

        return X, cls.strt_col
    
    @classmethod
    def remove_duplicates(cls, X):
        """ Removes duplicated columns in X

            Updates stored num_cols without duplicated columns names

            Paramaters
            ----------
            X: DataFrame where duplicate column will be removed

            Returns
            -------
            Xrem: DataFrame with removed column
        """
        X = X.drop('m8', axis=1)
        cls.num_cols.remove('m8')

        return X, cls.num_cols
    
    @classmethod
    def device_features(cls, X):
        """ Adds device derived features.

            Stores new column names.

            Parameters
            ----------
            X: DataFrame where device derived columns will be added.

            Returns
            -------
            Xdev: DataFrame with device derived columns added

            dev_cols: List of names for device derived columns
        """
        X['L1'] = X['device'].str[0:2]
        X['L2'] = X['device'].str[2:4]
        X['L12'] = X['device'].str[0:4]
        #X['L3'] = X['device'].str[4:6]
        #X['L4'] = X['device'].str[6:]

        cls.dev_cols = [c for c in ['L1', 'L2', 'L12', 'L3', 'L4'] if c in X.columns]

        return X, cls.dev_cols

    @classmethod
    def date_features(cls, X):
        """ Adds date derived features.

            Stores new column names.

            Parameters
            ----------
            X: DataFrame where date derived columns will be added.

            Returns
            -------
            Xdt: DataFrame with date derived columns added

            date_cols: List of names for date derived columns
        """
        #X['Y'] = X['date'].dt.year
        X['Q'] = X['date'].dt.quarter
        X['M'] = X['date'].dt.month
        X['D'] = X['date'].dt.day
        X['MW'] = X['D'].div(8).apply(np.floor).add(1).astype(int)
        X['WD'] = X['date'].dt.weekday
        X['WE'] = (X['WD'] > 4).astype(int)

        cls.date_cols = [c for c in ['Y', 'Q', 'M', 'D', 'MW', 'WD', 'WE'] if c in X.columns]

        return X, cls.date_cols

    @classmethod
    def change_metrics(cls, X):
        """ Adds Metric Percentage Change (Time Oriented).

            Stores name of change columns

            Parameters
            ----------
            X: DataFrame where change data will be added

            Returns
            -------
            Xchng: DataFrame with change metrics added

            chng_cols: List of names for change columns
        """
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

    @classmethod
    def rank_metrics(cls, X):
        """ Adds Standardized Rank Metrics (History dependent).

            Stores name of rank columns

            Parameters
            ----------
            X: DataFrame where rank data will be added

            Returns
            -------
            Xrnk: DataFrame with rank metrics added

            rnk_cols: List of names for rank columns
        """
        Xrnk = X[cls.num_cols].rank(method='min').sub(1).div(X.shape[0])
        Xrnk.columns = [c.replace('m', 'sr') for c in Xrnk.columns]
        Xrnk['srav'] = Xrnk.mean(axis=1)
        Xrnk['srmx'] = Xrnk.max(axis=1)

        drop_cols = ['sr1', 'sr5', 'sr6', 'srav']
        if len(drop_cols) > 0: Xrnk.drop(drop_cols, axis=1, inplace=True)

        cls.rnk_cols = Xrnk.columns.tolist()
        X = pd.concat([X, Xrnk], axis=1)

        return X, cls.rnk_cols

    @classmethod
    def add_features(cls, X, y):
        """ Adds all new features (zero mark, strata, device derived, data derived, change, rank) and remove duplocated columns.

            Stores lists for new categorical column names, new numeric column names, and control solumn name (full zero and strata)

            Parameters
            ----------
            X: DataFrame where new features will be added

            y: Series with target values

            Returns
            -------
            Xftr: DataFrame with additional features. 
        """
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

    @classmethod
    def transform(cls, X):
        """ Transforms numeric features based on analysis conclusions.

            Stores transformations performed per numeric column

            Parameters
            ----------
            X: DataFrame where transformation will be performed

            Return
            ------
            Xtrans: DataFrame with numeric data transformed
        """
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
    
    @classmethod
    def pca(cls, X):
        """ Performs PCA transformation on numeric features

            Stores name for PCA component columns.

            Parameters
            ----------
            X: DataFrame where PCA transformation will be performed.

            Returns
            -------
            Xpca: DataFrame with PCA transformed columns.
        """
        c_pca = cls.nnum_cols
        Xpca, cls.pca_cols = Tools.pca(X, c_pca)
        return Xpca
    
    @classmethod
    def encode(cls, X):
        """ Performs One Hot Encoding to categorical columns

            Stores names for encoded column names

            Parameters
            ----------
            X: DataFrame where encoding will be performed

            Returns
            -------
            Xenc: DataFrame with encoded columns
        """
        cat_cols = [c for c in PF.ncat_cols if c not in PF.cat_cols]
        Xenc, cls.enc_cols = Tools.oh_encode(X[cat_cols], cat_cols)

        return Xenc
    
    @classmethod
    def prepare_features(cls, X, num_transform=True):
        """ Performs all preparation steps on data (transform, pca and encode)-

            Parameters
            ----------
            X: DataFrame where prepaation will be performed

            num_transform: If numerical transformation and PCA should be performed

            Returns
            -------
            Xprp: DataFrame with prepared columns. Ready for modeling.

        """
        if num_transform:
            Xnum = cls.transform(X)
            Xnum = cls.pca(Xnum)
        else: Xnum = X[cls.nnum_cols]

        Xcat = cls.encode(X)
        Xprep = pd.concat([Xcat, Xnum, X[cls.cntr_cols]], axis=1)

        return Xprep
    
    @classmethod
    def best_strat_bagg_boost(cls, X_train, strt_train, trained=True):
        """ Ensembles best Stratified Bagg + Boost Model, based on Random Search results.

            Parameters
            ----------
            X_train: DataFrame for training model

            strt_train: Series with strata data for training the model.

            trained: If want model to be trained or not.

            Returns
            -------
            bbagg: Best Stratified Bagg + Boost Model
        """
        strata_p = {0: 0.05, 1: 0.75, 2: 0.2}

        n_vals = Tools.strat_undersample_counts(strt_train, strata_p, cls.random_state)

        tree = DecisionTreeClassifier(class_weight=strata_p, max_depth=40, max_features=0.5)

        boost = AdaBoostClassifier(base_estimator=tree, n_estimators=20)

        bbagg = BalancedBaggingClassifier(base_estimator=boost, n_estimators=200, sampling_strategy=n_vals,
                                          bootstrap=False, random_state=cls.random_state, n_jobs=10)
        
        if trained: bbagg.fit(X_train, strt_train)

        return bbagg

    @classmethod
    def get_metrics(cls, y_true, y_proba, plot=True):
        """ Gets Precision / Recall curve with corresponding F1 score, Best Threshold and Confusion Matrix

            Parameters
            ----------
            y_true: Series with values for target

            y_proba: Series with probabilities for positive target

            plot: If visualizing plot that summarizes results is required.

            Returns
            -------
            pr_data: DataFrame with Precision, Recall curve data and corresponding Threshold and F1 Score

            thrs_best: Threshold for best F1 Score

            conf_matrix: Confusion matrix for best F1 Score
        """
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