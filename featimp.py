from operator import mod
import pandas as pd
import numpy as np
import copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

from scipy.stats import spearmanr


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

X = pd.DataFrame(housing['data'], columns=housing['feature_names'])
Y = pd.DataFrame(housing['target'], columns=housing['target_names'])

x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size=0.1, shuffle=True)

class FeatureImportance:


    #Drop column importance
    def dropcol_imp(self, model,x_train, y_train, x_val, y_val):
        model.fit(x_train,y_train)
        baseline_score = self.calculate_r2(model, x_val, y_val)
        score_list = []
        
        for col in x_train.columns:
            # print('Dropping column {}'.format(col))
            x_train_ = x_train.drop(col, axis=1).copy()
            x_val_ = x_val.drop(col, axis=1).copy()
            model_ = copy.deepcopy(model)
            model_.fit(x_train_, y_train)
            score = self.calculate_r2(model_, x_val_, y_val)
            score_list.append(baseline_score-score)
        return baseline_score, score_list

    #Permuation column importance
    def permutation_imp(self, model, x_train, y_train, x_val, y_val):
        model.fit(x_train, y_train)
        baseline_score = self.calculate_r2(model, x_val, y_val)
        score_list = []
        
        for col in x_val.columns:
            orginal_vals = x_val[col].copy()
            x_val[col] = np.random.permutation(x_val[col])
            score = self.calculate_r2(model, x_val, y_val)
            x_val[col] = orginal_vals
            score_list.append(baseline_score-score)
            
        return baseline_score, score_list

    #Minimum redudancy maximum relevance use spearman correlation criterion
    def mrmr_spearman(self, X, Y):
        corr_target = []
        for col in X.columns:
            corr_target.append(spearmanr(X[col],Y)[0])
        corr_target = abs(np.array(corr_target))



        final_df = {'Feature':[], 'Importance':[]}
        features = X.columns
        selected_features = []
        selected_features.append(features[np.argmax(corr_target)])
        final_df = {'Feature':[features[np.argmax(corr_target)]], 'Importance':[max(corr_target)]}
        for i in range(len(features)-1):
            intra = {'Feature':[], 'Importance':[]}
            # print(i)
            for col in features:
                
                if col not in selected_features:
                    # print("feature", col)
                    xi_xs_avg = 0
                    
                    for s in selected_features:
                        xi_xs_avg += abs(spearmanr(X[col],X[s])[0])
                    xi_xs_avg = xi_xs_avg/len(selected_features)
                    intra['Feature'].append(col)
                    intra['Importance'].append(abs(spearmanr(X[col],Y)[0]) - xi_xs_avg)

            df=pd.DataFrame(intra)
            max_score=df.sort_values(by='Importance',ascending=False)['Importance'].values[0]
            next_selected = df.sort_values(by='Importance',ascending=False)['Feature'].values[0]
            final_df['Feature'].append(next_selected)
            final_df['Importance'].append(max_score) 
            selected_features.append(next_selected)
    
        return pd.DataFrame(final_df)
    
    
    # OLS importance    
    def ols_imp(self,X,Y):
        scaled_x = self.__standardize(X)
        model = LinearRegression()
        model.fit(scaled_x, Y)
        imp = self.__get_coeff(model)
        imp = map(abs, imp) #take absolute values
        return imp

    #pca importance
    def pca_imp(self, df):
        scaled_df = self.__standardize(df)
        pca = PCA(n_components=1)
        pca.fit(scaled_df)
        imp = pca.components_[0].tolist()
        imp = map(abs, imp) #take absolute values
        return imp




################################################
############## Utility methodds ################
################################################

    def create_featimp_df(self, df, score_list):
        labels = df.columns
        I = pd.DataFrame(data={'Feature': labels, 'Importance': np.array(score_list)})
        I = I.sort_values('Importance', ascending=False)
        return I
    
    def __standardize(self, df):
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        return scaled_df

    def __get_coeff(self, model):
        return model.coef_[0].tolist()

    def calculate_r2(self, model, x_val, y_val):
        """
        Calculated the accuracy of model on validation set
        """
        y_hat = model.predict(x_val)
        return r2_score(y_val,y_hat)

    def calculate_mae(self, model, x_val, y_val):
        y_hat = model.predict(x_val)
        return mean_absolute_error(y_val, y_hat)

    def automatic_feature_selection(self,x_train, y_train, x_val, y_val):

        def caculate_feat_importance(X):
            feat = FeatureImportance()
            imp_pca = feat.pca_imp(X)
            I_pca = feat.create_featimp_df(X, imp_pca)
            I_pca = I_pca.sort_values(by='Importance')
            I_pca.reset_index(inplace=True, drop=True)
            return I_pca

        model = RandomForestRegressor(n_estimators=30, n_jobs = -1)
        model.fit(x_train,y_train)
        baseline= self.calculate_mae(model, x_val, y_val)
        best = baseline

        I_pca = caculate_feat_importance(x_train)

        features = I_pca['Feature'].values
        dropped=[]
        errors = []
        errors.append(best)
        for i in range(len(I_pca['Feature'])):
            # print(I_pca)
            model.fit(x_train.drop(dropped + [I_pca.iloc[0,0]],axis=1),y_train)
            error = self.calculate_mae(model, x_val.drop(dropped + [I_pca.iloc[0,0]],axis=1),y_val)
            if error <=  best:
                best = error
                errors.append(best)
                # print("dropped",I_pca.iloc[0,0])
                dropped.append(I_pca.iloc[0,0])
                I_pca = caculate_feat_importance(x_train.drop([I_pca.iloc[0,0]]+dropped,axis=1))
        #         dropped.append(I_pca.iloc[0,0])
            else:
                break
        selected_features = np.setdiff1d(x_train.columns,np.array(dropped))
        return selected_features, errors


############################################        
##### Vizualization methods ################
###########################################
    def plot_featimp(self, df):
        size=15
        params = {'legend.fontsize': 'large',
                'figure.figsize': (8,8),
                'axes.labelsize': size,
                'axes.titlesize': size,
                'xtick.labelsize': size*0.90,
                'ytick.labelsize': size*0.90,
                'axes.titlepad': 25}
        plt.rcParams.update(params)

        fig, ax = plt.subplots(1)
        sns.stripplot(x='Importance', y='Feature', data=df, color='black', ax=ax, size=10, jitter=True)
        for i in range(len(df.index)):
            ax.axhline(y=i, xmin=0.0, xmax=1, linestyle='--')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.title("Feature Importance Chart")
        plt.show()

    def plot_compare_strategies(self,df, title):
        size=15
        params = {'legend.fontsize': 'large',
                'figure.figsize': (10,8),
                'axes.labelsize': size,
                'axes.titlesize': size,
                'xtick.labelsize': size*0.90,
                'ytick.labelsize': size*0.90,
                'axes.titlepad': 25}
        plt.rcParams.update(params)
        fig, ax = plt.subplots(1)
        sns.lineplot(data=df, x=df.index.values, y=df['drop'].values, ax=ax, marker='p', markersize=14, label='Drop Column')
        sns.lineplot(data=df, x=df.index.values, y=df['permu'].values, ax=ax, marker='*', markersize=14, label='Permu Column')
        sns.lineplot(data=df, x=df.index.values, y=df['mrmr'].values, ax=ax, marker='P', markersize=14, label='mrMR Spearman')
        sns.lineplot(data=df, x=df.index.values, y=df['ols'].values, ax=ax, marker='D', markersize=14, label='OLS')
        sns.lineplot(data=df, x=df.index.values, y=df['pca'].values, ax=ax, marker='s', markersize=14, label='PCA')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel('Top K features')
        ax.set_ylabel('Validation MAE')
        plt.title(title)
        plt.show()

    def plot_automatic_feat_selection(self, s):
        size=15
        params = {'legend.fontsize': 'large',
                'figure.figsize': (10,8),
                'axes.labelsize': size,
                'axes.titlesize': size,
                'xtick.labelsize': size*0.90,
                'ytick.labelsize': size*0.90,
                'axes.titlepad': 25}
        plt.rcParams.update(params)
        fig, ax = plt.subplots(1)
        sns.lineplot(data=s, x=s.index, y=s['Validation MAE'], marker='o', markersize=14)
        ax.set_xticks([0,1,2,3,4])
        ax.set_xlabel("Number of features dropped")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()

    #################################################
    ############# Compare feature rankings ##########
    #################################################
    def compare_feat_ranking(self, I_drop, I_permu, I_mrmr, I_ols, I_pca, model):
        mae_drop = []
        for i in range(1,len(I_drop['Feature'])+1):
            features = I_drop['Feature'].values.tolist()[:i]
            model.fit(x_train[features], y_train)
            mae_drop.append(self.calculate_mae(model, x_val[features], y_val))

        mae_permu = []
        for i in range(1,len(I_permu['Feature'])+1):
            features = I_permu['Feature'].values.tolist()[:i]
            model.fit(x_train[features], y_train)
            mae_permu.append(self.calculate_mae(model, x_val[features], y_val))
            
        mae_mrmr=[]
        for i in range(1,len(I_mrmr['Feature'])+1):
            features = I_mrmr['Feature'].values.tolist()[:i]
            model.fit(x_train[features], y_train)
            mae_mrmr.append(self.calculate_mae(model, x_val[features], y_val))
            
        mae_ols=[]
        for i in range(1,len(I_ols['Feature'])+1):
            features = I_ols['Feature'].values.tolist()[:i]
            model.fit(x_train[features], y_train)
            mae_ols.append(self.calculate_mae(model, x_val[features], y_val))
            
        mae_pca=[]
        for i in range(1,len(I_pca['Feature'])+1):
            features = I_pca['Feature'].values.tolist()[:i]
            model.fit(x_train[features], y_train)
            mae_pca.append(self.calculate_mae(model, x_val[features], y_val))

        names = ['drop', 'permu', 'mrmr', 'ols', 'pca']
        df = pd.DataFrame(dict(zip(names,[mae_drop, mae_permu, mae_mrmr, mae_ols, mae_pca])))
        df.index=[1,2,3,4,5,6,7,8]
        return df