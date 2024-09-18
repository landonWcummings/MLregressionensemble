import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import pandas.api.types as ptypes
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import *
from sklearn.preprocessing import *

import optuna

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import *
from tqdm import tqdm, trange
import random

import re

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from .pipe import pipe

from sklearn.model_selection import cross_val_score



score = -1
dtype =""
binary = True

class LGB:
    def __init__(self, trainpath, predictpath, target, savepath, split=0.2,
                num_estimators=[100], depths=[None], learning_rate=[3], exclude_values_limit=-1,
                require_individual_correlation=True,
                cramers_v_cut=0.08, impute_method="median", include_nan=False, 
                breakup=[],optuna=False,optuna_depth=15):
        self.trainpath = trainpath
        self.predictpath = predictpath
        self.target = target
        self.split = split
        self.num_estimators = num_estimators
        self.depths = depths
        self.learning_rate = learning_rate
        self.savepath = savepath
        self.exclude_values_limit = exclude_values_limit
        self.require_individual_correlation = require_individual_correlation
        self.cramers_v_cut = cramers_v_cut
        self.impute_method = impute_method
        self.include_nan = include_nan
        self.breakup = breakup
        self.optuna = optuna
        self.optuna_depth = optuna_depth

        self.firstcolname = ""

    def prepforuse(self,train):
        global dtype
        global binary
        dtype = train[self.target].dtype
        if dtype == bool:
            train[self.target] = train[self.target].map({False: 0, True: 1})
        
        unique_values = train[self.target].unique()
        if len(unique_values) == 2:
            binary = True
        else:
            binary = False

        for column in train.columns:
            unique_types = set(train[column].apply(type))

            if train[column].dtype == bool:
                train[column] = train[column].map({False: 0, True: 1})

       
        return train        
    
    def dataisbad(self, train, predictit, target):

        if target not in train.columns:
            print("ERROR ------ your target is not found in the train data")
            return True

        if len(train.columns) > len(predictit.columns) + 1:
            print("ERROR ------ your prediction data has notably less columns than the train data")
        
        if len(train.columns) <= len(predictit.columns):
            print("ERROR ------ your prediction data has more columns than the train set")

        if len(self.breakup) != 0 and len(self.breakup) % 2 != 0:
            print("ERROR ------ your breakup array must contain an even number of features. Canceled breakup")
            self.breakup = []

        j = 0
        while j< len(self.breakup):
            if self.breakup[j] not in train.columns and "_part_" not in self.breakup[j]:
                print("ERROR ------ column %s in your breakup array doesn't exist. It is deleted from breakup" % self.breakup[j])
                self.breakup.pop(j)
                self.breakup.pop(j)
                j -= 2
            j += 2


        return False
    
    def complete(self):
        print("Starting LGB")
        train = pd.read_csv(self.trainpath)
        predictit = pd.read_csv(self.predictpath)

        id = predictit[predictit.columns[0]]
        
        #self.firstcolname = train.columns[0]

        if self.dataisbad(train, predictit, self.target):
            print("ERROR PROGRAM SHUT DOWN")
            return

        train = self.prepforuse(train)

        #strat_train_set, strat_test_set = self.splittrain(train)

        pipeline = pipe(self.target,require_individual_correlation=self.require_individual_correlation,
                        exclude_values_limit=self.exclude_values_limit,
                        cramers_v_cut=self.cramers_v_cut, impute_method=self.impute_method,
                        include_nan=self.include_nan, breakup=self.breakup)
        savenames, pipeline = pipeline.make()

        strat_train_set = pipeline.fit_transform(train)
        print("pipelined train set")
        #strat_test_set = pipeline.transform(predictit)
        print("pipelined test set")
        predictit = pipeline.transform(predictit)
        
        

        #strat_train_set['transmission'] = strat_train_set['transmission'].str.replace('/', '').str.replace('-', '')
        #predictit['transmission'] = predictit['transmission'].str.replace('/', '').str.replace('-', '')
        

        for col in strat_train_set:
            strat_train_set[col] = strat_train_set[col].astype('category')
        for col in predictit:
            predictit[col] = predictit[col].astype('category')

        
        test = predictit
        X = strat_train_set.drop(self.target, axis=1)
        y = strat_train_set[self.target].astype({self.target: 'float32'})

        SEED = random.randint(1,100)
        n_splits = 15

        class Auto_ML:
            def __init__(self, models, n_splits=n_splits, seed=SEED):
                self.models = models
                self.n_splits = n_splits
                self.seed = seed
                self.results = {}
            
            def Train_M(self, X, y, test):
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
                
                all_oof_preds = []
                test_preds_list = []
                
                for model_name, model in tqdm(self.models, desc="Training Models"):
                    oof_preds = np.zeros(X.shape[0])
                    test_preds = np.zeros(test.shape[0])
                    val_rmse_list = []
                    
                    for fold_idx, (train_index, val_index) in tqdm(enumerate(kf.split(X)), desc=f"Model: {model_name}", total=self.n_splits):
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                        
                        model_clone = clone(model)
                        model_clone.fit(X_train, y_train)
                        
                        val_preds = model_clone.predict(X_val)
                        oof_preds[val_index] = val_preds
                        
                        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
                        val_rmse_list.append(val_rmse)
                        
                        test_preds += model_clone.predict(test)
                                    
                    mean_test_preds = test_preds / self.n_splits
                    mean_val_rmse = np.mean(val_rmse_list)
                    
                    self.results[model_name] = {
                        'per_fold_rmse': val_rmse_list,
                        'mean_rmse': mean_val_rmse
                    }
                    
                    all_oof_preds.append(oof_preds)
                    test_preds_list.append(mean_test_preds)
                                    

                results_df = self.display_results()
                
                test_preds_dict = {model_name: preds for model_name, preds in zip([model_name for model_name, _ in self.models], test_preds_list)}
                
                return all_oof_preds, test_preds_dict

            def display_results(self):
                fold_columns = [f'Fold {i+1}' for i in range(self.n_splits)]
                results_data = {}

                for model_name, metrics in self.results.items():
                    if 'per_fold_rmse' in metrics:
                        fold_rmse = metrics['per_fold_rmse']
                        results_data[model_name] = fold_rmse + [metrics['mean_rmse']]
                    else:
                        results_data[model_name] = [np.nan] * self.n_splits + [metrics['mean_rmse']]
                    
                
                results_df = pd.DataFrame(results_data, index=fold_columns + ['Mean'])
                print(results_df)
                


                return results_df


        lgb_params1 = {
            'learning_rate': 0.02382055632394208, 'max_depth': self.depths[0], 'reg_alpha': 2.3528303290567725,
            'reg_lambda': 8.794253825944624, 'num_leaves': 19, 'subsample': 0.7285056578914512,
            'colsample_bytree': 0.4455390703746619
        }
        if self.optuna:
            print("Looking for the best hyperparamters")
            # Define the objective function to optimize
            def objective(trial):
                # Define the search space for hyperparameters
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),  # Use suggest_float with log=True
                    'max_depth': trial.suggest_int('max_depth', 4, 50),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Use suggest_float without log=True for uniform
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Use suggest_float without log=True for uniform
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),  # Use suggest_float with log=True
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),  # Use suggest_float with log=True
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'verbose': -1
                }
                

                # Create the model with the given parameters
                model = LGBMRegressor(**params)
                
                # Evaluate the model using cross-validation
                score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=4).mean()
                
                return score

            # Create the Optuna study
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            study = optuna.create_study(direction='maximize')

            with tqdm(total=self.optuna_depth, desc="Hyperparameter Optimization") as pbar:
                def tqdm_callback(study, trial):
                    pbar.update(1)
                study.optimize(objective, n_trials=self.optuna_depth, callbacks=[tqdm_callback])

            best_params = study.best_params
            print(f"Best parameters: {best_params}")



            lgb1 = LGBMRegressor(**best_params, random_state=SEED, verbose=-1)
        else:
            lgb1 = LGBMRegressor(**lgb_params1, random_state=SEED, verbose=-1, n_estimators=self.num_estimators)



        models = [
            ('LGBM_Tunned_100', lgb1),
        ]
        
        Train_m = Auto_ML(models, n_splits=n_splits, seed=SEED)
        oof_preds_all, test_preds_dict = Train_m.Train_M(X, y, test)
        mp1 = test_preds_dict['LGBM_Tunned_100']
       

        ep = mp1

        final_df = pd.DataFrame(id)

        final_df['price'] = ep
        final_df.to_csv(self.savepath, index=False)
        print(final_df.head())
        print("done")



        
