import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas.api.types as ptypes
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from .pipe import pipe

score = -1
dtype =""
binary = True

class XGB:
    def __init__(self, trainpath, predictpath, target, savepath, split=0.2,
                num_estimators=[100], depths=[None], learning_rate=[3], exclude_values_limit=-1,
                require_individual_correlation=True,
                cramers_v_cut=0.08, impute_method="median", include_nan=False, breakup=[] ):
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
    def splittrain(self, train):
        is_numeric = ptypes.is_numeric_dtype(train[self.target])
        if is_numeric:
            strat_train_set, strat_test_set = train_test_split(train, test_size=self.split, random_state=42)
        else:
            splits = StratifiedShuffleSplit(n_splits=1, test_size=self.split)
            for train_indices, test_indices in splits.split(train, train[self.target]):
                strat_train_set = train.loc[train_indices]
                strat_test_set = train.loc[test_indices]
            
        return strat_train_set, strat_test_set
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
    def prepforpredict(self, train, test):
        train_columns = set(train.columns)
        test_columns = set(test.columns)

        
        unique_to_train = train_columns - test_columns
        unique_to_test = test_columns - train_columns

        
        print("Columns unique to train:")
        for column in unique_to_train:
            if column != self.target :
                print(column)
                train = train.drop(column, axis=1)

        print("\nColumns unique to test:")
        for column in unique_to_test:
            print(column)
            test = test.drop(column, axis=1)

        num_columns = train.shape[1]
        print("Number of columns:", num_columns)
        return train, test
    def trainontrain(self, train):
        global binary
        #no scale for XGBoost
        X_train = train.drop([self.target], axis=1)
        y_train = train[self.target]

        for column in X_train.columns:
            if X_train[column].dtype == 'object':
                X_train[column] = LabelEncoder().fit_transform(X_train[column].astype(str))

        
        X_data = X_train.to_numpy()
        y_data = y_train.to_numpy()

        parameters = {
    'booster': 'gbtree', 
    'n_estimators': self.num_estimators,
    'max_depth': self.depths,
    'learning_rate': self.learning_rate, 
    'subsample': 0.9947997083813288,
    'colsample_bytree': 0.5336230391923533,
    'gamma': 0.16126940334635828,
    'tree_method': 'hist',  # Use 'hist' with the 'device' parameter
    'device': 'cuda'        # Specify the device as 'cuda' for GPU
}


        if binary:
            howtoscore = "accuracy"
            xgb_model = xgb.XGBClassifier(**parameters)
        else:
            howtoscore = "neg_mean_squared_error"
            xgb_model = xgb.XGBRegressor(**parameters)
        

        param_grid = {
            "n_estimators": self.num_estimators,  # Example: [50, 100, 200]
            "max_depth": self.depths,             # Example: [3, 5, 7]
            "learning_rate": self.learning_rate   # Example: [0.01, 0.1, 0.2]
        }

        # Create the GridSearchCV object
        
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring=howtoscore, return_train_score=True, n_jobs=-1)

        # Fit the model using grid search
        print("grid searching - this takes a while depending on your number of columns and train size")

        grid_search.fit(X_data, y_data)

        # Get the best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best Parameters: {best_params}")
        print(f"Best Accuracy: {best_score}")

        final_xgb = grid_search.best_estimator_

        print("found best ai")
        printthese = ['n_estimators', 'learning_rate', 'max_depth']
        for param_name in final_xgb.get_params():
            if param_name in printthese:
                print(f"{param_name}: {final_xgb.get_params()[param_name]}")


        return final_xgb
    def predictaccuracy(self, model, trained, test):
        global score
        global binary
        train_columns = set(trained.columns)
        test_columns = set(test.columns)

        # Find unique columns in each DataFrame
        unique_to_train = train_columns - test_columns
        unique_to_test = test_columns - train_columns
        
        for column in unique_to_train:
            if column != self.target :
                length = len(test)
                emptycol = np.full(length, 0)
                test[column] = emptycol

        
        for column in unique_to_test:
            test = test.drop(column, axis=1)
        
        X_test = test.drop([self.target], axis=1)
        y_test = test[self.target]

        for column in X_test.columns:
            if X_test[column].dtype == 'object':
                encoder = LabelEncoder()
                X_test[column] = encoder.fit_transform(X_test[column].astype(str))

        
        X_test_data = X_test.to_numpy()
        y_test_data = y_test.to_numpy()

        if binary:
            score = model.score(X_test_data, y_test_data)
        else:
            y_pred = model.predict(X_test_data)
            score = np.sqrt(mean_squared_error(y_test_data, y_pred))


        print("The predicted accuracy for this model (given a simaler prediction set) is: ")
        print(score)
    def finish(self, model, predict, id, savepath, savenames):
        global dtype

        for column in predict.columns:
            if predict[column].dtype == 'object':
                encoder = LabelEncoder()
                predict[column] = encoder.fit_transform(predict[column].astype(str))
        final_predict = predict.to_numpy()

        predictions = model.predict(final_predict)
        final_df = pd.DataFrame(id)

        
        if len(predictions.shape) != 1:
            predictions = predictions[:,0]
        
        final_df[self.target] = predictions

        if savenames[0] :
            final_df[self.target] = final_df[self.target].map({1: savenames[1], 0: savenames[2]})
        
        if dtype == bool:
            final_df[self.target] = final_df[self.target].map({1: True, 0: False})

        final_df[self.target] = final_df[self.target].astype(dtype)
        
        print(final_df.head(20))
        final_df.to_csv(savepath, index=False)
        
        


    def complete(self):
        print("Starting XGB")
        train = pd.read_csv(self.trainpath)
        predictit = pd.read_csv(self.predictpath)

        id = predictit[predictit.columns[0]]
        
        self.firstcolname = train.columns[0]

        if self.dataisbad(train, predictit, self.target):
            print("ERROR PROGRAM SHUT DOWN")
            return

        train = self.prepforuse(train)

        strat_train_set, strat_test_set = self.splittrain(train)

        pipeline = pipe(self.target,require_individual_correlation=self.require_individual_correlation,
                        exclude_values_limit=self.exclude_values_limit,
                        cramers_v_cut=self.cramers_v_cut, impute_method=self.impute_method,
                        include_nan=self.include_nan, breakup=self.breakup)
        savenames, pipeline = pipeline.make()

        strat_train_set = pipeline.fit_transform(strat_train_set)
        print("pipelined train set")
        strat_test_set = pipeline.transform(strat_test_set)
        print("pipelined test set")
        predictit = pipeline.transform(predictit)

        strat_train_set_mod, predictit_mod = self.prepforpredict(strat_train_set, predictit)

        print(strat_train_set_mod.info())
        final_XGB = self.trainontrain(strat_train_set_mod)

        self.predictaccuracy(final_XGB, strat_train_set_mod, strat_test_set)
        print(strat_train_set_mod.info())
        print("____________________")
        print(predictit_mod.info())

        self.finish(final_XGB, predictit_mod, id, self.savepath, savenames)
        print("completed and saved")
       


    def guantlet(self):
        import os
        global score
        global binary
        print("Begun")
        cycle = [True,False]

        def increment_path(base_path, iteration):
            directory, filename = os.path.split(base_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"guantlet_" + str(iteration) + str(ext)
            return os.path.join(directory, new_filename)
        
        i = 1
        bestscore = 0
        if not binary:
            bestscore = 10000000000
        bestfile = -1
        combo = [True, True]
        for tempinclude_nan in cycle:
            for temprequire_individual_correlation in cycle:
                self.savepath = increment_path(self.savepath,i)
                self.include_nan = tempinclude_nan
                self.require_individual_correlation = temprequire_individual_correlation
                
                print("\n " + str(self.include_nan) + "  " + str(self.require_individual_correlation))
                self.complete()
                if binary:
                    if score > bestscore:
                        bestscore = score
                        bestfile = i
                        combo = [tempinclude_nan, temprequire_individual_correlation]
                else:
                    if score < bestscore:
                        bestscore = score
                        bestfile = i
                        combo = [tempinclude_nan, temprequire_individual_correlation]

            
                i += 1
        print("File " + str(bestfile) + " performed best with a score of " + str(bestscore))
        i = "final"
        self.savepath = increment_path(self.savepath,i)
        self.include_nan = combo[0]
        self.require_individual_correlation = combo[1]
        self.split = 0.01
        self.complete()
        print("Guantlet finished")


        







        
        

