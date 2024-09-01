import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas.api.types as ptypes
from .pipe import pipe

score = -1
dtype =""

class RFC:
    def __init__(self, trainpath, predictpath, target, savepath, split=0.2,
                num_estimators=[100], depths=[None], min_samples_split=[3], exclude_values_limit=-1,
                require_individual_correlation=True,
                cramers_v_cut=0.08, impute_method="median", include_nan=False, breakup=[] ):
        self.trainpath = trainpath
        self.predictpath = predictpath
        self.target = target
        self.split = split
        self.num_estimators = num_estimators
        self.depths = depths
        self.min_samples_split = min_samples_split
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
        dtype = train[self.target].dtype
        if dtype == bool:
            train[self.target] = train[self.target].map({False: 0, True: 1})
        
        
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
            if self.breakup[j] not in train.columns:
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
        X_train = train.drop([self.target], axis=1)
        y_train = train[self.target]

        scaler = StandardScaler()
        X_data = scaler.fit_transform(X_train)
        y_data = y_train.to_numpy()

        clf = RandomForestClassifier()

        param_grid = [
            {"n_estimators": self.num_estimators, "max_depth": self.depths, "min_samples_split": self.min_samples_split}
        ]

        grid_search = GridSearchCV(clf, param_grid, cv=3,
                                scoring="accuracy",
                                return_train_score=True)


        print("grid searching - this takes a while depending on your number of columns and train size")
        grid_search.fit(X_data, y_data)


        final_clf = grid_search.best_estimator_

        print("found best ai")
        printthese = ['n_estimators', 'min_samples_split', 'max_depth']
        for param_name in final_clf.get_params():
            if param_name in printthese:
                print(f"{param_name}: {final_clf.get_params()[param_name]}")


        return final_clf
    def predictaccuracy(self, model, trained, test):
        global score
        train_columns = set(trained.columns)
        test_columns = set(test.columns)

        # Find unique columns in each DataFrame
        unique_to_train = train_columns - test_columns
        unique_to_test = test_columns - train_columns
        
        for column in unique_to_train:
            if column != self.target :
                test[column] = 0

        
        for column in unique_to_test:
            test = test.drop(column, axis=1)
        
        X_test = test.drop([self.target], axis=1)
        y_test = test[self.target]

        scaler = StandardScaler()
        X_test_data = scaler.fit_transform(X_test)
        y_test_data = y_test.to_numpy()

        score = model.score(X_test_data, y_test_data)

        print("The predicted accuracy for this model (given a simaler prediction set) is: ")
        print(score)
    def finish(self, model, predict, id, savepath, savenames):
        global dtype
        scaler = StandardScaler()
        scaled_predict = scaler.fit_transform(predict)

        predictions = model.predict(scaled_predict)
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
        print("Starting RFC")
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
        final_RFC = self.trainontrain(strat_train_set_mod)

        self.predictaccuracy(final_RFC, strat_train_set_mod, strat_test_set)
        print(strat_train_set_mod.info())
        print("____________________")
        print(predictit_mod.info())

        self.finish(final_RFC, predictit_mod, id, self.savepath, savenames)
        print("completed and saved")


    def guantlet(self):
        import os
        global score
        print("Begun")
        cycle = [True,False]

        def increment_path(base_path, iteration):
            directory, filename = os.path.split(base_path)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{iteration}{ext}"
            return os.path.join(directory, new_filename)


        # Example usage: Save 5 files with incremented filenames
        
        i = 1
        topscore = 0
        bestfile = -1
        combo = [True, True]
        for include_nan in cycle:
            for require_individual_correlation in cycle:
                self.savepath = increment_path(self.savepath,i)
                self.include_nan = include_nan
                self.require_individual_correlation = require_individual_correlation
                self.complete()
                if score > topscore:
                    topscore = score
                    bestfile = i
                    combo = [include_nan, require_individual_correlation]

            
                i += 1
        print("File " + str(bestfile) + " performed best")
        i = 00000
        self.savepath = increment_path(self.savepath,i)
        self.include_nan = combo[0]
        self.require_individual_correlation = combo[1]
        self.split = 0.01
        self.complete()
        print("Guantlet finished")


        







        
        

