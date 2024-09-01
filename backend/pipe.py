import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import chi2_contingency
#save the names of the target values that are encoded
savenames = [False,"",""]

class pipe:
    def __init__(self, target, exclude_values_limit=-1, cramers_v_cut=0.08, 
                 istrain=True, impute_method="median", require_individual_correlation=True, include_nan=False, breakup=[]):
        self.target = target
        self.exclude_values_limit = exclude_values_limit
        self.cramers_v_cut = cramers_v_cut
        self.istrain = istrain
        self.impute_method = impute_method
        self.require_individual_correlation = require_individual_correlation
        self.include_nan = include_nan
        self.breakup = breakup

        if exclude_values_limit == -1:
            self.exclude_values_limit = 10
        


    def make(self):
        originals = []
        class prepare(BaseEstimator, TransformerMixin):
            def __init__(self, params):
                self.target, self.exclude_values_limit, self.cramers_v_cut, self.istrain, self.impute_method, self.require_individual_correlation, self.include_nan, self.breakup = params


            def fit(self, X, y=None):
                return self

            def transform(self, X):
                cramerdrops = 0
                cramerspasses = 0
                crameravg = 0
                def convert_to_most_common_dtype(df, column_name):
                    # Get the data types of the values in the column
                    types = df[column_name].apply(type)
                    
                    # Count the occurrences of each type
                    type_counts = types.value_counts()
                    
                    # Get the most common data type
                    most_common_type = type_counts.idxmax()
                    
                    # Convert the column to the most common data type
                    if most_common_type == str or most_common_type == bool:
                        # Convert to string
                        df[column_name] = df[column_name].astype(str)
                    elif most_common_type == int:
                        # Convert to integer
                        df[column_name] = df[column_name].fillna(0).astype(int)
                    elif most_common_type == float:
                        # Convert to float
                        df[column_name] = df[column_name].astype(float)
                    else:
                        # Handle any other types as needed
                        raise ValueError(f"Unsupported data type: {most_common_type}")

                    return df
                
                # Function to calculate Cramér's V
                def cramers_v(confusion_matrix):
                    chi2 = chi2_contingency(confusion_matrix)[0]
                    n = confusion_matrix.sum()
                    cramers_v_value = np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
                    
                    return np.nan_to_num(cramers_v_value)

                #breakup here
                if len(self.breakup) != 0:  # breakup function
                    j = 0
                    while j < (len(self.breakup)):
                        if self.breakup[j] in X.columns:
                            split_columns = X[self.breakup[j]].str.split(self.breakup[j+1], expand=True)

                            for col in split_columns.columns:
                                numeric_count = pd.to_numeric(split_columns[col], errors='coerce').notnull().sum()
                                total_count = len(split_columns[col])
                                
                                if numeric_count / total_count >= 0.7:
                                    split_columns[col] = pd.to_numeric(split_columns[col], errors='coerce')
                                    split_columns[col] = split_columns[col].astype('float64')

                            split_columns.columns = [f'{str(self.breakup[j])}_part_{i+1}' for i in range(split_columns.shape[1])]
                            X = pd.concat([X, split_columns], axis=1)
                            X = X.drop(self.breakup[j], axis=1)
                            j += 2
                
                originals = X.select_dtypes(include=['object']).columns
                originals = originals.difference([self.target])

                categorical_columns = X.select_dtypes(include=['object']).columns
                

                for column in categorical_columns:

                    if self.include_nan:
                        X[column] = X[column].fillna('no_value')
                    else:
                        X[column] = X[column].fillna('Missing')
                    
                    unique_types = set(X[column].apply(type))
                    if len(unique_types) > 1:
                        X = convert_to_most_common_dtype(X,column)
                    
                    if self.istrain:
                        if column != self.target and self.require_individual_correlation:
                            
                            
                            unique_values = X[column].value_counts()

                            # Filter out unique values that occur less than 50 times
                            frequent_values = unique_values[unique_values > self.exclude_values_limit].index

                            # Calculate Cramér's V for each of these frequent unique values
                            for value in frequent_values:
                                binary_column = X[column] == value  # Create a binary column
                                contingency_table = pd.crosstab(binary_column, X[self.target])
                                cramers_v_value = cramers_v(contingency_table.values)
                                

                                if(cramers_v_value < self.cramers_v_cut):
                                    X[column] = X[column].replace(value, 'Missing')
                                    cramerdrops += 1
                                    #killed uninportant values
                                else: 
                                    cramerspasses += 1
                                    crameravg += cramers_v_value



                    for column in categorical_columns:
                        value_counts = X[column].value_counts()

                        # Create a mask for values that occur less than x times
                        rare_values = value_counts[value_counts < self.exclude_values_limit].index.tolist()

                        # Replace those values with "missing"
                        X[column] = X[column].replace(rare_values, "Missing")
                self.istrain = False
                if(cramerspasses == 0):
                    print("prepped - cramer dropped " + str(cramerdrops) + " values.")
                else:
                    print("prepped - cramer dropped " + str(cramerdrops) + " values. Avg cramers V is " + str(crameravg / cramerspasses))
                return X
            
                    
        class Imputermain(BaseEstimator, TransformerMixin):
            def __init__(self, params):
                self.target, self.exclude_values_limit, self.cramers_v_cut, self.istrain, self.impute_method, self.require_individual_correlation, self.include_nan, self.breakup = params

            def fit(self, X, y=None):
                return self

            def transform(self, X):

                imputer = SimpleImputer(strategy=self.impute_method)
                categorical_columns = X.select_dtypes(include=['number']).columns
                for column in categorical_columns:
                    X[column] = imputer.fit_transform(X[[column]])

                print("imputed")
                return X
                        
        
        class FeatureEncoder(BaseEstimator, TransformerMixin):
            def __init__(self, params):
                self.target, self.exclude_values_limit, self.cramers_v_cut, self.istrain, self.impute_method, self.require_individual_correlation, self.include_nan, self.breakup = params

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                global savenames
                categorical_columns = X.select_dtypes(include=['object']).columns
                all_encoded_dfs = []

                for i, column in enumerate(categorical_columns):
                    if column != self.target:
                        column_names = X[column].unique()
                        column_names = [f"{name}_{i+1}" if name != 'Missing' else name for name in column_names]

                        # Encode and create a DataFrame
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        matrix = encoder.fit_transform(X[[column]])
                        encoded_df = pd.DataFrame(matrix, columns=column_names, index=X.index)
                        try:
                            encoded_df = encoded_df.astype('int8')
                        except:
                            pass

                        all_encoded_dfs.append(encoded_df)
                    else:
                        # target column
                        unique_values = X[self.target].unique()

                        if len(unique_values) == 2:
                            value1, value2 = unique_values[0], unique_values[1]
                            X[self.target] = X[self.target].map({value1: 0, value2: 1})
                            savenames[0] = True
                            savenames[1] = value1
                            savenames[2] = value2
                        else:
                            savenames[0] = False

                        
                        all_encoded_dfs.append(X[[self.target]])
                        i -= 1

                # Drop original categorical columns except 'class'
                X_dropped = X.drop(categorical_columns.difference([self.target]), axis=1)

                # Concatenate all encoded dataframes with the rest of X
                final_df = pd.concat([X_dropped] + all_encoded_dfs, axis=1)

                

                print("encoded")
                return final_df
            
        class FeatureDropper(BaseEstimator, TransformerMixin):
            def __init__(self, params):
                self.target, self.exclude_values_limit, self.cramers_v_cut, self.istrain, self.impute_method, self.require_individual_correlation, self.include_nan, self.breakup = params
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                X = X.drop(['Missing'], axis=1, errors='ignore')
                return X.drop(originals, axis=1, errors='ignore')
        

        passpipeline = [self.target, self.exclude_values_limit, self.cramers_v_cut, self.istrain, self.impute_method, self.require_individual_correlation, self.include_nan, self.breakup]
        return  savenames, Pipeline([
                            ('prepare', prepare(passpipeline)),
                            ('ageimputer', Imputermain(passpipeline)),
                            ('featureencoder', FeatureEncoder(passpipeline)),
                            ('featuredropper', FeatureDropper(passpipeline))
                            ])
        
        
