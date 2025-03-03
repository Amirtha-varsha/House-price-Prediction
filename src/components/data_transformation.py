import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def remove_unnecessary_columns(self, df):
        logging.info("Dropping unnecessary columns")
        columns_to_drop=['area_type', 'society', 'balcony', 'availability']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        return df

    def handle_missing_values(self, df):
        logging.info("Dropping null values")
        return df.dropna()

    def create_bhk_feature(self, df):
        logging.info("Creating BHK feature")
        df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
        return df

    def convert_sqft_to_num(self, x):
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None

    def transform_total_sqft(self, df):
        logging.info("Converting total_sqft to numerical values")
        df['total_sqft'] = df['total_sqft'].apply(self.convert_sqft_to_num)
        return df[df['total_sqft'].notnull()]

    def calculate_price_per_sqft(self, df):
        logging.info("Calculating price per square foot")
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        return df

    def handle_location_feature(self, df):
        logging.info("Handling location feature")
        df['location'] = df['location'].apply(lambda x: x.strip())
        location_stats = df['location'].value_counts(ascending=False)
        location_stats_less_than_10 = location_stats[location_stats <= 10]
        df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
        return df

    def remove_outliers(self, df):
        logging.info("Removing outliers based on price per square foot")
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    def remove_bhk_outliers(self, df):
        logging.info("Removing BHK outliers")
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["total_sqft", "bath", "bhk"]
            categorical_columns = ["location"]
            
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            print(f"Type of preprocessor before returning: {type(preprocessor)}")
            
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    
    
        
    def apply_transformations(self, df):
        logging.info("Applying all transformations")
        df = self.remove_unnecessary_columns(df)
        df = self.handle_missing_values(df)
        df = self.create_bhk_feature(df)
        df = self.transform_total_sqft(df)
        df = self.calculate_price_per_sqft(df)
        df = self.handle_location_feature(df)
        df = df[~(df.total_sqft / df.bhk < 300)]
        df = self.remove_outliers(df)
        df = self.remove_bhk_outliers(df)
        df = df[df.bath < df.bhk + 2]
        df = df.drop(['size', 'price_per_sqft'], axis='columns')
        
        return df


        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            
            preprocessor_obj = self.get_data_transformer_object()

            
            input_feature_train_df = train_df.drop(columns=["price"], axis=1)
            target_feature_train_df = train_df["price"]
            input_feature_test_df = test_df.drop(columns=["price"], axis=1)
            target_feature_test_df = test_df["price"]
            
            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()
                input_feature_test_arr = input_feature_test_arr.toarray()
            print("Input Feature Train Shape:", input_feature_train_arr.shape)
            print("Target Feature Train Shape:", target_feature_train_df.shape)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            print(f"Type of preprocessor_obj before saving: {type(preprocessor_obj)}")

            
            logging.info("Saving preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)


            