import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
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
        return df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

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

    def apply_transformations(self, df):
        logging.info("Applying all transformations")
        df = self.remove_unnecessary_columns(df)
        df = self.handle_missing_values(df)
        df = self.create_bhk_feature(df)
        df = self.transform_total_sqft(df)
        df = self.calculate_price_per_sqft(df)
        df = self.handle_location_feature(df)
        df = self.remove_outliers(df)
        df = self.remove_bhk_outliers(df)
        df = df[df.bath < df.bhk + 2]
        df = df.drop(['size', 'price_per_sqft'], axis='columns')
        dummies = pd.get_dummies(df['location'])
        df = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
        df = df.drop('location', axis='columns')
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df = self.apply_transformations(train_df)
            test_df = self.apply_transformations(test_df)
            logging.info("Saving transformation results")
            # Save transformed data as an object (dictionary)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={"train": train_df, "test": test_df})
            logging.info("Data transformation completed")
            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)
            