import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            # Convert NumPy arrays back to DataFrames
            column_names = [f"feature_{i}" for i in range(train_array.shape[1] - 1)] + ["price"]
            train_df = pd.DataFrame(train_array, columns=column_names)
            test_df = pd.DataFrame(test_array, columns=column_names)
            # Now dropping the target column properly
            logging.info("split training and test input data")
            X_train, y_train = train_df.drop(columns=["price"]).values, train_df["price"].values
            X_testt, y_test = test_df.drop(columns=["price"]).values, test_df["price"].values
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_testt.shape)
            model = LinearRegression()
            model.fit(X_train, y_train)
            # Save trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            
            predicted=model.predict(X_testt)
            r2_square=r2_score(y_test,predicted)
            logging.info(f"Linear Regression Model trained successfully with R²: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
