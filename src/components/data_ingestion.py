import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass,field 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = field(default_factory=lambda:os.path.join('artifacts',"train.csv"))
    test_data_path:str = field(default_factory=lambda:os.path.join('artifacts',"test.csv"))
    raw_data_path:str = field(default_factory=lambda:os.path.join('artifacts',"data.csv"))

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('D:\\Desktop\\project\\Houseprice\\Notebook\\bengaluru_house_prices.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Applying data transformation before splitting")
            data_transformation = DataTransformation()
            transformed_df = data_transformation.apply_transformations(df)
            logging.info("Train test initiated")
            train_set,test_set=train_test_split(transformed_df,test_size=0.2,random_state=10)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion and Transformation of data completed")
            print("Train_df shape:",train_set.shape)
            print("Test_df shape:",test_set.shape)
            logging.info(f"Train Columns: {train_set.columns.tolist()}")
            logging.info(f"Test Columns: {test_set.columns.tolist()}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
   
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(pd.read_csv(train_data), pd.read_csv(test_data)))


