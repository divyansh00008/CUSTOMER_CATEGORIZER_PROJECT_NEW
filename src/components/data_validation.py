import json
import sys
from typing import Tuple, Union
import pandas as pd

from pandas import DataFrame

from evidently import Report
from evidently.presets import DataDriftPreset



from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig

from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, write_yaml_file


class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config

        self.utils = MainUtils()
        self._schema_config = self.utils.read_schema_config_file()

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        """
        Method Name :   validate_schema_columns
        Description :   Validates the schema columns for the particular dataframe
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present[{status}]")
            return status
        except Exception as e:
            raise CustomerException(e, sys) from e

    def validate_dataset_schema_columns(self, train_set, test_set) -> Tuple[bool, bool]:
        """
        Method Name :   validate_dataset_schema_columns
        Description :   Validates schema columns for both train and test set
        """
        logging.info("Entered validate_dataset_schema_columns method of Data_Validation class")

        try:
            logging.info("Validating dataset schema columns")

            train_schema_status = self.validate_schema_columns(train_set)
            logging.info("Validated dataset schema columns on the train set")

            test_schema_status = self.validate_schema_columns(test_set)
            logging.info("Validated dataset schema columns on the test set")

            logging.info("Validated dataset schema columns")
            return train_schema_status, test_schema_status

        except Exception as e:
            raise CustomerException(e, sys) from e

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        """
        Method Name :   detect_dataset_drift
        Description :   Detects dataset drift using Evidently (0.7.x) DataDriftPreset.
                        Saves full report (as dict) to YAML path configured.
        Output      :   Returns True if dataset drift is detected, else False.
        """
        try:
        

            report = Report([DataDriftPreset()])

            evaluation=report.run(current_df,reference_df)
            
        
            
            print(evaluation)
            report_dict = evaluation.dump_dict()
            write_yaml_file(
                self.data_validation_config.drift_report_file_path,
                report_dict
            )

            # pretty HTML
            # report.save_html("data_drift_report.html")

            # Extract key numbers & drift status from the new structure update 
            metrics = report_dict.get("metrics", [])
            drift_status = False
            n_features = None
            n_drifted = None

            # Find the preset-level summary (contains dataset_drift + counts)
            

            for m in report_dict.get("metrics", []):
               res = m.get("result", {})
               if "dataset_drift" in res or "number_of_drifted_columns" in res:
                   drift_status = bool(res.get("dataset_drift", False))
                   n_features = res.get("number_of_columns")
                   n_drifted = res.get("number_of_drifted_columns")
                   break

            if n_features is not None and n_drifted is not None:
               logging.info(f"{n_drifted}/{n_features} drift detected.")
            return drift_status

        except Exception as e:
           raise CustomerException(e, sys) from e
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   Orchestrates validation: schema checks + drift detection.
        """
        logging.info("Entered initiate_data_validation method of Data_Validation class")

        try:
            logging.info("Initiated data validation for the dataset")

            train_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            drift = self.detect_dataset_drift(train_df, test_df)

            (
                schema_train_col_status,
                schema_test_col_status,
            ) = self.validate_dataset_schema_columns(train_set=train_df, test_set=test_df)

            logging.info(
                f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}"
            )

            # Same logic as before: valid only if both schemas match AND no drift detected.
            validation_status = (
                schema_train_col_status is True
                and schema_test_col_status is True
                and drift is False
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
