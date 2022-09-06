'''
Test Log Module
Author: Akindele Abdulrasheed
Date: Aug 6th, 2022
'''
import os
import logging
import constants as cs

from data_transform import DataTransform
from feature_train import TrainEvaluate


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class DataLogging():
    '''
    Performs Testing on modules on Classes DataTransform and TrainEvaluate
    '''

    def __init__(self):
        self.data = None
        self.label = None
        self.dataloader = DataTransform()
        self.featureloader = None

    def test_import(self, pth):
        '''
        test data import and loads import_data from DataTransform
        Args:
            pth: path that contains data
        '''
        try:
            self.dataloader.import_data(pth)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing import_eda: The file wasn't found")
            raise err  # without data the whole process should be cancelled

    def test_eda(self):
        '''
        Performs eda function from DataTransform and checks if images
        were created.
        '''
        try:
            self.dataloader.perform_eda()
            logging.info("Success: performing exploratory data analysis")
        except NameError:
            logging.error("Error")
        # Checking if all file path were created
        files = ['Churn_plot', 'Customer_age_plot', 'Correlation_plot',
                 'describe_plot', 'Marital_Status_plot', 'Total_Trans_Ct_plot']
        try:
            for path_name in files:
                logging.info("Checking if %s filepath exist", path_name)
                assert os.path.exists(f'./images/eda/{path_name}.png')
                logging.info("Success: file path exist")
        except AssertionError:
            logging.info("Error: filepath doesnt exist")
        logging.info("End of EDA process.")
        logging.info("-------------------\n")

    def test_encoder_helper(self, response):
        '''
        encoder helper function checks if all categorical column
        has been converted to numerical percentage of label column
        Args:
            response: LABEL COLUMN
        '''
        logging.info("-----Data Encoder in progress -----")
        try:
            self.dataloader.encoder_helper(response)
            assert len(self.dataloader.data.select_dtypes(
                include=['object']).columns.tolist()) == 0
            logging.info(
                "Success: All categorical columns have been turned into numerical columns")
        except AssertionError:
            logging.error("Error: data still contain categorical columns")

    def test_perform_feature_engineering(self, text_size):
        '''
        perform_feature_engineering performs feature extraction
        from TrainEvaluate takes in the text size function 
        and creates a test set and a train set for both data and label
        '''
        self.featureloader = TrainEvaluate(
            self.dataloader.data, self.dataloader.label)
        self.featureloader.perform_feature(text_size)
        logging.info("-----Performing Feature Transformation-------")
        try:
            assert len(self.featureloader.x_test) < len(self.dataloader.data)
            assert len(self.featureloader.x_train) < len(self.dataloader.data)
            assert len(self.featureloader.y_test) < len(self.dataloader.label)
            assert len(self.featureloader.y_train) < len(self.dataloader.label)
            logging.info("Success: Feature transform complete")
        except AssertionError:
            logging.error("Failure: Feature Transform not complete")

    def test_train_models(self):
        '''
        test train_models function and checks if LogisticRegression AND RandomForest model
        was created.. It also checks if all model performance files were created
        '''
        self.featureloader.train_models()
        logging.info("Training Model in progress")
        try:
            assert os.path.exists(cs.LOGISTIC_REGRESSION_PATH)
            assert os.path.exists(cs.RANDOM_FOREST_PATH)
            logging.info(
                "Success: Both Logistic and Randomforest model have been created")
        except AssertionError:
            logging.error("Either one or None of Models haven't been created")

        # Assert Model_Performance Files were created
        try:
            assert os.path.exists(cs.COMPARISON_MODEL)
            assert os.path.exists(cs.PLT_LOGISTIC)
            assert os.path.exists(cs.PLT_RANDOMFOREST)
            logging.info(
                "Success: All model performance image as been created")
        except AssertionError:
            logging.error(
                "Error: One or more of model performance image hasn't been created")

    def test_classification_report(self):
        '''
        test classification report function and checks if all image reports have been created
        '''
        try:
            self.featureloader.classification_report_image()
            assert os.path.exists(cs.CLF_RANDOMFOREST_TEST)
            assert os.path.exists(cs.CLF_RANDOMFOREST_TRAIN)
            assert os.path.exists(cs.CLF_LOGREG_TEST)
            assert os.path.exists(cs.CLF_LOGREG_TRAIN)
            logging.info(
                "Success: All classification reports have been created")
        except AssertionError:
            logging.error(
                "One or More of the classifictaion reports have not been created.")

    def test_feature_importance(self):
        '''
        test_feature_importance
        '''
        self.featureloader.feature_importance_plot(cs.OUTPUT_PATH)
        try:
            assert os.path.exists(cs.OUTPUT_PATH)
            logging.info("Success: Feature Importance plot created")
        except AssertionError:
            logging.error("Error: Feature importance plot not created")

    logging.info("---------End-----of Data Churn Process")


if __name__ == "__main__":
    Testing = DataLogging()
    Testing.test_import('./data/bank_data.csv')
    Testing.test_eda()
    Testing.test_encoder_helper('Churn')
    Testing.test_perform_feature_engineering(0.3)
    Testing.test_train_models()
    Testing.test_classification_report()
    Testing.test_feature_importance()
