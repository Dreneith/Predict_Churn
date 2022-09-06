'''
data_transform module
Author: Akindele Abdulrasheed
Date: Aug 6th, 2022
'''
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

from constants import COMPARISON_MODEL, grid_params, LOGISTIC_REGRESSION_PATH, OUTPUT_PATH, PLT_LOGISTIC, PLT_RANDOMFOREST, RANDOM_FOREST_PATH, CLF_RANDOMFOREST_TEST, CLF_RANDOMFOREST_TRAIN, CLF_LOGREG_TEST, CLF_LOGREG_TRAIN
from data_transform import DataTransform
from functions import create_reports, fit_predict, plt_model_result

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


class TrainEvaluate():
    '''
    this class performs feature engineering, train models
    and save reports and performance as png file
    '''

    def __init__(self, data, label):
        # trainig data and label
        self.data = data
        self.label = label
        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.y_train = None
        self.rf_ytrain_pred = None
        self.rf_ytest_pred = None
        self.lf_ytrain_pred = None
        self.lf_ytest_pred = None

    def perform_feature(self, text_size):
        '''
        Creates a training set and evaluations set from data and label column
        Args:
            text_size: percentage size of evaluation set
        '''
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.label, test_size=text_size, random_state=42)
        return self.x_train, self.x_test, self.y_test, self.y_train

    def train_models(self):
        '''
        train, store model results: images + scores, and store models
        '''
        # initiate an instance of a Random forest classifier
        random_cl = RandomForestClassifier(random_state=42)
        # Initiate an instance of logistic Regression with solver set to lbfgs
        # and max_iteration at 3000
        log_cl = LogisticRegression(solver='lbfgs', max_iter=300)

        # Initiating Grid_search on our RandomForest Classifier
        grid_rfc = GridSearchCV(
            estimator=random_cl,
            param_grid=grid_params,
            cv=5)
        # Fitting  and predicting with our model
        self.rf_ytrain_pred, self.rf_ytest_pred = fit_predict(
            grid_rfc, self.x_train, self.x_test, self.y_train, grid=True)
        self.lf_ytrain_pred, self.lf_ytest_pred = fit_predict(
            log_cl, self.x_train, self.x_test, self.y_train)
        # plotting logistic model_performance
        plt_model_result(log_cl, self.x_test, self.y_test, PLT_LOGISTIC)
        plt_model_result(
            grid_rfc.best_estimator_,
            self.x_test,
            self.y_test,
            PLT_RANDOMFOREST)

        # Plotting a comparison between logistics and RandomForest
        plt.figure(figsize=(15, 8))
        axis = plt.gca()
        plot_roc_curve(
            grid_rfc.best_estimator_,
            self.x_test,
            self.y_test,
            ax=axis,
            alpha=0.8)
        plot_roc_curve(log_cl, self.x_test, self.y_test, ax=axis, alpha=0.8)
        plt.savefig(COMPARISON_MODEL)

        # saving models for future use
        joblib.dump(grid_rfc.best_estimator_, RANDOM_FOREST_PATH)
        joblib.dump(log_cl, LOGISTIC_REGRESSION_PATH)

    def classification_report_image(self):
        '''
        produces classification report for training and testing results
        and stores report as image
        in images folder
        '''
        # creates reports for randomforest reports and save
        create_reports(self.y_train, self.rf_ytrain_pred, self.y_test,
                       self.rf_ytest_pred, CLF_RANDOMFOREST_TEST, CLF_RANDOMFOREST_TRAIN)
        # Creates reports for logistic regression and save
        create_reports(
            self.y_train,
            self.lf_ytrain_pred,
            self.y_test,
            self.lf_ytest_pred,
            CLF_LOGREG_TEST,
            CLF_LOGREG_TRAIN)

    def feature_importance_plot(self, pth):
        '''
        creates and stores the feature importances in pth
        input:
             model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                 None
        '''
        grid_rfc = joblib.load(RANDOM_FOREST_PATH)
        importances = grid_rfc.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        names = [self.data.columns[i] for i in indices]
        # Create plot
        plt.figure(figsize=(20, 15))
        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        # Add bars
        plt.bar(range(self.data.shape[1]), importances[indices])
        # Add feature names as x-axis labels
        plt.xticks(range(self.data.shape[1]), names, rotation=90)
        plt.savefig(pth)


