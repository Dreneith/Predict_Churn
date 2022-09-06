'''
Constant Module
Author: Akindele Abdulrasheed
Date: Aug 6th, 2022
'''

grid_params = {
    'n_estimators': [100, 50, 20],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 10, 20, 15],
    'criterion': ['gini', 'entropy']
}
#model Reports
RANDOM_FOREST_PATH = './models/rfc_model.pkl'
LOGISTIC_REGRESSION_PATH = './models/logistic_model.pkl'

#Image reports Path
OUTPUT_PATH = "./images/reports/feature_importance.png"
COMPARISON_MODEL = "./images/results/Comparison_Model.png"
PLT_LOGISTIC = './images/results/Logistic_Reg.png'
PLT_RANDOMFOREST = './images/results/RandomForest.png'

#Classification reports
CLF_RANDOMFOREST_TEST ="./images/reports/RandForestCLF_TEST.png"
CLF_RANDOMFOREST_TRAIN = "./images/reports/RandForestCLF_TRAIN.png"
CLF_LOGREG_TEST = "./images/reports/LogRegCLF_TEST.png"
CLF_LOGREG_TRAIN = "./images/reports/LogRegCLF_TRAIN.png"