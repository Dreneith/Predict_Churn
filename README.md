# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project we are trying to identify credit card customers who are likely to churn. 
using python package to write project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).




## Files and data description

  ### There are 4 main Folders
> data: contains Credit card customers data from [kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code)

> images: contains 3 subfolder. eda, reports and result. The eda folder contains exploratory 
data visualizations report. the reports folder contains model roc curves. The result folder contains model evaluation reports and Feature importance reports.

> logs: contains function test on data_transform and feature_train module.

> models: contains model details in .pkl file

 ### There are Five main modules
> Constants.py: contains variable  constants to save file paths

> data_transform.py: contains class methods that perform eda explorations and standardizing

> feature_train.py: contains class methods to create Training Evaluation data, Modelling using Random Forest and logistic regression Architecture.

> functions.py: contains helper functions to help in class methods

> churn_script_logging_and_tests.py: contains class modules that test and logs functions in methods in DataTransform and FeatureEvaluation classes.


## Running Files

### Install dependencies 
'''
python -m pip install -r project_requirements.txt
'''
### To run the entire workflow
'''
python churn_library.py
'''


