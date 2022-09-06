'''
data_transform module
Author: Akindele Abdulrasheed
Date: Aug 6th, 2022
'''
import os

from pandas.plotting import table
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import col_encoder_helper, distribution_plot, univariate_plot_save

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

class DataTransform():
    '''
    This class string performs dataframe creation,
    eda and data encodement
    '''

    def __init__(self):
        # Data processing
        self.data = None
        self.label = None

    def import_data(self, data_path):
        '''
    function: Import data and Creates a churn column
    input:
            pth: a path to the csv
         '''
        # Creating a dataframe for data
        self.data = pd.read_csv(data_path)
        self.data['Churn'] = self.data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        self.data.drop(columns=['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], inplace=True)
        return self.data

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        '''
        describe = self.data.describe()
        plt.figure(figsize=(20, 20))
        plot = plt.subplot(111, frame_on=False)
        # remove axis
        plot.xaxis.set_visible(False)
        plot.yaxis.set_visible(False)
        # create the table plot and position it in the middle
        table(plot, describe, loc='upper right')
        # save the plot as a png file
        plt.savefig('./images/eda/describe_plot.png')
        

        # Create and  save Univariate plot for columns Churn and Customer_Age
        univariate_plot_save(self.data, "Churn", "Churn_plot")
        univariate_plot_save(self.data, 'Customer_Age', "Customer_age_plot")

        # Bar chart plot on Marital_status value count
        univariate_plot_save(
            self.data,
            "Marital_Status",
            "Marital_Status_plot",
            value_count=True)
        distribution_plot(self.data, 'Total_Trans_Ct', 'Total_Trans_Ct_plot')

        # Creating a Corr plot and saving it in the eda folder
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.data.corr(), annot=False, cmap='plasma', linewidths=2)
        plt.savefig("./images/eda/Correlation_plot.png")

    def encoder_helper(self, response):
        '''
        Creates propotion of churn for each  categorical column in dataset

        input:
                response: string of response name for data label
'''
        # creating a list of categorical variable
        cat_columns = self.data.select_dtypes(
            include=['object']).columns.tolist()
        for col in cat_columns:
            col_encoder_helper(self.data, col)
        # label column
        self.label = self.data[response]
        self.data.drop(columns=[response], inplace=True)


# if __name__ == "__main__":
#     data_input = DataTransform()
#     data_input.import_data(r"./data/bank_data.csv")
#     data_input.perform_eda()
#     data_input.encoder_helper('Churn')
#     print(data_input.data.shape)
