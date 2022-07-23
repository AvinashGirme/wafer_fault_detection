
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class Preprocessor:
    def __init__(self,file_object,logger_object):
        self.file_object=file_object
        self.logger_object=logger_object

    def remove_columns(self,data,columns):
        self.logger_object.log(self.file_object,'Entered the remove_columns method of Preprocessor class')
        self.data=data
        self.columns=columns

        try:
            self.useful_data=self.data.drop(labels=self.columns,axis=1)
            self.logger_object.log(self.file_object,
                                   'Column removal Successful,Exited the remove_column method of Preprocessor class')

            return self.useful_data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of Preprocessor Class. Exception message :'+str(e))
            self.logger_object.log(self.file_object,'Column removal Unsuccessful.Exited the remove_column method of Preprocessor class')
            raise Exception()

    def separate_label_feature(self,data,label_column_name):

        self.logger_object.log(self.file_object,'Entered the separate_label_feature of Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1)
            self.Y=data[label_column_name]

            self.logger_object.log(self.file_object,'Label Separation Successful.Exited the separate_label_feature method of Preprocessor class')
            return self.X,self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class')
            self.logger_object.log(self.file_object,'Label Separation Unsuccessful.Exited the separate_label_feature method of Preprocessor class')
            raise Exception()

    def is_null_present(self,data):
        self.logger_object.log(self.file_object,'Entered the is_null_present method of the Preprocessor class')
        self.null_present=False
        try:
            self.null_counts=data.isna().sum()
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            if(self.null_present):
                dataframe_with_null=pd.DataFrame()
                dataframe_with_null['column']=data.columns
                dataframe_with_null['missing_value_count']=np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv')

            self.logger_object.log(self.file_object,'Finding missing values is a success. Data written to the null values file.exited is_null_present method of preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of preprocessor class. Exception message : '+str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()


    def impute_missing_values(self,data):

        self.logger_object.log(self.file_object,'Entered the impute_missing_values method of preprocessor class')
        self.data=data

        try:
            imputer=KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data)
            self.new_data=pd.DataFrame(data=self.new_array,columns=data.columns)
            self.logger_object.log(self.file_object,'Impute missing values Successful. Exited the impute_missing_values method of preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of Preprocessor class. Exception Message :'+str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values methid of Preprocessor class')
            raise Exception()


    def get_columns_with_zero_std_deviation(self,data):
        self.logger_object.log(self.file_object,'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n=data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std']==0):
                    self.col_to_drop.append(x)

            self.logger_object.log(self.file_object,'Column search for Std Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of Preprocessor class')
            return self.col_to_drop
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of Preprocessor class. Exception Message : '+str(e))
            self.logger_object.log(self.file_object,'Column search for Std Deviation of zero failed. Exited the get_columns_with_zero_std_deviation method of Preprocessor class')
            raise Exception()


