
import pandas as pd

class Data_Getter:

    def __init__(self,file_object,logger_object):
        self.training_file='Training_FileFromDB/InputFile.csv'
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self):
        self.logger_object.log(self.file_object,'Enterd the get data method of Data_Getter class')
        try:
            self.data=pd.read_csv(self.training_file)
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get data method of the Data_Getter class')

            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occurred in get_data method of Data_Getter class. Exception message:'+str(e))
            self.logger_object.log(self.file_object,'Data Load Unsuccessful. Exited the get data_method of Data_Getter class')
            raise Exception()



