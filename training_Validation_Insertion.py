
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTransform_Training.DataTransformation import dataTransform
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from application_logging import logger
class train_validation:
    def __init__(self,path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform=dataTransform()
        self.dBOperation=dBOperation()
        self.file_object=open("Training_Logs/Training_Main_Log.txt",'a+')
        self.log_writer=logger.App_Logger()

    def train_validation(self):
        try:
            self.log_writer.log(self.file_object,'Start of Validation on files!!')
            LenghtOfDateStampFile,LengthOfTimeStampInFile,column_names,NumberofColumns=self.raw_data.valuesFromSchema()

            #getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            self.raw_data.validationFileNameRaw(regex,LenghtOfDateStampFile,LengthOfTimeStampInFile)

            self.raw_data.validateColumnLength(NumberofColumns)
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object,"Raw Data Validation Completed!!")

            self.log_writer.log(self.file_object,"Starting Data Transformation!!")
            self.dataTransform.replaceMissingWithNull()
            self.log_writer.log(self.file_object,"Data Transformation Completed!!!")

            self.log_writer.log(self.file_object,
                                "Creating Training Database and Tables on the basis of given Schema!!!")


            self.dBOperation.createTableDb('Training',column_names)
            self.log_writer.log(self.file_object,"Table Creation Completed!!")
            self.log_writer.log(self.file_object,"Insertion of Data into Table started!!!")

            self.dBOperation.insertIntoTableGoodData('Training')
            self.log_writer.log(self.file_object,"Insertion in Table Completed!!!")
            self.log_writer.log(self.file_object,"Deleting Good Data Folder")

            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object,"Good_Data folder deleted!!!")

            self.log_writer.log(self.file_object,"Moving bad files to Archive and deleting Bad_Data folder!!")

            self.raw_data.moveBadFilesToArchivBad()
            self.log_writer.log(self.file_object,"Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object,"Validation Operation Completed!!")
            self.log_writer.log(self.file_object,"Extracting csv file from table")

            self.dBOperation.selectingDatafromtableintocsv('Training')
            self.file_object.close()



        except Exception as e:
            raise e

