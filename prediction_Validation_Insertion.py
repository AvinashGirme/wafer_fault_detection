from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import dBOperation
from DataTransformation_Prediction.DataTransformation_Pred import dataTransformPredict
from application_logging import logger

class pred_validation:
    def __init__(self,path):
        self.raw_data=Prediction_Data_validation(path)
        self.dBOperation=dBOperation()
        self.dataTransform=dataTransformPredict()
        self.file_object=open("Prediction_Logs/Prediction_Log.txt",'a+')
        self.log_writer=logger.App_Logger()

    def prediction_validation(self):
        try:
            self.log_writer.log(self.file_object,'Start oof Validation on files for prediction!!')
            LenghtOfDateStampFile,LengthOfTimeStampInFile,column_names,NumberofColumns=self.raw_data.valuesFromSchema()

            regex=self.raw_data.manualRegexCreation()
            self.raw_data.validationFileNameRaw(regex, LenghtOfDateStampFile, LengthOfTimeStampInFile)

            self.raw_data.validateColumnLength(NumberofColumns)
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, "Raw Data Validation Completed!!")

            self.log_writer.log(self.file_object, "Starting Data Transformation!!")
            self.dataTransform.replaceMissingWithNull()
            self.log_writer.log(self.file_object, "Data Transformation Completed!!!")

            self.log_writer.log(self.file_object,
                                "Creating Training Database and Tables on the basis of given Schema!!!")

            self.dBOperation.createTableDb('Prediction', column_names)
            self.log_writer.log(self.file_object, "Table Creation Completed!!")
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!!")

            self.dBOperation.insertIntoTableGoodData('Prediction')
            self.log_writer.log(self.file_object, "Insertion in Table Completed!!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder")

            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")

            self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!")

            self.raw_data.moveBadFilesToArchivBad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation Completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")

            self.dBOperation.selectingDatafromtableintocsv('Prediction')
            self.file_object.close()

        except Exception as e:
            raise e

