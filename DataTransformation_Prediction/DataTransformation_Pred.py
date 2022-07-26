from os import listdir
import pandas
from application_logging.logger import App_Logger

class dataTransformPredict:
    def __init__(self):
        self.goodDataPath="Prediction_Raw_Files_Validated/Good_Raw"
        self.logger=App_Logger()

    def replaceMissingWithNull(self):

        log_file=open("Prediction_Logs/dataTransformLog.txt",'a+')
        try:
           onlyfiles=[f for f in listdir(self.goodDataPath)]
           for file in onlyfiles:
               csv = pandas.read_csv(self.goodDataPath + "/" + file)
               csv.fillna('NULL', inplace=True)
               # #csv.update("'"+ csv['Wafer'] +"'")
               # csv.update(csv['Wafer'].astype(str))
               csv['Wafer'] = csv['Wafer'].str[6:]
               csv.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
               self.logger.log(log_file, " %s: File Transformed successfully!!" % file)

        except Exception as e:
            self.logger.log(log_file,"Data Transformation failed because::%s"%e)
            log_file.close()
        log_file.close()