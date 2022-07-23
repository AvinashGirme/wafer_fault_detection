
from wsgiref import simple_server
from flask import Flask,request,render_template
from flask import Response
import flask_monitoringdashboard as dashboard
from training_Validation_Insertion import train_validation
from trainingModel import trainModel
from prediction_Validation_Insertion import pred_validation
from predictFromModel import prediction
import os
from flask_cors import CORS,cross_origin
import json

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

@app.route("/",methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')





@app.route("/predict",methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val=pred_validation(path)
            pred_val.prediction_validation()

            pred=prediction(path)

            path,json_predictions=pred.predictionFromModel()
            return Response("Prediction File created at !!!"+str(path)+ 'and few of the predictions are '+str(json.loads(json_predictions)))
        elif request.form is not None:
            path=request.form['filepath']
            pred_val=pred_validation(path)
            pred_val.prediction_validation()
            
            pred=prediction(path)
            path,json_predictions=pred.predictionFromModel()
            return Response("Prediction File created at !!!"+str(path)+' and few of the predictions are '+
                            str(json.loads(json_predictions)))
        
        
        else:
            print('Nothing Matched')
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s"%e)


@app.route("/train",methods=['GET', 'POST'])
@cross_origin()
def trainRouteClient():
    try:
        folder_path="Training_Batch_Files"

        if folder_path is not None:
            path=folder_path

            train_valObj=train_validation(path)
            train_valObj.train_validation()
            trainModelObj=trainModel()
            trainModelObj.trainingModel()

    except ValueError:
        return Response("Error Occured! %s"%ValueError)

    except KeyError:
        return Response("Error Occured! %s" %KeyError)

    except Exception as e:
        return Response("Error Occurred! %s"%e)
    return Response("Training Successful!!")

port = int(os.getenv("PORT",5000))
if __name__ == '__main__':
    host='0.0.0.0'
    httpd = simple_server.make_server(host,port,app)
    httpd.serve_forever()
