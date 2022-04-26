
import os
import sys
import time
sys.path.insert(0,os.getcwd() + "\TF2")
sys.path.insert(0,os.getcwd() + "\detectron")
sys.path.insert(0,os.getcwd() + "\Yolo5")
sys.path.insert(0,os.getcwd() + "\\utils")
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from Helpers.utils import decodeImage
from TF2.detect import TF2Predictor
from detectron.detectron_object_detector import Dectron_Detector
from TF2.object_detection.utils import label_map_util
from Yolo5.Yolo5_detect import DetectorYolov5

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        framework= request.json['framework']
        print(framework)
        model = request.json['model']
        print(model)
        detection = request.json['detection']
        print(detection)
        decodeImage(image, clApp.filename)
        if framework == 'TF2':
            start = time.time()
            clApp.obj_detect = TF2Predictor(model)
            result = clApp.obj_detect.run_inference()
            end = time.time()
            time_taken =end-start
            print("Time Taken to execute the model",time_taken)
        elif framework == "Detectron2":
            start = time.time()           
            clApp.objectDetection = Dectron_Detector(clApp.filename,model)
            result = clApp.objectDetection.inference(clApp.filename)
            end = time.time()
            time_taken =end-start
            print("Time Taken to execute the model",time_taken)
        
        elif framework == "YOLO":
            
            start = time.time()           
            clApp.yolo_objectDetection = DetectorYolov5(clApp.filename,model)
            result = clApp.yolo_objectDetection.detect_action()
            end = time.time()
            time_taken =end-start
            print("Time Taken to execute the model",time_taken)
        else:
            print("Please choose the correct framework")
        
    except ValueError as val:
        print(val)
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    return jsonify(result)


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 8000
    app.run(host='127.0.0.1', port=port)
    #app.run(host='0.0.0.0', port=7000, debug=True)