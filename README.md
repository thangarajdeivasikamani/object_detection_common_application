# object_detection_common_application
This repo will help to predict the object with in image using various framework like TF2 ,Detectron and Yolo5 and also user can choose the model they need to experiment

How to use?
Fork or clone the repo.
1.Create the separate conda or virtual envoriment.
2.pip install -r requirements.txt
3. Run 
python clientApp.py

Note:
=====
If already model copied into respective framework model folder it won't download. Otherwise first time detection the model will download.


TF2 Models Prediction:
======================

1.EfficientDet D1 640x640:

![image](https://user-images.githubusercontent.com/46878296/164759399-06afea00-a1e3-4b29-9704-7e83aff58361.png)



Terminal Output:
No_of_objects_detected_by_mode: 3

Time Taken to execute the model 86.15892791748047

2.SSD MobileNet v2 320x320:

![image](https://user-images.githubusercontent.com/46878296/164757670-ec0364b8-4fe7-4506-a6ca-87fd154aa94c.png)

Terminal Output:
No_of_objects_detected_by_mode: 8

Time Taken to execute the model 27.62358021736145

3.SSD MobileNet V2 FPNLite 320x320:

![image](https://user-images.githubusercontent.com/46878296/164757303-3199f50c-e6a1-42b6-b9bd-b22a33c48c81.png)

Terminal Output:

No_of_objects_detected_by_mode: 5

Time Taken to execute the model 27.080549240112305

4.Faster R-CNN ResNet50 V1 640x640:

![image](https://user-images.githubusercontent.com/46878296/164757107-2878826d-c2b9-4150-b755-d98740796b82.png)

Terminal Output:
No_of_objects_detected_by_mode: 8

Time Taken to execute the model 78.90951347351074

5.Faster R-CNN ResNet101 V1 640x640:

![image](https://user-images.githubusercontent.com/46878296/164756209-6f7d3d2e-862a-4086-8154-55f3052d2dbb.png)

Terminal Output:

No_of_objects_detected_by_mode: 9

Time Taken to execute the model 189.88386058807373

Detectron2 Models Prediction:
============================
1.R50-FPN:


![image](https://user-images.githubusercontent.com/46878296/164746614-c3355f00-0b76-44ba-b1db-48a101f04e83.png)

Terminal Output:
The no.of object detected by model are: 10

['bowl 99%', 'chair 98%', 'refrigerator 95%', 'sink 93%', 'oven 81%', 'bottle 66%', 'dining table 61%', 'vase 61%', 'cup 58%', 'bowl 57%']

Time Taken to execute the model 25.72547149658203

Help Required:

2.R50-C4:

Model Downloading into required directory, But the detection class are wrong:

![image](https://user-images.githubusercontent.com/46878296/164747149-813f1dfb-ab8f-4513-b00f-83199917381d.png)


3.R101-FPN
Model Downloading into required directory, But the detection class are wrong:
![image](https://user-images.githubusercontent.com/46878296/164753697-dc0b9b48-19bf-4451-bbe2-71d786a3bb8c.png)
