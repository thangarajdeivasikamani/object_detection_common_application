import cv2 as cv
import json
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from utils.download_models import detectron_download_model
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
import torch
import numpy as np
from PIL import Image
from utils.utils import encodeImageIntoBase64


class Dectron_Detector:

	def __init__(self,model,filename):
		self.filename = filename   
		if model == "R50-FPN":
			model_config_download_url = 'https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml'
			model_download_url = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl'
			model_config_filename = 'faster_rcnn_R_50_FPN_1x.yaml'
			model_name = 'model_final_b275ba.pkl'
			detection_model_dir,detection_model_config = detectron_download_model(model,model_config_download_url,model_download_url,model_config_filename,model_name)
			number_of_classes_of_model =80

		elif model =="R50-C4":
			model_config_download_url = 'https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/rpn_R_50_C4_1x.yaml'
			model_download_url = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/model_final_450694.pkl'
			model_config_filename = 'rpn_R_50_C4_1x.yaml'
			model_name = 'model_final_450694.pkl'
			detection_model_dir,detection_model_config = detectron_download_model(model,model_config_download_url,model_download_url,model_config_filename,model_name)
			number_of_classes_of_model =80
			
		elif model =="R101-C4":
			model_config_download_url = 'https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml'
			model_download_url = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl'
			model_config_filename = 'faster_rcnn_R_101_C4_3x.yaml'
			model_name = 'model_final_298dad.pkl'
			detection_model_dir,detection_model_config = detectron_download_model(model,model_config_download_url,model_download_url,model_config_filename,model_name)
			number_of_classes_of_model =80

		elif model =="R101-FPN":
			model_config_download_url = 'https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
			model_download_url = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl'
			model_config_filename = 'faster_rcnn_R_101_FPN_3x.yaml'
			model_name = 'model_final_f6e8b1.pkl'
			detection_model_dir,detection_model_config = detectron_download_model(model,model_config_download_url,model_download_url,model_config_filename,model_name)
			number_of_classes_of_model =80
		elif model =="X101-FPN":
			model_config_download_url = 'https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'
			model_download_url = 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl'
			model_config_filename = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
			model_name = 'model_final_68b088.pkl'
			detection_model_dir,detection_model_config = detectron_download_model(model,model_config_download_url,model_download_url,model_config_filename,model_name)
			number_of_classes_of_model = 80
		else:
			print("Please choose correct model")
		# set model and test set
		#self.model = os.getcwd() + '\\detectron\detectron_models\\'+model+'faster_rcnn_R_50_FPN_1x.yaml'
		self.model = detection_model_config           
		# obtain detectron2's default config
		self.cfg = get_cfg() 
		# load values from a file
		self.cfg.merge_from_file(os.getcwd() + "\\detectron\\"+"config.yml")
	
		#self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/"+self.model))

		# set device to cpu
		self.cfg.MODEL.DEVICE = "cpu"

		# get weights 
		# self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/"+self.model) 
		#self.cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
		self.cfg.MODEL.WEIGHTS = detection_model_dir

		# set the testing threshold for this model
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
		# set the class to resolve  the 'roi_heads.box_predictor.cls_score.weight'  warning
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_of_classes_of_model
		# build model from weights
		#self.cfg.MODEL.WEIGHTS = self.convert_model_for_inference()

	# build model and convert for inference
	def convert_model_for_inference(self):

		# build model
		model = build_model(self.cfg)

		# save as checkpoint
		torch.save(model.state_dict(), 'checkpoint.pth')

		# return path to inference model
		return 'checkpoint.pth'


	def inference(self, file):

		predictor = DefaultPredictor(self.cfg)
		im = cv.imread(file)
		outputs = predictor(im)
		#print(outputs["instances"].pred_classes)
		#print(outputs["instances"].pred_boxes)
		metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).set(thing_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
		
		# visualise
		v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		predicted_image = v.get_image()
		im_rgb = cv.cvtColor(predicted_image, cv.COLOR_RGB2BGR)
		cv.imwrite('color_img.jpg', im_rgb)
		# imagekeeper = []
		opencodedbase64 = encodeImageIntoBase64("color_img.jpg")
		# imagekeeper.append({"image": opencodedbase64.decode('utf-8')})
		result = {"image" : opencodedbase64.decode('utf-8') }
		return result




