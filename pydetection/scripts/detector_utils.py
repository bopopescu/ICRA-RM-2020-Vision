# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : detector_utils.py
#   Author      : lyjsly
#   Created date: 2020-04-26
#   Description : functions for detector
#
#================================================================
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

# define input tensor, should not defined in detect() because detect is in loop
tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')


def load_trt_pb(pb_path):
	'''
	@param 
		pb_path: path to weight file(.pb file)
	@function
		read from the .pb file and creat tensorRT graph
	@return
		tensorRT graph
	'''
	
	# defalut graph
	trt_graph_def = tf.GraphDef()

	with tf.gfile.GFile(pb_path, 'rb') as pf:
		trt_graph_def.ParseFromString(pf.read())

	# put NMS in cpu to reduce gpu, if ssd, only nms
	for node in trt_graph_def.node:
		if 'rfcn_' in pb_path and 'SecondStage' in node.name:
			node.device = '/device:GPU:0'
		if 'faster_rcnn_' in pb_path and 'SecondStage' in node.name:
			node.device = '/device:GPU:0'
		if 'NonMaxSuppression' in node.name:
			node.device = '/device:CPU:0'

	with tf.Graph().as_default() as trt_graph:
		tf.import_graph_def(trt_graph_def, name='')
	return trt_graph




def create_tfsess(trt_graph):
	
	tf_config = tf.ConfigProto()
	# some settings about gpu
	tf_config.gpu_options.allow_growth = True
	tf_config.allow_soft_placement=True
	tf_config.log_device_placement=True
	
	tf_sess = tf.Session(config=tf_config, graph=trt_graph)
	
	return tf_sess




# process image
def preprocess(src, shape=None, to_rgb=True):
	'''
	@param 
		src: image from camera
		shape: the size of the network input
		to_rgb: opencv(BGR) to network need(RGB)
	@function
		preprocess the image from camera to fit the network
	@return
		image for network input
	'''

	img = src.astype(np.uint8)

	#resize img for inputs:fstrcnn:1024*576 ssd:300*300 
	if shape:
		img = cv2.resize(img, shape)
	if to_rgb:
		# BGR2RGB
		img = img[..., ::-1]
	return img



# process the output 
def postprocess(img, boxes, scores, classes, conf_thre):
	'''
	@param 
		img: image from camera
		boxes: all boxes from detect
		scores: all scores from detect
		classes: all classes from detect
		conf_thre: threshold you want
	@function
		process the output and make it standard
	@return
		armor infomation
	'''
	h, w, _ = img.shape
	out_box = boxes[0] * np.array([h, w, h, w])
	out_box = out_box.astype(np.int32)
	out_conf = scores[0]
	out_cls = classes[0].astype(np.int32)

	mask = np.where(out_conf >= conf_thre)
	roi_gray, box, armor_cls, is_enemy = boxprocess(img, out_box[mask], out_cls[mask])
	
	return roi_gray, box, armor_cls, is_enemy
	#return (out_box[mask], out_conf[mask], out_cls[mask])



def boxprocess(img, box_list, cls_list, FIX = 0.5):
	'''
	@param 
		img: image from camera
		boxe_list: all boxes from detect
		class_list: all classes from detect
		FIX: boundingbox correction factor
	@function
		process precision loss and judge whether enemy
	@return
		preciser armor infomation
	'''
	# 'for' cycle go through box, once encounter enenmy box, return True 
	# And this box must be the most confident.
	# else if no enenmy box or box_list is none, return None and False
	#print(box_list)
	for box, armor_cls in zip(box_list, cls_list):
		
		# uncertain value(because of trt)
		box_width= box[3]-box[1] # xmax-xmin
		box_height = box[2]-box[0] # ymax-ymin
		l = max(box_width, box_height)
		
		# fix and get new
		box[0] = box[0] - int(FIX * l)
		box[1] = box[1] - int(FIX * l)
		box[2] = box[2] + int(FIX * l)
		box[3] = box[3] + int(FIX * l)
		
		# to avoid negetive number that roi will be enmpty
		box = np.maximum(box, 0)
		#print(box)
		
		# get roi and process it
		roi_img = img[box[0]:box[2],box[1]:box[3],]
		#cv2.imshow('roi',roi_img)
		#cv2.imwrite('roi.jpg', roi_img)
		roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
		
		# for color judgement
		enemy_mask = cv2.inRange(roi_gray, lower_enmclr, upper_allclr)
		
		# num of blue pixel
		n_pix = np.sum(enemy_mask == 255)
		all_pix = float(enemy_mask.size)
		
		# enemy color pixel's percentage in roi
		percentage = n_pix / all_pix
		#print(percentage)
		
		# once found, return immediately, must be the most conf enemy box
		if percentage > 0.15:
			return roi_gray, box, armor_cls, True
		else:
			continue
	
	return None, None, None, False






def detect(origimg, tf_sess, conf, od_type):
	'''
	@param 
		origimg: image from camera
		tf_sess: tensorflow session
		conf: threshold you want
		od_type: algorithm type
	@function
		detect the object
	@return
		armor infomation
	'''
	# define inputs and outputs tensor(better defined as global)
	#tf_num = tf_sess.graph.get_tensor_by_name('num_detections:0')

	# process img to fit the model
	
	img = preprocess(origimg, (300, 300))

	# detect and get outputs
	boxes_out, scores_out, classes_out = tf_sess.run(
		[tf_boxes, tf_scores, tf_classes],
		feed_dict={tf_input: img[None, ...]})
	
	# process outputs
	roi_gray, box, cls, is_enemy = postprocess(origimg, boxes_out, scores_out, classes_out, conf)

	return (roi_gray, box, cls, is_enemy) 



