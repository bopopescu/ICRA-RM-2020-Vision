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

tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')


def load_trt_pb(pb_path):

	trt_graph_def = tf.GraphDef()

	with tf.gfile.GFile(pb_path, 'rb') as pf:
		trt_graph_def.ParseFromString(pf.read())

	for node in trt_graph_def.node:
		node.device = '/device:GPU:0'

	with tf.Graph().as_default() as trt_graph:
		tf.import_graph_def(trt_graph_def, name='')
	return trt_graph




def create_tfsess(trt_graph):	
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	tf_config.allow_soft_placement=True
	tf_config.log_device_placement=True
	
	tf_sess = tf.Session(config=tf_config, graph=trt_graph)
	
	return tf_sess





def preprocess(src, shape=None, to_rgb=True):

	img = src.astype(np.uint8)

	if shape:
		img = cv2.resize(img, shape)
	if to_rgb:
		img = img[..., ::-1]
		
	return img




def postprocess(img, boxes, scores, classes, conf_thre):

	h, w, _ = img.shape
	out_box = boxes[0] * np.array([h, w, h, w])
	out_box = out_box.astype(np.int32)
	out_conf = scores[0]
	out_cls = classes[0].astype(np.int32)

	mask = np.where(out_conf >= conf_thre)
	roi_gray, box, armor_cls, is_enemy = boxprocess(img, out_box[mask], out_cls[mask])
	
	return roi_gray, box, armor_cls, is_enemy




def boxprocess(img, box_list, cls_list, FIX = 0.5):

	for box, armor_cls in zip(box_list, cls_list):
		
		box_width= box[3]-box[1]
		box_height = box[2]-box[0]
		l = max(box_width, box_height)
		box[0] = box[0] - int(FIX * l)
		box[1] = box[1] - int(FIX * l)
		box[2] = box[2] + int(FIX * l)
		box[3] = box[3] + int(FIX * l)
		box = np.maximum(box, 0)
		roi_img = img[box[0]:box[2],box[1]:box[3],]
		roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
		
		enemy_mask = cv2.inRange(roi_gray, lower_enmclr, upper_allclr)
		n_pix = np.sum(enemy_mask == 255)
		all_pix = float(enemy_mask.size)
		percentage = n_pix / all_pix
		
		if percentage > 0.15:
			return roi_gray, box, armor_cls, True
		else:
			continue
	
	return None, None, None, False




def detect(origimg, tf_sess, conf, od_type):

	
	img = preprocess(origimg, (300, 300))

	boxes_out, scores_out, classes_out = tf_sess.run(
		[tf_boxes, tf_scores, tf_classes],
		feed_dict={tf_input: img[None, ...]})
	
	roi_gray, box, cls, is_enemy = postprocess(origimg, boxes_out, scores_out, classes_out, conf)

	return (roi_gray, box, cls, is_enemy) 



