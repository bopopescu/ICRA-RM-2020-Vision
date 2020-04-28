#! /home/lyjslay/py3env/bin python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : detect_ctrl.py
#   Author      : lyjsly
#   Created date: 2020-03-22
#   Description : only detector and ctrl task
#
#================================================================
import time
import logging
import numpy as np
import cv2
import tensorflow as tf
import rospy
import gxipy as gx
#from cv_bridge import CvBridge, CvBridgeError
from roborts_msgs.msg import GimbalAngle
from roborts_msgs.msg import PyArmorInfo
from roborts_msgs.srv import FricWhl, ShootCmd
#from sensor_msgs.msg import Image
import tensorflow.contrib.tensorrt as trt
# from utils.visualization import BBoxVisualization
from object_detection.utils import label_map_util
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph




pb_path = ''
DEFAULT_LABELMAP = ''  
conf_th = 0.5
od_type = 'ssd'
enemy_color = 'blue'

if enemy_color == 'blue':
	lower_enmclr = BLUE_LOW_THRESH
	lower_pnpclr = BLUE_LOW_THRESH
	upper_allclr = BLUE_HIGH_THRESH
if enemy_color == 'red':
	lower_enmclr = RED_LOW_THRESH
	lower_pnpclr = RED_LOW_THRESH
	upper_allclr = RED_HIGH_THRESH
	


camera_matrix = CAMERA_MATRIX
dist_coefs = DIST_MATRIX


object_3d_points = ARMOR_POINT_COORD





def load_trt_pb(pb_path):

    trt_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as pf:
        trt_graph_def.ParseFromString(pf.read())
    for node in trt_graph_def.node:
        node.device = '/device:GPU:0'
    with tf.Graph().as_default() as trt_graph:
        tf.import_graph_def(trt_graph_def, name='')
    return trt_graph




def preprocess(src, shape=None, to_rgb=True):

    img = src.astype(np.uint8)
    
    #resize, fstrcnn:1024*576 ssd:300*300 
    if shape:
        img = cv2.resize(img, shape)
    if to_rgb:
        # BGR2RGB
        img = img[..., ::-1]
    return img



def postprocess(img, boxes, scores, classes, conf_thre):

	h, w, _ = img.shape
	out_box = boxes[0] * np.array([h, w, h, w])
	out_box = out_box.astype(np.int32)
	out_conf = scores[0]
	out_cls = classes[0].astype(np.int32)

	mask = np.where(out_conf >= conf_thre)
	roi, box, armor_cls, is_enemy = boxprocess(img, out_box[mask], out_cls[mask])
	
	return roi, box, armor_cls, is_enemy
	#return (out_box[mask], out_conf[mask], out_cls[mask])





def detect(origimg, tf_sess, conf, od_type):
	#MEASURE_MODEL_TIME = False
	# contount time
	#avg_time = 0.0

	# define inputs and outputs tensor
	
	#tf_num = tf_sess.graph.get_tensor_by_name('num_detections:0')

	#input must be RGB, not BGR
	img = preprocess(origimg, (300, 300))
	
	# detect and get outputs
	boxes_out, scores_out, classes_out = tf_sess.run(
		[tf_boxes, tf_scores, tf_classes],
		feed_dict={tf_input: img[None, ...]})

	# process outputs
	roi_gray, box, cls, is_enemy = postprocess(origimg, boxes_out, scores_out, classes_out, conf)

	return (roi_gray, box, cls, is_enemy)


def boxprocess(img, box_list, cls_list, FIX = 0.5):
	
	# for cycle go through box, once encounter enenmy box, return True and this box, and this box must be most confident
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
		box = np.maximum(box, 0)
		#print(box)
		# get roi and process it
		roi_img = img[box[0]:box[2],box[1]:box[3],]
		#cv2.imshow('roi',roi_img)
		#cv2.imwrite('roi.jpg', roi_img)
		roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
		enemy_mask = cv2.inRange(roi_gray, lower_enmclr, upper_allclr)
		
		# num of blue pixel
		n_pix = np.sum(enemy_mask == 255)
		all_pix = float(enemy_mask.size)
		
		# blue pixel's percentage in roi
		percentage = n_pix / all_pix
		#print(percentage)
		# once find enemy box, return 
		if percentage > 0.15:
			return roi_gray, box, armor_cls, True
		else:
			continue
	
	return None, None, None, False



def get_img(cam):

	raw_frame = cam.data_stream[0].get_image()
	
	# raw img to numpy array (cv2 img)
	rgb_frame = raw_frame.convert('RGB')
	img_rgb = rgb_frame.get_numpy_array()
	
	img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
	
	return img



def detect_and_pub(img, tf_sess, conf_th, od_type, pub, udetected_conut):

	roi_gray, box, cls, is_enemy = detect(img, tf_sess, conf_th, od_type=od_type)
	
	if is_enemy:
		pitch, yaw = calc_xyz(roi_gray, box)
		#set_fricwhl(True)
		#yaw = yaw * (0.1 * np.exp(-np.square(yaw)/0.2048)/0.8021 + 0.08)
		pub.publish(True, True, yaw*0.5, pitch*0.7)
		#pub_xyz.publish(x, y, z, 2, True)
		rospy.loginfo(str(yaw)+str(pitch))
		#rospy.loginfo(str(x)+str('  ')+str(z))
		udetected_conut = 70
		shoot(1,1) # mod: shoot once, num: shoot one
		#pub.publish(float(x), float(y), float(z), int(cls), 1)
		#rospy.loginfo(str(x)+' '+str(y)+' '+str(z)+' '+str(cls)+' '+str(is_enemy))
	elif udetected_conut !=0: 
		# udetected, delay for a while to stay at this position, then init
		#pub_xyz.publish(0,0,0,0, False)
		pub.publish(True, True, 0, 0)
		udetected_conut -= 1
		#set_fricwhl(False)
		shoot(0,0)
	else:
		#pub.publish(200.0,100.0,1500.0,1,True)
		#pub_xyz.publish(0,0,0,0, False)
		pub.publish(False, False, 0, 0)
		rospy.loginfo('no enemy detected')
		#set_fricwhl(False)
		shoot(0,0)
		


def calc_xyz(roi_gray, box):
	
	#frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	pnp_mask = cv2.inRange(roi_gray, lower_pnpclr, upper_allclr)

	# find connected region and get its xmin ymin height width area
	nlabels, _, stats, _ = cv2.connectedComponentsWithStats(pnp_mask)
	#print(nlabels)

	if nlabels >= 3: # 2 lightbar in sights

		# 2 max area is lightbar(except the whole fig)
		stats = stats[stats[:,4].argsort()][-3:-1]

		# sort according to xmin in order to adapt to p1p2p3p4
		stats = stats[stats[:,0].argsort()]

		#print(stats)
		#if nlabels == 3: #including backgroud
		#[box_xmin+roi_xmin,box_ymin+roi_ymin]

		p1 = [box[1] + stats[0][0], box[0] + stats[0][1]] # left top
		p2 = [box[1] + stats[0][0] + stats[0][2], box[0] + stats[0][1] + stats[0][3]] #left bottom
		p3 = [box[1] + stats[1][0], box[0] + stats[1][1]] # right top
		p4 = [box[1] + stats[1][0] + stats[1][2], box[0] + stats[1][1] + stats[1][3]] # right bottom
		#else:
			#print('no enough ponit'+str(nlabels))
			#return float(0.0), float(0.0),float(0.0)

		object_2d_point = np.array((p1, p2, p3, p4), dtype=np.double)
		# calc tvec to get xyz
		_, _, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs, cv2.SOLVEPNP_EPNP)
		pitch = float(np.arctan2(tvec[1][0]+100, tvec[2][0])) 
		yaw = -float(np.arctan2(tvec[0][0]-60, tvec[2][0]))
		#return tvec[0][0], tvec[1][0], tvec[2][0]
		return pitch, yaw
	else:
		return 0, 0



def set_fricwhl(can_start):
	
	rospy.wait_for_service("cmd_fric_wheel")
	try:
		# define service, service name is 'cmd_fric_wheel', service class is FricWhl
		fricwhl_client = rospy.ServiceProxy("cmd_fric_wheel",FricWhl)

		# pull request to server, request components is 'open', value is 'can_start'
		# which is defined in FricWhl.srv
		resp = fricwhl_client.call(can_start)
		#rospy.loginfo("Message From fricwheelserver:%s"%resp.received)
	except rospy.ServiceException:
		rospy.logwarn("Service call failed")



def shoot(shoot_mode, shoot_number):
	
	rospy.wait_for_service("cmd_shoot")
	try:
		shoot_client = rospy.ServiceProxy("cmd_shoot",ShootCmd)
		resp = shoot_client.call(shoot_mode, shoot_number)
		#rospy.loginfo("Message From shootserver:%s"%resp.received)
	except rospy.ServiceException:
		rospy.logwarn("Service call failed")
	



if __name__ == '__main__':
	rospy.init_node('PyDetectorNode', anonymous=True)
	rospy.loginfo("inti ok!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	#pub_xyz = rospy.Publisher('PyArmorInfo', PyArmorInfo, queue_size=1)
	pub = rospy.Publisher('cmd_gimbal_angle', GimbalAngle, queue_size=1)
	#rate = rospy.Rate(30) # 30hz
	set_fricwhl(True)
	rospy.loginfo('reading label map')
	cls_dict = read_label_map(DEFAULT_LABELMAP)


	rospy.loginfo('loading TRT graph from pb: %s' % pb_path)
	trt_graph = load_trt_pb(pb_path)

	# creat TFSession, set gpu param to avoid CUDA_OUTOFF_MEMORY
	rospy.loginfo('starting up TensorFlow session')
	tf_config = tf.ConfigProto()
	tf_config.gpu_options.allow_growth = True
	tf_config.allow_soft_placement=True
	tf_config.log_device_placement=True
	tf_sess = tf.Session(config=tf_config, graph=trt_graph)


	tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
	tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
	tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
	tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')

	
	device_manager = gx.DeviceManager()
	dev_num, dev_info_list = device_manager.update_device_list()
	if dev_num == 0:
		sys.exit(1)
	# open the device by sn code , and set relative params
	str_sn = dev_info_list[0].get("sn")
	cam = device_manager.open_device_by_sn(str_sn)
	cam.ExposureTime.set(3000) 
	cam.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)
	# start
	cam.stream_on()


	# warm up trt
	rospy.loginfo('warming up the TRT graph with a dummy image')
	dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
	_, _, _, _ = detect(dummy_img, tf_sess, conf_th, od_type=od_type)


	# start to loop
	rospy.loginfo('starting to subscribe camera and detect and pub')

	# detect, pub ctrl and shoot info
	udetected_conut = 10
	while not rospy.is_shutdown():
		try:
			img = get_img(cam)
			if img is None:
				continue
			rospy.loginfo('get img')
			detect_and_pub(img, tf_sess, conf_th, 'ssd', pub, udetected_conut)
		except RuntimeError:
			break

	set_fricwhl(False)
	cam.stream_off()
	cam.close_device()
	rospy.loginfo('cleaning up')

    
