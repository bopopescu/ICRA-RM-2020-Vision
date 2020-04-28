#! /home/lyjslay/py3env/bin python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : pyarmor_ros_test.py
#   Author      : lyjsly
#   Created date: 2020-03-22
#   Description : KCF Tracker + Detector in multiprocess
#
#================================================================

import sys
import KCF
import time
import rospy
import multiprocessing
from ctypes import c_bool
from multiprocessing import Process, Value, Array
from cv_bridge import CvBridge, CvBridgeError

from shared_ram import *
from detector_utils import *
from gimctrl_utils import *
from sensor_msgs.msg import Image
from roborts_msgs.msg import GimbalAngle
from roborts_msgs.srv import FricWhl, ShootCmd



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




class Detector(Process):
	

	def __init__(self, name, detecting, tracking, initracker, boundingbox, is_enemy, flag, image_in):
		
		super().__init__()
		self.name = name 

		self.detecting   = detecting
		self.tracking    = tracking
		self.initracker  = initracker
		self.boundingbox = boundingbox
		self.is_enemy    = is_enemy
		self.flag        = flag
		self.image_in    = image_in

		self.cls_dict  = {YOUR OWN CLASS DICT}
		self.pb_path   = 'YOUR OWN PB FILE'
		self.od_type   = 'ssd'
		self.conf_th   = 0.8



	def run(self):
		
		trt_graph = load_trt_pb(self.pb_path)
		tf_sess = create_tfsess(trt_graph)
		
		while True:
			
			img = self.image_in[:].copy()
			roi_gray, box, cls, is_enemy = detect(img, tf_sess, self.conf_th, self.od_type)
			
			if is_enemy:
				self.boundingbox[:] = [box[1], box[0], box[3]-box[1], box[2]-box[0]] # xmin,ymin,width,height

				self.detecting.value = False
				self.initracker.value = True
				self.tracking.value = False
				self.xyz_ang[:] = calc_xyz_ang(roi_gray, box)
				self.is_enemy.value = is_enemy
			else:
				self.boundingbox[:] = None
				rospy.loginfo('no enemy detected')
				continue





class Tracker(Process):
	
	def __init__(self, name, detecting, tracking, initracker, boundingbox, is_enemy, flag, image_in):
		
		super().__init__()
		self.name = name 
	
		self.detecting   = detecting
		self.tracking    = tracking
		self.initracker  = initracker
		self.boundingbox = boundingbox
		self.is_enemy    = is_enemy
		self.flag        = flag
		self.image_in    = image_in



	def run(self):

		tracker = KCF.kcftracker(False, False, False, False)
		
		while True:
			
			if self.detecting.value is True:
				continue

			elif self.initracker.value is True:
				print('initing')
				frame = self.image_in[:].copy()
				tracker.init(self.boundingbox[:], frame)
				self.detecting.value = False
				self.initracker.value = False
				self.tracking.value = True

			elif self.tracking.value is True:
				#print('tracking')
				frame = self.image_in[:].copy()
				tracker_box = tracker.update(frame)
				tracker_box = list(map(int, tracker_box)) 
				box = [tracker_box[1], tracker_box[0], tracker_box[1]+tracker_box[3], tracker_box[0]+tracker_box[2]] 
				roi_gray = cv2.cvtColor(frame[box[0]:box[2],box[1]:box[3],], cv2.COLOR_BGR2HSV)
				self.xyz_ang[:] = calc_xyz_ang(roi_gray, box)
				self.flag.value += 1
				
				if self.flag.value > 20:
					self.flag.value = 0
					self.detecting.value = True
					self.initracker.value = False
					self.tracking.value = False





class ArmorDetectionNode():
	
	def __init__(self):
		
		rospy.init_node('armor_detection_node')
		
		self.bridge = CvBridge()
		
		self._ctrlinfo_pub = rospy.Publisher('cmd_gimbal_angle', GimbalAngle, queue_size=1, tcp_nodelay=True)
		#self._decision_pub = rospy.Publisher('topic_name',MSGNAME,queue_size=1, tcp_nodelay=True)
		self._image_sub = rospy.Subscriber('camera/image', Image, self._update_images, tcp_nodelay=True)
		#self._ctrlpower_sub = rospy.Subscriber('topic_name',MSGNAME, self._update_ctrlpower, tcp_nodelay=True )
		
		self._fricwhl_client = rospy.ServiceProxy("cmd_fric_wheel",FricWhl)
		self._shoot_client = rospy.ServiceProxy("cmd_shoot",ShootCmd)
		
		self._can_ctrl = True
		undet_count = 40
		#num_bullets = 100
		
		while not rospy.is_shutdown():
			if self._can_ctrl:
				if is_enemy:
					self._set_fricwhl(True)
					self.ctrlinfo_pub.publish(xyz_ang[3],xyz_ang[4])
					self._shoot(1,1)
					rospy.loginfo('pub angle: pitch '+str(xyz_ang[3])+' yaw '+str(xyz_ang[4]))
					
				elif undet_count != 0:
					self._set_fricwhl(False)
					self._shoot(0,0)
					undet_count -= 1
					self.ctrlinfo_pub.publish(xyz_ang[3],xyz_ang[4])
					rospy.loginfo('pub angle: pitch '+str(xyz_ang[3])+' yaw '+str(xyz_ang[4]))
					
				else:
					self._set_fricwhl(False)
					self._shoot(0,0)
					searching_mode()
					rospy.loginfo('searching')
			else:
				rospy.loginfo('decision node needs to control the gimbal')
				
			#rospy.sleep(0.01)


	def _update_images(self, img_msg):
	
		cv_image = self.bridge.imgmsg_to_cv2(img_msg,"bgr8")
		image_in[:] = cv_image.copy()


	def _update_ctrlpower(self, ctrlpower_msg):
	
		self.can_ctrl = ctrlpower_msg.data


	def _set_fricwhl(can_start):

		rospy.wait_for_service("cmd_fric_wheel")
		try:
			# pull request to server, request components is 'open', value is 'can_start'
			# which is defined in FricWhl.srv
			resp = fricwhl_client.call(can_start)
			#rospy.loginfo("Message From fricwheelserver:%s"%resp.received)
		except rospy.ServiceException:
			rospy.logwarn("Service call failed")


	def _shoot(shoot_mode, shoot_number):

		try:
			resp = shoot_client.call(shoot_mode, shoot_number)
			#rospy.loginfo("Message From shootserver:%s"%resp.received)
		except rospy.ServiceException:
			rospy.logwarn("Service call failed")






if __name__ == '__main__':

	#multiprocessing.set_start_method('spawn')

	detecting   = Value(c_bool, True)
	initracker  = Value(c_bool, False)
	tracking    = Value(c_bool, False)
	is_enemy    = Value(c_bool, False)
	flag        = Value('I', 0)  
	boundingbox = Array('I', [0, 0, 0, 0]) 
	xyz_ang     = Array('f',[0.0, 0.0, 0.0, 0.0, 0.1]) 
	image_in    = shared_ram.empty(800*600*3, 'uint8')
	
	detector = Detector('detector')
	tracker  = Tracker ('tracker')
	detector.start()
	tracker.start()

	armor_detection_node = ArmorDetectionNode()

