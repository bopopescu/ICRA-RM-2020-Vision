#! /home/lyjslay/py3env/bin python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   File name   : pyarmor_ros_test.py
#   Author      : lyjsly
#   Created date: 2020-03-22
#   Description : KCF Tracker + Detector in multiprocess
#
#================================================================

# import libs
import sys
import KCF
import time
import rospy
import multiprocessing
from ctypes import c_bool
from multiprocessing import Process, Value, Array
from cv_bridge import CvBridge, CvBridgeError

# import self defined class and methods
from shared_ram import *
from detector_utils import *
from gimctrl_utils import *
from sensor_msgs.msg import Image
from roborts_msgs.msg import GimbalAngle
from roborts_msgs.srv import FricWhl, ShootCmd




##################################### TEST LOGS ######################################
'''
@@@@@@@@encounter one error:
	tracker_box[-1, 324, 267, 120]
	detecting
	detector_box[300, -10, 477, 236]
	initing
	OverflowError: value too large to convert to int
	Exception ignored in: 'cvt.pylist2cvrect'
	OverflowError: value too large to convert to int
	OpenCV Error: Assertion failed (ssize.width > 0 && ssize.height > 0) in resize, file /build/opencv-L2vuMj/opencv-3.2.0+dfsg/modules/imgproc/src/imgwarp.cpp, line 3492
	terminate called after throwing an instance of 'cv::Exception'
	  what():  /build/opencv-L2vuMj/opencv-3.2.0+dfsg/modules/imgproc/src/imgwarp.cpp:3492: error: (-215) ssize.width > 0 && ssize.height > 0 in function resize

analysis:
	the boundingbox by detector locate at the corner,so after resort_board_info() fix it ,the box may have negetive nums,cv2.rectangle cant draw it


@@@@@@@@encounter one error:
Traceback (most recent call last):
  File "/home/lyjslay/anaconda3/envs/tensorflow1/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
    self.run()
  File "pyarmor_test.py", line 91, in run
    detector_box_fix,x,y,z,is_enemy = resort_board_info(img, detector_box)
  File "pyarmor_test.py", line 337, in resort_board_info
    frame = cv2.cvtColor(roi_img,cv2.COLOR_BGR2HSV)
cv2.error: OpenCV(4.1.1) /io/opencv/modules/imgproc/src/color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'

analysis: 
	<1> camera get uncompleted image,should add  iamge's integrity verification
	<2> fix box has negetive number
'''





################################ global variables ###################################

# color and camera concerned param
enemy_color = 'blue'

# for enemy judgement and lightbar fliter
if enemy_color == 'blue':
	lower_enmclr = np.array([90,0,0])
	lower_pnpclr = np.array([90,20,150])
	upper_allclr = np.array([130,255,255])
if enemy_color == 'red':
	lower_enmclr = np.array([0,20,100])
	lower_pnpclr = np.array([0,50,150])
	upper_allclr = np.array([20,255,255])

# camera params that calculate by matlab
camera_matrix = np.array(([1750, 0, 356.3],
                         [0, 1756, 375.9],
                         [0, 0, 1.0]), dtype=np.double)
dist_coefs = np.array([0, 0, 0, 0, 0], dtype=np.double)

# create 3d coordinates at armor centre, axis which vertically crosses armor is 'z' axis
object_3d_points = np.array(([-72, -32, 0], #xmin ymin 左上角
		                    [-58, 32, 0],   #xmin ymax 左下角
		                    [58, -32, 0],    #xmax ymax 右shang角
		                    [72, 32, 0]), dtype=np.double)#xmax ymin 右xia







class Detector(Process):
	
	'''
	Detector Subprocess
	In this subprocess, the detector continously run to get armor(roi) box.
	Meanwhile, judge enemy and calc PNP. If is enemy, share the box in all 
	processes, and the box will be used in Tracker process to initialize the 
	tracker. Otherwise, continuously run and share box as None.
	About 35-40 ms per loop on Jetson TX2.
	'''

	def __init__(self, name, detecting, tracking, initracker, boundingbox, is_enemy, flag, image_in):
		
		super().__init__()
		self.name = name # process name

		# these variables are defined in main process and shared in all processes 
		self.detecting   = detecting
		self.tracking    = tracking
		self.initracker  = initracker
		self.boundingbox = boundingbox
		self.is_enemy    = is_enemy
		self.flag        = flag
		self.image_in    = image_in

		# inference concerned param
		# self defined class can't be initialized in __init__() for pickle, should be inti in run()
		self.cls_dict  = {1: 'front', 2: 'side', 3: 'back'}
		self.pb_path   = './data/ssd_inception_v2_coco_trt.pb'
		self.od_type   = 'ssd'
		self.conf_th   = 0.8



	def run(self):
		
		# self defined variables
		trt_graph = load_trt_pb(self.pb_path)
		tf_sess = create_tfsess(trt_graph)
		
		# continously run detect , if detect , process info and share box and shoot angle
		while True:
			img = self.image_in[:].copy()
			# cls is unused, if decision node need, add it to sharing variables.
			# And also, the box is fixed box.
			roi_gray, box, cls, is_enemy = detect(img, tf_sess, self.conf_th, self.od_type)
			if is_enemy:
				self.boundingbox[:] = [box[1], box[0], box[3]-box[1], box[2]-box[0]] # xmin,ymin,width,height
				# first start init_tracker.
				self.detecting.value = False
				self.initracker.value = True
				self.tracking.value = False
				# then calc the angle by detector box.
				self.xyz_ang[:] = calc_xyz_ang(roi_gray, box)
				self.is_enemy.value = is_enemy
			else:
				self.boundingbox[:] = None
				rospy.loginfo('no enemy detected')
				continue







class Tracker(Process):
	
	'''
	Tracker Subprocess
	In this subprocess, the KCF Tracker continiously run in 3 situations:
	<1> detecting: means the detector is running and no need for tracking
					so the tracker process will be blocked.
	<2> initracker: means the detector sucessfully detect enemy box for 
					tracker init, so use the shared box and images to init.
	<3> tracking: means the tracker has been initialized, so use the tracker 
					to update some frames(self defined), and calc PNP, then 
					run detector to restart(reinitialize) the tracker.
	'''
	
	def __init__(self, name, detecting, tracking, initracker, boundingbox, is_enemy, flag, image_in):
		
		super().__init__()
		self.name = name # process name
	
		# # these variables are defined in main process and shared in all processes
		self.detecting   = detecting
		self.tracking    = tracking
		self.initracker  = initracker
		self.boundingbox = boundingbox
		self.is_enemy    = is_enemy
		self.flag        = flag
		self.image_in    = image_in



	def run(self):
		# create kcftracker instance
		tracker = KCF.kcftracker(False, False, False, False)# hog, fixed_window, multiscale, lab
		
		# start tracker loop
		while True:

			# detecting is True means detector process is runing, so tracker process should block
			if self.detecting.value is True:
				#print('detecting')
				continue

            # initracker is True means successfully get boundingbox from detector
			elif self.initracker.value is True:
				print('initing')
				frame = self.image_in[:].copy()
				tracker.init(self.boundingbox[:], frame)
				self.detecting.value = False
				self.initracker.value = False
				self.tracking.value = True

			# start tracking
			elif self.tracking.value is True:
				print('tracking')
				frame = self.image_in[:].copy()
				
				# update next frame to get next box
				tracker_box = tracker.update(frame)  # frame had better be contiguous
				
				# format: [xmin,ymin,w,h]
				tracker_box = list(map(int, tracker_box)) 
				
				# transform format to [ymin,xmin,ymax,xmax]
				box = [tracker_box[1], tracker_box[0], tracker_box[1]+tracker_box[3], tracker_box[0]+tracker_box[2]] 
				
				# calc ctrl info
				roi_gray = cv2.cvtColor(frame[box[0]:box[2],box[1]:box[3],], cv2.COLOR_BGR2HSV)
				self.xyz_ang[:] = calc_xyz_ang(roi_gray, box)
				self.flag.value += 1
				
				# if track some frames, let detector run to fix the box
				if self.flag.value > 20:
					self.flag.value = 0
					self.detecting.value = True
					self.initracker.value = False
					self.tracking.value = False







class ArmorDetectionNode():
	
	'''
	Main Process ROS Node：ArmorDetectionNode 
	In this process, all ROS concerned instance will be created, such as
	publisher, sbscriber and seveice client, and continously subscribe topic
	to get images and publish messages that from 2 subprocesses. Meanwhile,
	some decisions are made when publish messages.
	'''
	
	def __init__(self):
		
		rospy.init_node('armor_detection_node')
		
		#self.target_boundingbox = target_boundingbox
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
		
		# main process loop, subscriber is a single thread so no need for rospy.spin()
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
				
			#rospy.sleep(0.005) # 200Hz frequency



	def _update_images(self, img_msg):
		'''
		image subscriber callback
		'''
		cv_image = self.bridge.imgmsg_to_cv2(img_msg,"bgr8")
		image_in[:] = cv_image.copy()



	def _update_ctrlpower(self, ctrlpower_msg):
		'''
		decision node callback
		'''
		self.can_ctrl = ctrlpower_msg.data



	def _set_fricwhl(can_start):
		'''
		fricwheel service client
		'''
		rospy.wait_for_service("cmd_fric_wheel")
		try:
			# pull request to server, request components is 'open', value is 'can_start'
			# which is defined in FricWhl.srv
			resp = fricwhl_client.call(can_start)
			#rospy.loginfo("Message From fricwheelserver:%s"%resp.received)
		except rospy.ServiceException:
			rospy.logwarn("Service call failed")



	def _shoot(shoot_mode, shoot_number):
		'''
		shoot service client
		'''
		try:
			resp = shoot_client.call(shoot_mode, shoot_number)
			#rospy.loginfo("Message From shootserver:%s"%resp.received)
		except rospy.ServiceException:
			rospy.logwarn("Service call failed")







# main process
if __name__ == '__main__':

	#multiprocessing.set_start_method('spawn')


	# control the sub processes run
	detecting   = Value(c_bool, True)
	initracker  = Value(c_bool, False)
	tracking    = Value(c_bool, False)
	flag        = Value('I', 0)  # num of tracked frames
	
	# ArmorInfo varibles shared by all process
	is_enemy    = Value(c_bool, False)
	boundingbox = Array('I', [0, 0, 0, 0]) # 'I' means usigned_int
	xyz_ang     = Array('f',[0.0, 0.0, 0.0, 0.0, 0.1]) # real x,y,z,pitch,yaw

	# init camera and image shared memory
	image_in    = shared_ram.empty(800*600*3, 'uint8')

	
	
	# create 2 sub processes instance and start them
	# detecting, initracker, tracking, boundingbox, is_enemy, flag, image_in
	# are global variables, no need for param passing
	detector = Detector('detector')
	tracker  = Tracker ('tracker')
	detector.start()
	tracker.start()

	# create ros node, and start main process loop
	armor_detection_node = ArmorDetectionNode()

