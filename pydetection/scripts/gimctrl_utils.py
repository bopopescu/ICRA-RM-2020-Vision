#! /home/lyjslay/py3env/bin python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   File name   : gimctrl_utils.py
#   Author      : lyjsly
#   Created date: 2020-03-22
#   Description : calculate the pitch and yaw angle functions
#
#================================================================
import cv2
import numpy

'''
def calc_xyz_ang(roi_gray, box):
	
	pnp_mask = cv2.inRange(roi_gray, lower_pnpclr, upper_allclr)
	nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(pnp_mask)
	stats = stats[stats[:,4].argsort()][-3:-1]
	stats = stats[stats[:,0].argsort()]
	p1 = [box[1] + stats[0][0], box[0] + stats[0][1]]
	p2 = [box[1] + stats[0][0] + stats[0][2], box[0] + stats[0][1] + stats[0][3]] 
	p3 = [box[1] + stats[1][0], box[0] + stats[1][1]]
	p4 = [box[1] + stats[1][0] + stats[1][2], box[0] + stats[1][1] + stats[1][3]] 
	object_2d_point = np.array((p1, p2, p3, p4), dtype=np.double)
	found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs, cv2.SOLVEPNP_EPNP)
	pitch = float(np.arctan2(tvec[1][0]+OFFSET_Y, tvec[2][0]+OFFSET_Z)) 
	yaw   = float(np.arctan2(tvec[0][0]+OFFSET_X, tvec[2][0]+OFFSET_Z)) 
	
	return tvec[0][0], tvec[1][0], tvec[2][0]
'''



def bullet_model(x, v, angle, K=0.026, GRAVITY=9.78):
	
	t = float((np.exp(K*x) - 1) / (K*v*np.cos(angle)))
	y = float(v*np.sin(angle)*t - GRAVITY*t*t / 2)
	
	return y



def get_pitch(x, y, v):
	
	y_temp = y
	
	for i in range(20):
		a = np.atan2(y_temp, x)
		y_actual = bullet_model(x, v, a)
		dy = y - y_actual
		y_temp = y_temp + dy
		if (fabsf(dy) < 0.001) 
			break		
	return a




