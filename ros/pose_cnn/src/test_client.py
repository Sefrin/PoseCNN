#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from vision_msgs import *
from pose_cnn.srv import *

def callback(rgb, depth, info):
    rospy.wait_for_service('posecnn_recognize')
    try:
        rec = rospy.ServiceProxy('posecnn_recognize', recognize)
        resp1 = rec(rgb, depth, info)
        print resp1.detections
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == '__main__':
	# initialize a node
	rospy.init_node('test_client')
	# self.posecnn_pub = rospy.Publisher('posecnn_result', PoseCNNMsg, queue_size=1)
	# self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=1)
	rgb_sub = message_filters.Subscriber('/camera/rgb/image_rect_color', Image, queue_size=2)
	depth_sub = message_filters.Subscriber('/camera/depth/image_rect', Image, queue_size=2)
	camera_info_sub = message_filters.Subscriber('/camera/depth/camera_info', CameraInfo, queue_size=2)
	# depth_sub = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, queue_size=2)
	
	queue_size = 1
	slop_seconds = 0.05
	ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, camera_info_sub], queue_size, slop_seconds)
	ts.registerCallback(callback)
	rospy.spin()

