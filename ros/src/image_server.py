import rospy
import message_filters
import cv2
import numpy as np
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from normals import gpu_normals
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from synthesizer.msg import PoseCNNMsg

class ImageServer:

    def __init__(self, sess, network, imdb, cfg):

        self.image_listener = ImageListener(sess, network, imdb, cfg)

        # initialize a node
        rospy.init_node("image_server")
        image_listener.callback()