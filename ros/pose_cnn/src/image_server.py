import rospy
import cv2
import numpy as np
from pose_cnn.srv import *
from vision_msgs.msg import *
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from pose_cnn_msgs.msg import PoseCNNMsg
from pyquaternion import Quaternion
from dynamic_reconfigure.server import Server
class ImageServer:

    def __init__(self, sess, network, imdb, cfg):

        self.cv_bridge = CvBridge()
        self.sess = sess
        self.net = network
        self.imdb = imdb
        self.cfg = cfg
        # initialize a node
        rospy.init_node('posecnn_server')
        self.s = rospy.Service('posecnn_recognize', posecnn_recognize, self.rec)
        self.posecnn_pub = rospy.Publisher('posecnn_result', PoseCNNMsg, queue_size=1)
        self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=1)
        print "PoseCNN recognition ready."
        rospy.spin()

    def rec(self, req):
        K = np.array([[req.camera_info.P[0], req.camera_info.P[1], req.camera_info.P[2]],
                      [req.camera_info.P[4], req.camera_info.P[5], req.camera_info.P[6]],
                      [req.camera_info.P[8], req.camera_info.P[9], req.camera_info.P[10]]])

        self.meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 10000.0})
        if req.depth_image.encoding == '32FC1':
            depth_32 = self.cv_bridge.imgmsg_to_cv2(req.depth_image) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif req.depth_image.encoding == '16UC1':
            depth_cv = self.cv_bridge.imgmsg_to_cv2(req.depth_image)
        else:
            rospy.logerr_throttle(
                1, 'Unsupported req.depth_image type. Expected 16UC1 or 32FC1, got {}'.format(
                    req.depth_image.encoding))
            return

        # write images
        im = self.cv_bridge.imgmsg_to_cv2(req.rgb_image, 'bgr8')

        # resize so we dont run out of memory :(
        im_height, im_width, _ = im.shape
        if (im.shape[1]>1000):
            aspect = float(im_height) / float(im_width)
            processing_width = 640
            processing_height = processing_width * aspect
            im = cv2.resize(im, (processing_width, int(processing_height)))
            depth_cv = cv2.resize(depth_cv, (processing_width, int(processing_height)))
        else:
            processing_width = im_width
            processing_height = im_height
            print(depth_cv.shape)
        # filename = 'images/%06d-color.png' % self.count
        # cv2.imwrite(filename, im)

        # filename = 'images/%06d-depth.png' % self.count
        # cv2.imwrite(filename, depth_cv)
        # print filename
        # self.count += 1

        # run network
        labels, probs, vertex_pred, rois, poses = self.im_segment_single_frame(self.sess, self.net, im, depth_cv, self.meta_data, \
            self.imdb._extents, self.imdb._points_all, self.imdb._symmetry, self.imdb.num_classes)

        # resize back to original resolution
        im_label = self.imdb.labels_to_image(im, labels)
        im_label = cv2.resize(im_label, (im_width, im_height))

        # publish
        pose_cnn_msg = PoseCNNMsg()
        pose_cnn_msg.height = int(im.shape[0])
        pose_cnn_msg.width = int(im.shape[1])
        pose_cnn_msg.roi_num = int(rois.shape[0])
        pose_cnn_msg.roi_channel = int(rois.shape[1])
        pose_cnn_msg.fx = float(self.meta_data['intrinsic_matrix'][0, 0])
        pose_cnn_msg.fy = float(self.meta_data['intrinsic_matrix'][1, 1])
        pose_cnn_msg.px = float(self.meta_data['intrinsic_matrix'][0, 2])
        pose_cnn_msg.py = float(self.meta_data['intrinsic_matrix'][1, 2])
        pose_cnn_msg.factor = float(self.meta_data['factor_depth'])
        pose_cnn_msg.znear = float(0.25)
        pose_cnn_msg.zfar = float(6.0)
        pose_cnn_msg.label = self.cv_bridge.cv2_to_imgmsg(labels.astype(np.uint8), 'mono8')
        pose_cnn_msg.depth = self.cv_bridge.cv2_to_imgmsg(depth_cv, 'mono16')
        pose_cnn_msg.rois = rois.astype(np.float32).flatten().tolist()
        pose_cnn_msg.poses = poses.astype(np.float32).flatten().tolist()
        self.posecnn_pub.publish(pose_cnn_msg)

        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = req.rgb_image.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)
        # for row in label_msg:
        #     for p in row:
        #         if
        # for x in range(0, label_msg.height*label_msg.width):
        # print([[x for x in row if 255 not in x] for row in im_label])

        response = posecnn_recognizeResponse()
        msg = Detection3DArray()
        msg.header = req.rgb_image.header
        for i in range(int(rois.shape[0])):
            if (all(rois[i]== 0)):
                continue
            center_x = (rois[i, 2] + rois[i, 4]) / 2
            center_y = (rois[i, 3] + rois[i, 5]) / 2
            width = rois[i, 4] - rois[i, 2]
            height = rois[i, 5] - rois[i, 3]
            bbox = BoundingBox2D()
            bbox.center.x = center_x * im_width / processing_width
            bbox.center.y = center_y * im_height / processing_height
            bbox.size_x = width * im_width / processing_width
            bbox.size_y = height * im_height / processing_height

            response.bboxes.append(bbox)

            detection = Detection3D()
            detection.header = req.rgb_image.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = rois[i, 1]
            q = Quaternion (poses[i, :4])
            q_norm = q.normalised
            hyp.pose.pose.orientation.w = q_norm[0]
            hyp.pose.pose.orientation.x = q_norm[1]
            hyp.pose.pose.orientation.y = q_norm[2]
            hyp.pose.pose.orientation.z = q_norm[3]
            hyp.pose.pose.position.x = poses[i][4]
            hyp.pose.pose.position.y = poses[i][5]
            hyp.pose.pose.position.z = poses[i][6]
            detection.results.append(hyp)
            msg.detections.append(detection)
        response.detections = msg
        #resize raw labels to old resolution and append to msg
        labels = cv2.resize(labels.astype(np.uint8), (im_width, im_height))
        response.label_image_raw = self.cv_bridge.cv2_to_imgmsg(labels, 'mono8')
        response.label_image_color = label_msg
        return response


    def get_image_blob(self, im, im_depth, meta_data):
        """Converts an image into a network input.

        Arguments:
            im (ndarray): a color image in BGR order

        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
               in the image pyramid
        """

        # RGB
        im_orig = im.astype(np.float32, copy=True)
        # mask the color image according to depth
        if self.cfg.EXP_DIR == 'rgbd_scene':
            I = np.where(im_depth == 0)
            im_orig[I[0], I[1], :] = 0

        processed_ims_rescale = []
        im_scale = self.cfg.TEST.SCALES_BASE[0]
        im_rescale = cv2.resize(im_orig / 127.5 - 1, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_rescale.append(im_rescale)

        im_orig -= self.cfg.PIXEL_MEANS
        processed_ims = []
        im_scale_factors = []
        assert len(self.cfg.TEST.SCALES_BASE) == 1

        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

        # depth
        im_orig = im_depth.astype(np.float32, copy=True)
        # im_orig = im_orig / im_orig.max() * 255
        im_orig = np.clip(im_orig / 2000.0, 0, 1) * 255
        im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
        im_orig -= self.cfg.PIXEL_MEANS

        processed_ims_depth = []
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im)

        if cfg.INPUT == 'NORMAL':
            # meta data
            K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]

            # normals
            depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
            nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
            im_normal = 127.5 * nmap + 127.5
            im_normal = im_normal.astype(np.uint8)
            im_normal = im_normal[:, :, (2, 1, 0)]
            im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

            processed_ims_normal = []
            im_orig = im_normal.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_normal = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            processed_ims_normal.append(im_normal)
            blob_normal = im_list_to_blob(processed_ims_normal, 3)
        else:
            blob_normal = []

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims, 3)
        blob_rescale = im_list_to_blob(processed_ims_rescale, 3)
        blob_depth = im_list_to_blob(processed_ims_depth, 3)

        return blob, blob_rescale, blob_depth, blob_normal, np.array(im_scale_factors)


    def im_segment_single_frame(self, sess, net, im, im_depth, meta_data, extents, points, symmetry, num_classes):
        """segment image
        """

        # compute image blob
        im_blob, im_rescale_blob, im_depth_blob, im_normal_blob, im_scale_factors = self.get_image_blob(im, im_depth, meta_data)
        im_scale = im_scale_factors[0]

        # construct the meta data
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        # x = np.zeros(3)
        # print("check: ")
        # x[0]=0 * factor_x
        # x[1]=0*  factor_y
        # x[2]=0.5
        # print(np.matmul(K, x))
        # x[0]=320
        # x[1]=240
        # x[2]=0.5
        # print(np.matmul(Kinv,x))
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()
        # mdata[18:30] = pose_world2live.flatten()
        # mdata[30:42] = pose_live2world.flatten()
        meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
        meta_data_blob[0,0,0,:] = mdata

        # use a fake label blob of ones
        height = int(im_depth.shape[0] * im_scale)
        width = int(im_depth.shape[1] * im_scale)
        label_blob = np.ones((1, height, width), dtype=np.int32)

        pose_blob = np.zeros((1, 13), dtype=np.float32)
        vertex_target_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)
        vertex_weight_blob = np.zeros((1, height, width, 3*num_classes), dtype=np.float32)

        # forward pass
        if self.cfg.INPUT == 'RGBD':
            data_blob = im_blob
            data_p_blob = im_depth_blob
            print "RGBD"
        elif self.cfg.INPUT == 'COLOR':
            data_blob = im_blob
        elif self.cfg.INPUT == 'DEPTH':
            data_blob = im_depth_blob
        elif self.cfg.INPUT == 'NORMAL':
            data_blob = im_normal_blob

        if self.cfg.INPUT == 'RGBD':
            if self.cfg.TEST.VERTEX_REG_2D or self.cfg.TEST.VERTEX_REG_3D:
                feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                             net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                             net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
            else:
                feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}
        else:
            if self.cfg.TEST.VERTEX_REG_2D or self.cfg.TEST.VERTEX_REG_3D:
                feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0, \
                             net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                             net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
            else:
                feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: 1.0}

        sess.run(net.enqueue_op, feed_dict=feed_dict)

        if self.cfg.TEST.VERTEX_REG_2D:
            if self.cfg.TEST.POSE_REG:
                labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                              net.get_output('rois'), net.get_output('poses_init'), net.get_output('poses_tanh')])

                # non-maximum suppression
                # keep = nms(rois, 0.5)
                # rois = rois[keep, :]
                # poses_init = poses_init[keep, :]
                # poses_pred = poses_pred[keep, :]
                # print rois

                # combine poses
                num = rois.shape[0]
                poses = poses_init
                for i in xrange(num):
                    class_id = int(rois[i, 1])
                    if class_id >= 0:
                        poses[i, :4] = poses_pred[i, 4*class_id:4*class_id+4]
            else:
                labels_2d, probs, vertex_pred, rois, poses = \
                    sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
                print rois
                print rois.shape
                # non-maximum suppression
                # keep = nms(rois[:, 2:], 0.5)
                # rois = rois[keep, :]
                # poses = poses[keep, :]

                #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
                #vertex_pred = []
                #rois = []
                #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
        else:
            labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            vertex_pred = []
            rois = []
            poses = []

        return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses
