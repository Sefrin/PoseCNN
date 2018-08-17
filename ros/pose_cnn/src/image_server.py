import rospy
import cv2
import numpy as np
from pose_cnn.srv import *
from vision_msgs.msg import *
from fcn.config import cfg
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String
from sensor_msgs.msg import Image
# from sensor_msgs.msg import CameraInfo
from pose_cnn_msgs.msg import PoseCNNMsg

class ImageServer:

    def __init__(self, sess, network, imdb, cfg):


        self.cv_bridge = CvBridge()
        self.sess = sess
        self.net = network
        self.imdb = imdb
        self.cfg = cfg
        # initialize a node
        rospy.init_node('posecnn_server')
        self.s = rospy.Service('posecnn_recognize', recognize, self.rec)
        self.posecnn_pub = rospy.Publisher('posecnn_result', PoseCNNMsg, queue_size=1)
        self.label_pub = rospy.Publisher('posecnn_label', Image, queue_size=1)
        print "PoseCNN recognition ready."
        rospy.spin()

    def rec(self, req):
        K = np.array([[req.camera_info.K[0], req.camera_info.K[1], req.camera_info.K[2]],
                      [req.camera_info.K[3], req.camera_info.K[4], req.camera_info.K[5]],
                      [req.camera_info.K[6], req.camera_info.K[7], req.camera_info.K[8]]])
        self.meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 1000.0})
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

        im = cv2.resize(im, (640, 480))
        depth_cv = cv2.resize(depth_cv, (640, 480))
        # filename = 'images/%06d-color.png' % self.count
        # cv2.imwrite(filename, im)

        # filename = 'images/%06d-depth.png' % self.count
        # cv2.imwrite(filename, depth_cv)
        # print filename
        # self.count += 1

        # run network
        labels, probs, vertex_pred, rois, poses = self.im_segment_single_frame(self.sess, self.net, im, depth_cv, self.meta_data, \
            self.imdb._extents, self.imdb._points_all, self.imdb._symmetry, self.imdb.num_classes)

        im_label = self.imdb.labels_to_image(im, labels)

        print np.shape(rois)
        for x in rois:
            for i in x:
                print i
        print "nums {}".format(int(rois.shape[0]))
        print "channel {}".format(int(rois.shape[1]))
        # publish
        # msg = PoseCNNMsg()
        # msg.height = int(im.shape[0])
        # msg.width = int(im.shape[1])
        # msg.roi_num = int(rois.shape[0])
        # msg.roi_channel = int(rois.shape[1])
        # msg.fx = float(self.meta_data['intrinsic_matrix'][0, 0])
        # msg.fy = float(self.meta_data['intrinsic_matrix'][1, 1])
        # msg.px = float(self.meta_data['intrinsic_matrix'][0, 2])
        # msg.py = float(self.meta_data['intrinsic_matrix'][1, 2])
        # msg.factor = float(self.meta_data['factor_depth'])
        # msg.znear = float(0.25)
        # msg.zfar = float(6.0)
        # msg.label = self.cv_bridge.cv2_to_imgmsg(labels.astype(np.uint8), 'mono8')
        # msg.depth = self.cv_bridge.cv2_to_imgmsg(depth_cv, 'mono16')
        # msg.rois = rois.astype(np.float32).flatten().tolist()
        # msg.poses = poses.astype(np.float32).flatten().tolist()
        # self.posecnn_pub.publish(msg)
        msg = Detection3DArray()
        for i in range(int(rois.shape[0])):
            detection = Detection3D()
            hyp = ObjectHypothesisWithPose()
            hyp.id = rois[i, 1]
            hyp.pose.pose.orientation.x = poses[i][0]
            hyp.pose.pose.orientation.y = poses[i][1]
            hyp.pose.pose.orientation.z = poses[i][2]
            hyp.pose.pose.orientation.w = poses[i][3]
            hyp.pose.pose.position.x = poses[i][4]
            hyp.pose.pose.position.y = poses[i][5]
            hyp.pose.pose.position.z = poses[i][6]
            detection.results.append(hyp)
            msg.detections.append(detection)

        label_msg = self.cv_bridge.cv2_to_imgmsg(im_label)
        label_msg.header.stamp = rospy.Time.now()
        label_msg.header.frame_id = req.rgb_image.header.frame_id
        label_msg.encoding = 'rgb8'
        self.label_pub.publish(label_msg)
        return recognizeResponse(msg)


# ICP CODE?!??!

    # if (cfg.TEST.VERTEX_REG_2D and cfg.TEST.POSE_REFINE) or (cfg.TEST.VERTEX_REG_3D and cfg.TEST.POSE_REG):
    #     synthesizer = libsynthesizer.Synthesizer(cfg.CAD, cfg.POSE)
    #     synthesizer.setup(cfg.TRAIN.SYN_WIDTH, cfg.TRAIN.SYN_HEIGHT)
            # if cfg.TEST.POSE_REG:
            #     # pose refinement
            #     fx = meta_data['intrinsic_matrix'][0, 0] * im_scale
            #     fy = meta_data['intrinsic_matrix'][1, 1] * im_scale
            #     px = meta_data['intrinsic_matrix'][0, 2] * im_scale
            #     py = meta_data['intrinsic_matrix'][1, 2] * im_scale
            #     factor = meta_data['factor_depth']
            #     znear = 0.25
            #     zfar = 6.0
            #     poses_new = np.zeros((poses.shape[0], 7), dtype=np.float32)        
            #     poses_icp = np.zeros((poses.shape[0], 7), dtype=np.float32)     
            #     error_threshold = 0.01
            #     
        #         labels_icp = labels.copy();
        #         rois_icp = rois
        #         if imdb.num_classes == 2:
        #             I = np.where(labels_icp > 0)
        #             labels_icp[I[0], I[1]] = imdb._cls_index
        #             rois_icp = rois.copy()
        #             rois_icp[:, 1] = imdb._cls_index
        #         im_depth = cv2.resize(im_depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        #         parameters = np.zeros((7, ), dtype=np.float32)
        #         parameters[0] = fx
        #         parameters[1] = fy
        #         parameters[2] = px
        #         parameters[3] = py
        #         parameters[4] = znear
        #         parameters[5] = zfar
        #         parameters[6] = factor

        #         height = labels_icp.shape[0]
        #         width = labels_icp.shape[1]
        #         num_roi = rois_icp.shape[0]
        #         channel_roi = rois_icp.shape[1]
        #         synthesizer.icp_python(labels_icp, im_depth, parameters, height, width, num_roi, channel_roi, \
            #                                rois_icp, poses, poses_new, poses_icp, error_threshold)






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
                print rois

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
