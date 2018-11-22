#include <map>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <dynamic_reconfigure/server.h>

#include "tf/transform_listener.h"
#include "tf/message_filter.h"
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#include <pose_cnn/posecnn_recognize.h>
#include <pose_cnn_msgs/PoseCNNMsg.h>
#include <vision_msgs/Detection3DArray.h>

#include <pose_cnn/PoseCNNRefinedConfig.h>
#include <pose_cnn/database_loader.h>

#include <cv_bridge/cv_bridge.h>
#include <depth_image_proc/depth_conversions.h>
#include <image_geometry/pinhole_camera_model.h>
#include <sensor_msgs/image_encodings.h>

#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/geometry.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> approx_sync;

namespace enc = sensor_msgs::image_encodings;

class PoseCNNRefined
{
	public:
		PoseCNNRefined(std::string home_path, std::string models_dir);
	private:
		ros::NodeHandle nh_;

		std::string models_dir_;
		std::string home_path_;

		message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub_;
		message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
		message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
		ros::Publisher cloud_pub_;
		ros::Publisher cloud_debug_pub_;
		ros::Publisher detection_pub_;
		message_filters::Synchronizer<approx_sync> * sync_;
		ros::ServiceClient obj_rec_client_, obj_rec_set_camera_client_;
		ros::ServiceServer recognition_server_;

		dynamic_reconfigure::Server<pose_cnn::PoseCNNRefinedConfig> dyn_reconf_server_;
		pose_cnn::PoseCNNRefinedConfig config_;

		DatabaseLoader db_loader_;

		void dynamicReconfigureCallback(pose_cnn::PoseCNNRefinedConfig &config, uint32_t level);
		void syncCallback(const sensor_msgs::Image::ConstPtr& rgb, const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& info);
		void callPoseCNN(const sensor_msgs::Image::ConstPtr& rgb, const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& info, boost::shared_ptr<vision_msgs::Detection3DArray>& d3d_out);
		void performICP(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr scene, const pcl::PointIndicesConstPtr cluster_indices, const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr model, tf::Transform& tf_out);
		void getPointsWithLabel(int id, const sensor_msgs::Image& label_image, const vision_msgs::BoundingBox2D& bbox, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr scene, pcl::PointCloud<pcl::PointXYZ>::Ptr& label_cloud);
		void convertToPointcloud(const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& info, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out);
		bool isInBBOX(int w, int h, const vision_msgs::BoundingBox2D& bbox);
		bool serviceCallback(pose_cnn::posecnn_refined::Request  &req,
         pose_cnn::posecnn_refined::Response &res);
};