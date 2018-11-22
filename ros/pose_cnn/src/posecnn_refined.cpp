#include "pose_cnn/posecnn_refined.h"

PoseCNNRefined::PoseCNNRefined(std::string home_path, std::string models_dir) :
    nh_("~"),
    models_dir_(models_dir),
    home_path_(home_path),
    cam_info_sub_(nh_, "/camera/depth_registered/camera_info", 1),
    rgb_sub_(nh_, "/camera/rgb/image", 1),
    depth_sub_(nh_, "/camera/depth_registered/image", 1),
    db_loader_(home_path, models_dir)
{

    obj_rec_client_ = nh_.serviceClient<pose_cnn::posecnn_recognize>("/posecnn_recognize");
    // tf_filter_ = new tf::MessageFilter<sensor_msgs::PointCloud2>(cloud_sub_, tf_, "aruco_ref", 10);
    // tf_filter_->registerCallback(boost::bind(&RecognitionNode::cloudCallback, this, _1));
    // tf_filter_ = new tf::MessageFilter<sensor_msgs::PointCloud2>(rgb, tf_, cam_name_ + "/aruco_node/aruco_ref", 10);
    sync_ = new message_filters::Synchronizer<approx_sync>(approx_sync(10), rgb_sub_, depth_sub_, cam_info_sub_);
    sync_->registerCallback(boost::bind(&PoseCNNRefined::syncCallback, this, _1, _2, _3));
    detection_pub_ = nh_.advertise<vision_msgs::Detection3DArray>("/PoseCNN/detections", 1, true);
    cloud_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGBNormal>>("detected_models", 1, true);
    cloud_debug_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("debug_cloud", 1, true);

    dynamic_reconfigure::Server<pose_cnn::PoseCNNRefinedConfig>::CallbackType f;
    f = boost::bind(&PoseCNNRefined::dynamicReconfigureCallback, this, _1, _2);
    dyn_reconf_server_.setCallback(f);
    recognition_server_ = nh_.advertiseService("posecnn_recognize_refined", &PoseCNNRefined::serviceCallback, this);

}

bool PoseCNNRefined::serviceCallback(pose_cnn::posecnn_recognize_refined::Request  &req,
         pose_cnn::posecnn_recognize_refined::Response &res)
{
    boost::shared_ptr<sensor_msgs::Image> rgb = boost::make_shared<sensor_msgs::Image>(req.rgb);
    boost::shared_ptr<sensor_msgs::Image> depth = boost::make_shared<sensor_msgs::Image>(req.depth);
    boost::shared_ptr<sensor_msgs::CameraInfo> info = boost::make_shared<sensor_msgs::CameraInfo>(req.cam_info);
    boost::shared_ptr<vision_msgs::Detection3DArray> detections = boost::make_shared<vision_msgs::Detection3DArray>();
    callPoseCNN(rgb, depth, info, detections);
    res.detections = *detections;
    return true;
}

void PoseCNNRefined::dynamicReconfigureCallback(pose_cnn::PoseCNNRefinedConfig &config, uint32_t level)
{
    config_ = config;
}

void PoseCNNRefined::syncCallback(const sensor_msgs::Image::ConstPtr& rgb, const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& info)
{
    boost::shared_ptr<vision_msgs::Detection3DArray> d3d_ptr = boost::make_shared<vision_msgs::Detection3DArray>();
	callPoseCNN(rgb, depth, info, d3d_ptr);
    detection_pub_.publish(d3d_ptr);
}

void PoseCNNRefined::callPoseCNN(const sensor_msgs::Image::ConstPtr& rgb, const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& info, boost::shared_ptr<vision_msgs::Detection3DArray>& d3d_out)
{
    pose_cnn::posecnn_recognize srv;
    // cloud_pub_.publish(cloud);
    srv.request.camera_info = *info;
    srv.request.rgb_image = *rgb;
    srv.request.depth_image = *depth;

    cv_bridge::CvImagePtr depth_ptr;
    try
    {
      depth_ptr = cv_bridge::toCvCopy(depth, depth->encoding);
      std::cout << "encoding: " << depth->encoding << std::endl;
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    if (obj_rec_client_.call(srv))
    {
        if(srv.response.detections.detections.size() == 0)
        {
            ROS_WARN("I did not detected any object from the model database in the current scene.");
        }
        else
        {
            //remove for serious results. this just returns the posecnn result for temporary testing.
            // d3d_out = boost::make_shared<vision_msgs::Detection3DArray>(srv.response.detections);
            // return;


            ROS_INFO("I detected: %zd objects.", srv.response.detections.detections.size());

            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_all (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            cloud_all->header.frame_id = rgb->header.frame_id;
            pcl_conversions::toPCL(ros::Time::now(), cloud_all->header.stamp);
            pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            convertToPointcloud(depth, info, scene_cloud);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

            // pcl::PointCloud<pcl::PointXYZ>::Ptr scene_downsampled = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            // pcl::VoxelGrid<pcl::PointXYZ> scene_vg;
            // scene_vg.setInputCloud(scene_cloud);
            // scene_vg.setLeafSize(config_.voxel_grid_leaf_size, config_.voxel_grid_leaf_size, config_.voxel_grid_leaf_size);
            // scene_vg.filter(*scene_downsampled);


            //step 1: RANSAC on complete scene to remove ground.
            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            // Create the segmentation object
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            // Optional
            seg.setOptimizeCoefficients (true);
            // Mandatory
            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            seg.setDistanceThreshold (config_.ransac_distance);

            seg.setInputCloud (scene_cloud);
            seg.segment (*inliers, *coefficients);
            //extract points that are not in the plane
            pcl::PointCloud<pcl::PointXYZ>::Ptr scene_filtered (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud (scene_cloud);
            extract.setIndices (inliers);
            extract.setNegative(true);
            extract.filter (*scene_filtered);
            std::vector<int> dummy;
            pcl::removeNaNFromPointCloud(*scene_filtered, *scene_filtered, dummy);

            #pragma omp parallel for
            for (size_t object_idx = 0 ; object_idx < srv.response.detections.detections.size() ; ++object_idx)
            {
                vision_msgs::Detection3D * object = &srv.response.detections.detections[object_idx];

                ROS_INFO(" - %s", db_loader_.getName(object->results[0].id).c_str());
                pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr model_cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
                model_cloud->header.frame_id = object->header.frame_id;
                model_cloud->header.seq = object->header.seq;
                pcl_conversions::toPCL(object->header.stamp, model_cloud->header.stamp);

                db_loader_.getCloud(object->results[0].id, model_cloud);

                pcl::PointCloud<pcl::PointXYZ>::Ptr label_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                getPointsWithLabel(object->results[0].id, srv.response.label_image_raw, srv.response.bboxes[object_idx], scene_cloud, label_cloud);
                //rotate PointCloud, transform it to find centroid respective to rotation
                tf::Transform init_rotation;
                tf::poseMsgToTF(object->results[0].pose.pose, init_rotation);
                init_rotation.setOrigin(tf::Vector3(0, 0, 0));
                pcl_ros::transformPointCloud(*model_cloud, *model_cloud, init_rotation);

                //calculate center offset
                pcl::PointXYZ model_centroid;
                pcl::computeCentroid(*model_cloud, model_centroid);
                tf::Transform center_offset;
                center_offset.setIdentity();
                tf::Vector3 center_offset_vector;
                center_offset_vector.setX(-model_centroid.x);
                center_offset_vector.setY(-model_centroid.y);
                center_offset_vector.setZ(-model_centroid.z);
                center_offset.setOrigin(center_offset_vector);
                //calculate center of labeled area, in order to find good initial position 1. option: from labeled clusters, 2. option: from 2d bbox center projection
                // 1
                pcl::PointXYZ label_cluster_centroid;
                pcl::computeCentroid(*label_cloud, label_cluster_centroid);
                //2
                image_geometry::PinholeCameraModel model;
                model.fromCameraInfo(info);
                cv::Point2d bbox_center_2d(srv.response.bboxes[object_idx].center.x, srv.response.bboxes[object_idx].center.y);
                cv::Point3d bbox_center_3d = model.projectPixelTo3dRay(bbox_center_2d);

                tf::Transform init_location;
                init_location.setIdentity();
                tf::Vector3 init_location_vector;

                init_location_vector.setX(bbox_center_3d.x);
                init_location_vector.setY(bbox_center_3d.y);
                init_location_vector.setZ(label_cluster_centroid.z);
                init_location.setOrigin(init_location_vector);

                //transform into initial position for ICP
                pcl_ros::transformPointCloud(*model_cloud, *model_cloud, center_offset * init_location);


                // step 2: take all points that are within reach of the largest object dimension, in every orientation
                pcl::PointCloud<pcl::PointXYZ>::Ptr object_roi (new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PointXYZRGBNormal min_pt, max_pt;
                pcl::getMinMax3D (*model_cloud, min_pt, max_pt);
                double max_size = std::max({max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z});
                pcl::PointIndices::Ptr roi_indices (new pcl::PointIndices);

                double min_x = init_location_vector.x() - max_size;
                double max_x = init_location_vector.x() + max_size;
                double min_y = init_location_vector.y() - max_size;
                double max_y = init_location_vector.y() + max_size;
                double min_z = init_location_vector.z() - max_size;
                double max_z = init_location_vector.z() + max_size;
                pcl::CropBox<pcl::PointXYZ> boxFilter (true);
                boxFilter.setMin(Eigen::Vector4f(min_x, min_y, min_z, 1.0));
                boxFilter.setMax(Eigen::Vector4f(max_x, max_y, max_z, 1.0));
                boxFilter.setInputCloud(scene_filtered);
                boxFilter.filter(*object_roi);

                std::vector<int> dummy;
                pcl::removeNaNFromPointCloud(*object_roi, *object_roi, dummy);

                // perform icp
                tf::Transform icp_tf;
                performICP(object_roi, boost::make_shared<pcl::PointIndices>(pcl::PointIndices()), model_cloud, icp_tf);

                //final transformation for output
                pcl_ros::transformPointCloud(*model_cloud, *model_cloud, icp_tf);
                *cloud_all += *model_cloud;
                cloud_pub_.publish(cloud_all);
                geometry_msgs::Transform ros_tf;
                //concatenate transforms frm right to left!!
                tf::transformTFToMsg((icp_tf * center_offset * init_location * init_rotation), ros_tf); // icp_tf * ...
                object->results[0].pose.pose.orientation = ros_tf.rotation;
                object->results[0].pose.pose.position.x = ros_tf.translation.x;
                object->results[0].pose.pose.position.y = ros_tf.translation.y;
                object->results[0].pose.pose.position.z = ros_tf.translation.z;
            }
            // cloud_debug_pub_.publish(cloud_cluster);
            cloud_pub_.publish(cloud_all);
            d3d_out = boost::make_shared<vision_msgs::Detection3DArray>(srv.response.detections);
        }
    }
    else
    {
        ROS_ERROR("Error calling posecnn service. ");
        return;
    }
}

void PoseCNNRefined::getPointsWithLabel(int id, const sensor_msgs::Image& label_image, const vision_msgs::BoundingBox2D& bbox, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr scene, pcl::PointCloud<pcl::PointXYZ>::Ptr& label_cloud)
{
    for (size_t i = 0 ; i < label_image.width ; i++)
    {
        for (size_t j = 0 ; j < label_image.height ; j++)
        {
            if (label_image.data[label_image.width * j + i] == id && isInBBOX(i, j, bbox) && pcl::isFinite(scene->at(i, j)))
            {
                label_cloud->push_back(scene->at(i, j));
            }

        }
    }
    std::vector<int> dummy;
    pcl::removeNaNFromPointCloud(*label_cloud, *label_cloud, dummy);
}

bool PoseCNNRefined::isInBBOX(int w, int h, const vision_msgs::BoundingBox2D& bbox)
{
    // return true;
    int wmax = bbox.center.x + bbox.size_x;
    int wmin = bbox.center.x - bbox.size_x;
    int hmax = bbox.center.y + bbox.size_y;
    int hmin = bbox.center.y - bbox.size_y;
    if (w <= wmax && w >= wmin && h <= hmax && h >= hmin)
    {
        return true;
    }
    return false;
}


void PoseCNNRefined::performICP(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr scene, const pcl::PointIndicesConstPtr cluster_indices, const pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr model, tf::Transform& tf_out)
{
    if (scene->empty())
    {
        tf_out.setIdentity();
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr model_no_rgb = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::copyPointCloud(*model, *model_no_rgb);

    //downsample model
    // pcl::PointCloud<pcl::PointXYZ>::Ptr model_downsampled = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    // pcl::VoxelGrid<pcl::PointXYZ> model_vg;
    // model_vg.setInputCloud(model_no_rgb);
    // model_vg.setLeafSize(0.005, 0.005, 0.005);
    // model_vg.filter(*model_downsampled);
    // pcl::removeNaNFromPointCloud(*model_downsampled, *model_downsampled, dummy);
    std::vector<int> dummy;
    pcl::removeNaNFromPointCloud(*model_no_rgb, *model_no_rgb, dummy);

    //downsample scene cluster
    // pcl::PointCloud<pcl::PointNormal>::Ptr scene_downsampled = boost::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    // pcl::VoxelGrid<pcl::PointNormal> scene_vg;
    // scene_vg.setInputCloud(scene_with_normals);
    // scene_vg.setLeafSize(config_.voxel_grid_leaf_size, config_.voxel_grid_leaf_size, config_.voxel_grid_leaf_size);
    // scene_vg.filter(*scene_downsampled);

    ROS_INFO("Optimizing pose...");
    ROS_INFO("Scene points: %lu, model points: %lu", scene->size(), model_no_rgb->size());
    ROS_INFO("With parameters:\n - MaxCorrespondenceDistance: %f\n"
                                 "- MaxIterations: %d\n"
                                 "- MaxTransformationEpsilon: %f\n"
                                 "- MaxEuclideanFitnessEpsilon: %f\n ", config_.icp_max_correspondence_distance, config_.icp_iterations, config_.icp_transformation_epsilon, config_.icp_euclidean_fitness_epsilon);
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    icp.setUseReciprocalCorrespondences(false);
    icp.setMaxCorrespondenceDistance (config_.icp_max_correspondence_distance);
    icp.setMaximumIterations (config_.icp_iterations);
    icp.setTransformationEpsilon (config_.icp_transformation_epsilon);
    icp.setEuclideanFitnessEpsilon (config_.icp_euclidean_fitness_epsilon);
    if (cluster_indices->indices.size() > 0)
    {
        icp.setIndices(cluster_indices);
    }
    icp.setInputSource(scene);
    icp.setInputTarget(model_no_rgb);
    pcl::PointCloud<pcl::PointXYZ> final;
    icp.align(final);
    if (icp.hasConverged())
    {
        std::cout << "ICP has converged: " << (icp.hasConverged() ? "TRUE" : "FALSE") << " score: " << icp.getFitnessScore() << std::endl;
        switch (icp.getConvergeCriteria()->getConvergenceState())
        {
            case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_NOT_CONVERGED:
                std::cout << "didnt converge" << std::endl;
                break;
            case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_ITERATIONS:
                std::cout << "Max Iterations exceeded." << std::endl;
                break;
            case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_TRANSFORM:
                std::cout << "Transform epsilon?." << std::endl;
                break;
            case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_ABS_MSE:
                std::cout << "Abs MSE exceeded." << std::endl;
                break;
            case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_REL_MSE:
                std::cout << "Rel MSE exceeded." << std::endl;
                break;
            case pcl::registration::DefaultConvergenceCriteria<float>::ConvergenceState::CONVERGENCE_CRITERIA_NO_CORRESPONDENCES:
                std::cout << "No correspondences found." << std::endl;
                break;
            default:
                std::cout << "well this didnt work" << std::endl;

        }
        Eigen::Matrix4f icp_mat = icp.getFinalTransformation();

        Eigen::Affine3d eigen_icp;
        eigen_icp.matrix() = icp_mat.cast<double>();
        tf::Transform tf_icp;
        tf_icp.setRotation(tf_icp.getRotation().normalize());
        tf::transformEigenToTF(eigen_icp, tf_icp);
        tf_out = tf_icp.inverse();
    }
    else
    {
        tf_out.setIdentity();
    }
}

void PoseCNNRefined::convertToPointcloud(const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& info, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out)
{
    //copied from depth_image_proc pointcloud xyz nodelet
    sensor_msgs::PointCloud2::Ptr cloud;
    cloud = boost::make_shared<sensor_msgs::PointCloud2>();
    cloud->header = depth->header;
    cloud->height = depth->height;
    cloud->width  = depth->width;
    cloud->is_dense = false;
    cloud->is_bigendian = false;
    sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud);
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

    // Update camera model
    image_geometry::PinholeCameraModel model;
    model.fromCameraInfo(info);

    if (depth->encoding == enc::TYPE_16UC1)
    {
        depth_image_proc::convert<uint16_t>(depth, cloud, model);
    }
    else if (depth->encoding == enc::TYPE_32FC1)
    {
        depth_image_proc::convert<float>(depth, cloud, model);
    }
    pcl::fromROSMsg(*cloud, *cloud_out);
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "posecnn_refined_node");
    std::string home_path = "";
    std::string models_dir = "data/models";
    if (argc == 1)
    {
        ROS_INFO("USAGE: start with parameters: [str]home_path, [str]models_dir");
    }
    if (argc > 1)
    {
        home_path = argv[1];
    }
    if (argc > 2)
    {
        models_dir = argv[2];
    }
    PoseCNNRefined rgt(home_path, models_dir);

    ros::spin();
}
