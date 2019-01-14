#include <map>
#include <yaml-cpp/yaml.h>
#include <pcl/point_types.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/common/common.h>
#include <pcl_ros/point_cloud.h>
typedef pcl::PointXYZRGBNormal Point;
typedef pcl::PointCloud<Point> PointCloud;
class ModelLoader
{
	public:
		ModelLoader(std::string home_path, std::string models_dir);
		std::string getName(int id);
		int getID(std::string name);
		void getCloud(int id, PointCloud::Ptr& cloud);
		void getCloud(std::string name, PointCloud::Ptr& cloud);
		bool objectInDB(int id);
		bool objectInDB(std::string name);
	private:
		std::string home_path_;
		std::string models_dir_;
		std::map<std::string, int> object_name_to_id_;
		std::map<int, std::string> object_id_to_name_;
		std::map<std::string, PointCloud::ConstPtr> models_;
		void loadMapping();
};