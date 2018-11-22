#include <pose_cnn/database_loader.h>

DatabaseLoader::DatabaseLoader(std::string home_path, std::string models_dir) :
	home_path_(home_path),
	models_dir_(models_dir)
{
	loadMapping();
}

void DatabaseLoader::loadMapping()
{
    try
    {
        YAML::Node mapping = YAML::LoadFile(home_path_ + "/objects.yaml");
        int index = 1; //ids start at 1
        for (auto object: mapping)
        {
            std::cout << "loaded object: " << object.as<std::string>() << " with id: " << index << std::endl;
            object_id_to_name_[index] = object.as<std::string>();
            object_name_to_id_[object.as<std::string>()] = index++;

        }
    } catch (YAML::BadFile) {
        ROS_ERROR("Could not load mapping..");
        ros::shutdown();
    }
}

std::string DatabaseLoader::getName(int id)
{
	return object_id_to_name_.at(id);
}

int DatabaseLoader::getID(std::string name)
{
	return object_name_to_id_.at(name);
}
void DatabaseLoader::getCloud(int id, PointCloud::Ptr& cloud)
{
    getCloud(getName(id), cloud);
}
void DatabaseLoader::getCloud(std::string name, PointCloud::Ptr& cloud)
{
    if (models_.count(name) == 0)
    {
        // PointCloud cloud;
        pcl::io::loadPCDFile<Point> (models_dir_ +"/" + name + "/3D_model.pcd", *cloud); //* load the file
        models_[name] = boost::make_shared<const PointCloud>(*cloud);
    }
    else
    {
        cloud = boost::make_shared<PointCloud>(*models_.at(name));
    }
}

bool DatabaseLoader::objectInDB(int id)
{
    return object_id_to_name_.count(id);
}
bool DatabaseLoader::objectInDB(std::string name)
{
    return object_name_to_id_.count(name);
}