#include "fastbev_ros.hpp"
#include <ros/package.h>

int main(int argc, char **argv)
{ 
  ros::init(argc, argv, "fastbev_node");
  ros::NodeHandle n;
  std::string pkg_path   = ros::package::getPath("fastbev");
  std::string model_name;
  std::string precision;

  n.param<std::string>("model_name", model_name, "resnet18");
  n.param<std::string>("precision",  precision, "fp16");
  
  std::cout << "\033[1;32m--pkg_path: " << pkg_path << "\033[0m" << std::endl;
  std::cout << "\033[1;32m--model_name: " << model_name << "\033[0m" << std::endl;
  std::cout << "\033[1;32m--precision : " << precision << "\033[0m" << std::endl;

  auto fastbev_node = std::make_shared<ROSNode>(pkg_path, model_name, precision);
  ros::spin();
  return 0;
}