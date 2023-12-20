#include "fastbev_ros.hpp"

int main(int argc, char **argv)
{ 
  rclcpp::init(argc, argv);
  auto param_node = std::make_shared<rclcpp::Node>("param_node");

  std::string pkg_path;
  std::string model_name = "resnet18";
  std::string precision = "fp16";
  
  // 声明参数
  param_node->declare_parameter<std::string>("model_name", model_name);
  // 从参数列表更新参数
  param_node->get_parameter("model_name", model_name);

  pkg_path = ament_index_cpp::get_package_share_directory("fastbev") + "/../../../../src/FastBEV-ROS-TensorRT";

  std::cout << "\033[1;32m--pkg_path: "   << pkg_path   << "\033[0m" << std::endl;
  std::cout << "\033[1;32m--model_name: " << model_name << "\033[0m" << std::endl;
  std::cout << "\033[1;32m--precision : " << precision  << "\033[0m" << std::endl;

  auto fastbev_node = std::make_shared<ROSNode>(pkg_path, model_name, precision);
  
  rclcpp::spin(fastbev_node);
  rclcpp::shutdown();
  return 0;
}