#ifndef fastbev_ros_h
#define fastbev_ros_h
#include "fastbev_plugin.hpp"

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/image.hpp>

// message_filters消息同步器
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h> // 时间接近

#include <cv_bridge/cv_bridge.h>

class ROSNode : public rclcpp::Node
{
private:
    std::string pkg_path_, model_name_, precision_;
    
    using ImageMsg = sensor_msgs::msg::Image;
    typedef message_filters::sync_policies::ApproximateTime<
    ImageMsg, ImageMsg,ImageMsg,ImageMsg,ImageMsg,ImageMsg> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;
    
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > sub_img_f_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > sub_img_fl_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > sub_img_fr_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > sub_img_b_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > sub_img_bl_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image> > sub_img_br_;

    // 定义一个FastBEV插件
    std::shared_ptr<FastBEVNode> fastbev_plugin;

public:
    ROSNode(const std::string &pkg_path, const std::string &model_name, const std::string &precision);
    ~ROSNode(){};
    void callback(const ImageMsg::SharedPtr msg_f_img,
    const ImageMsg::SharedPtr msg_fl_img,
    const ImageMsg::SharedPtr msg_fr_img,
    const ImageMsg::SharedPtr msg_b_img,
    const ImageMsg::SharedPtr msg_bl_img,
    const ImageMsg::SharedPtr msg_br_img);
};

#endif