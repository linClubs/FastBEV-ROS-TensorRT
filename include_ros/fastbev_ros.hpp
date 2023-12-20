#ifndef fastbev_ros_h
#define fastbev_ros_h
#include "fastbev_plugin.hpp"

#include <opencv2/opencv.hpp>

// #include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
// #include <sensor_msgs/msg/image.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>


// message_filters消息同步器
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h> // 时间接近


class ROSNode
{
private:
    ros::NodeHandle n_;
    std::string pkg_path_, model_name_, precision_;
    
    using ImageMsg = sensor_msgs::Image;
    ros::Publisher pub_img_;
    // rclcpp::Publisher<ImageMsg>::SharedPtr pub_img_;

    typedef message_filters::sync_policies::ApproximateTime<
      ImageMsg, ImageMsg,ImageMsg,ImageMsg, ImageMsg,ImageMsg> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    std::shared_ptr<Sync> sync_;
    
    // message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cloud_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_img_f_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_img_fl_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_img_fr_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_img_b_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_img_bl_; 
    message_filters::Subscriber<sensor_msgs::Image> sub_img_br_; 

    std::shared_ptr<FastBEVNode> fastbev_node_;

public:
    ROSNode(const std::string &pkg_path, const std::string &model_name, const std::string &precision);
    ~ROSNode(){};
    void callback(const ImageMsg::ConstPtr& msg_f_img,
    const ImageMsg::ConstPtr& msg_fl_img,
    const ImageMsg::ConstPtr& msg_fr_img,
    const ImageMsg::ConstPtr& msg_b_img,
    const ImageMsg::ConstPtr& msg_bl_img,
    const ImageMsg::ConstPtr& msg_br_img);
};
#endif