#include "fastbev_ros.hpp"
#include <memory>

ROSNode::ROSNode(const std::string &pkg_path, const std::string &model_name, const std::string &precision
) : Node("fastbev_node"), pkg_path_(pkg_path),model_name_(model_name), precision_(precision)
{   

  // 初始化FastBEV插件
  fastbev_plugin.reset(new FastBEVNode(pkg_path_, model_name_, precision_));

  sub_img_f_  = std::make_shared<message_filters::Subscriber<ImageMsg> >(this, "/cam_front/raw");
  sub_img_fl_ = std::make_shared<message_filters::Subscriber<ImageMsg> >(this, "/cam_front_left/raw");
  sub_img_fr_ = std::make_shared<message_filters::Subscriber<ImageMsg> >(this, "/cam_front_right/raw");
  sub_img_b_  = std::make_shared<message_filters::Subscriber<ImageMsg> >(this, "/cam_back/raw");
  sub_img_bl_ = std::make_shared<message_filters::Subscriber<ImageMsg> >(this, "/cam_back_left/raw");
  sub_img_br_ = std::make_shared<message_filters::Subscriber<ImageMsg> >(this, "/cam_back_left/raw");

  sync_ = std::make_shared<Sync>(MySyncPolicy(10), *sub_img_f_, *sub_img_fl_, *sub_img_fr_ ,*sub_img_b_, *sub_img_bl_, *sub_img_br_);
  sync_->registerCallback(&ROSNode::callback, this);
}

void ROSNode::callback(const ImageMsg::SharedPtr msg_img_f,
    const ImageMsg::SharedPtr msg_img_fl,
    const ImageMsg::SharedPtr msg_img_fr,
    const ImageMsg::SharedPtr msg_img_b,
    const ImageMsg::SharedPtr msg_img_bl,
    const ImageMsg::SharedPtr msg_img_br)
{
  
  cv::Mat img_f, img_fr, img_fl, img_b, img_bl, img_br;
  img_f  = cv_bridge::toCvShare(msg_img_f,  "bgr8")->image;
  img_fl = cv_bridge::toCvShare(msg_img_fl, "bgr8")->image;
  img_fr = cv_bridge::toCvShare(msg_img_fr, "bgr8")->image;
  img_b  = cv_bridge::toCvShare(msg_img_b,  "bgr8")->image;
  img_bl = cv_bridge::toCvShare(msg_img_bl, "bgr8")->image;
  img_br = cv_bridge::toCvShare(msg_img_br, "bgr8")->image;

  RCLCPP_INFO(this->get_logger(), "cv_bridge exception");
  
  fastbev_plugin->stb_images = fastbev_plugin->load_images(img_f, img_fr, img_fl, img_b, img_bl, img_br);
  auto bboxes = fastbev_plugin->inference();
  
  if(bboxes.size() != 0)
  {
    cv::Mat img = fastbev_plugin->draw_boxes(bboxes);

  }
}





