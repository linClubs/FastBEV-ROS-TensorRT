#include "fastbev_ros.hpp"

ROSNode::ROSNode(const std::string &pkg_path, const std::string &model_name, const std::string &precision
)
{ 
  fastbev_node_ = std::make_shared<FastBEVNode>(pkg_path, model_name, precision);  
  pub_img_ = n_.advertise<sensor_msgs::Image>("/bevfusion/image_raw", 10);
  // sub_img_f_.subscribe(n_,"/cam_front/raw", 10);
  // sub_img_b_.subscribe(n_,"/cam_back/raw", 10);

  // sub_img_fl_.subscribe(n_,"/cam_front_left/raw", 10);
  // sub_img_fr_.subscribe(n_,"/cam_front_right/raw", 10);
  
  // sub_img_bl_.subscribe(n_,"/cam_back_left/raw", 10);
  // sub_img_br_.subscribe(n_,"/cam_back_right/raw", 10);

  sub_img_f_.subscribe(n_,"/dev/video0/image_raw", 10);
  sub_img_b_.subscribe(n_,"/dev/video3/image_raw", 10);

  sub_img_fl_.subscribe(n_,"/dev/video5/image_raw", 10);
  sub_img_fr_.subscribe(n_,"/dev/video1/image_raw", 10);
  
  sub_img_bl_.subscribe(n_,"/dev/video4/image_raw", 10);
  sub_img_br_.subscribe(n_,"/dev/video2/image_raw", 10);

  
  sync_ = std::make_shared<Sync>( MySyncPolicy(10), 
    sub_img_f_,  sub_img_fl_, sub_img_fr_,
    sub_img_b_ , sub_img_bl_, sub_img_br_); 
  
  sync_->registerCallback(boost::bind(&ROSNode::callback, this, _1, _2,_3, _4, _5, _6)); // 绑定回调函数

};

void ROSNode::callback(const ImageMsg::ConstPtr& msg_f_img,
    const ImageMsg::ConstPtr& msg_fl_img,
    const ImageMsg::ConstPtr& msg_fr_img,
    const ImageMsg::ConstPtr& msg_b_img,
    const ImageMsg::ConstPtr& msg_bl_img,
    const ImageMsg::ConstPtr& msg_br_img)
{
  
  cv::Mat img_f, img_fr, img_fl, img_b, img_bl, img_br;
  img_f  = cv_bridge::toCvShare(msg_f_img , "bgr8")->image;
  img_fl = cv_bridge::toCvShare(msg_fl_img, "bgr8")->image;
  img_fr = cv_bridge::toCvShare(msg_fr_img, "bgr8")->image;
  img_b  = cv_bridge::toCvShare(msg_b_img , "bgr8")->image;
  img_bl = cv_bridge::toCvShare(msg_bl_img, "bgr8")->image;
  img_br = cv_bridge::toCvShare(msg_br_img, "bgr8")->image;
  cv::resize(img_f, img_f,cv::Size(1600, 900));
  cv::resize(img_fl, img_fl,cv::Size(1600, 900));
  cv::resize(img_fr, img_fr,cv::Size(1600, 900));
  cv::resize(img_b, img_b,cv::Size(1600, 900));
  cv::resize(img_bl, img_bl,cv::Size(1600, 900));
  
  cv::resize(img_br, img_br,cv::Size(1600, 900));

  std::cout << "1111" << std::endl;
  fastbev_node_->stb_images = fastbev_node_->load_images(img_f, img_fr, img_fl, img_b, img_bl, img_br);
  std::cout << "222" << std::endl;
  auto bboxes = fastbev_node_->inference();
  
  if(bboxes.size() != 0)
  {
    cv::Mat img = fastbev_node_->draw_boxes(bboxes);
    sensor_msgs::Image::Ptr msg_img_new; 
    msg_img_new = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    pub_img_.publish(msg_img_new);
  }
}





