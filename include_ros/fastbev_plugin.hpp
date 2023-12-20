#ifndef fastbev_plugin_h
#define fastbev_plugin_h

#include <cuda_runtime.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include <stb_image.h>
#include <stb_image_write.h>
#include <algorithm>

#include "fastbev/fastbev.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"
#include "common/visualize.hpp"

// const int final_height = 256;
// const int final_weith = 704;

class FastBEVNode
{
private:
    std::string pkg_path_;
    std::string model_name_;
    std::string precision_;
    std::string config_path_;


    std::shared_ptr<fastbev::Core> core;
    cudaStream_t stream;
public:    
    std::vector<unsigned char *> stb_images;

public:
    FastBEVNode(const std::string &pkg_path, const std::string &model_name, const std::string &precision);
    ~FastBEVNode();
    unsigned char* cv2stb(cv::Mat cv_img);
    // 读取图像， 读取6张图像文件
    std::vector<unsigned char*> load_images(const std::string& root);
    // 用6张mat喂图像 默认fr在fb前面
    std::vector<unsigned char*> load_images(const cv::Mat& img_f, const cv::Mat& img_fr, const cv::Mat& img_fl,
const cv::Mat& img_b, const cv::Mat& img_bl, const cv::Mat& img_br); 
    
    void free_images(std::vector<unsigned char*>& images); 
    void readParam(const std::string &config_path, nv::Tensor &valid_x, nv::Tensor &valid_y, nv::Tensor &valid_c_idx);
    
    std::shared_ptr<fastbev::Core> create_core(const std::string& model, const std::string& precision);
    
    std::vector<fastbev::post::transbbox::BoundingBox> inference();

    void SaveBoxPred(std::vector<fastbev::post::transbbox::BoundingBox> boxes, std::string file_name);
    cv::Mat draw_boxes(std::vector<fastbev::post::transbbox::BoundingBox> bboxes);
};

#endif