#include "stb_image.h"
#include <opencv2/opencv.hpp>

int main() {
    std::string img_path = "/home/lin/code/CUDA-FastBEV/example-data/0-FRONT.jpg";
     cv::Mat cv_img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    
    if (!cv_img.empty()) 
    {
       
        int w = cv_img.cols;
        int h = cv_img.rows;
        int c = cv_img.channels();
        std::vector<unsigned char> buffer;
        cv::imencode(".jpg", cv_img, buffer);
        
        // 使用stbi_load函数加载图像数据
        unsigned char* stbi_data;
        stbi_data = stbi_load_from_memory(buffer.data(), buffer.size(), &w, &h, &c, 0);

       
        if (stbi_data != nullptr) 
        {
            // 在这里使用加载的图像数据
            // ...
            // 释放stbi_load函数分配的内存
            stbi_image_free(stbi_data);
        } 
        else 
        {
            // 加载图像失败
        }
    } 
    else 
    {
        // 读取图像失败
    }

    return 0;
}
