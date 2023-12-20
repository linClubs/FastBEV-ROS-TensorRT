#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <opencv2/opencv.hpp>


int main() 
{
    int width, height, channels;
    std::string img_path = "/home/lin/code/CUDA-FastBEV/example-data/0-FRONT.jpg";
    unsigned char* image = stbi_load(img_path.c_str(), &width, &height, &channels, 0);
    
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "channels:" << channels << std::endl;
    
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC3);
    
    if (image != nullptr) 
    {   
        img.data = image;
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        
        cv::Mat cv_img(height, width, CV_8UC3, image);
        // 在这里使用图像数据进行显示或处理
        std::cout << "成功加载图像！" << std::endl;
        
        cv::imshow("img", img);
        cv::imshow("cv_img", cv_img);
        cv::waitKey();
        
        stbi_image_free(image);
        
    
    } else {
        std::cout << "无法加载图像。" << std::endl;
    }

    return 0;
}
