
#include "fastbev_plugin.hpp"


int main(int argc, char** argv)
{

    const char* data = "../example-data";
    const char* model = "resnet18";
    const char* precision = "fp16";

    if (argc > 1) data      = argv[1];
    if (argc > 2) model     = argv[2];
    if (argc > 3) precision = argv[3];

    // nv::format替换路径
    std::string Save_Dir = nv::format("../model/%s/result", model);

    FastBEVNode fastnode("../config", "resnet18", "fp16");
    // 配置网络基本的
    auto core = fastnode.create_core(model, precision);
    
    if (core == nullptr) 
    {
      printf("Core has been failed.\n");
      return -1;
    }
 
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    core->print();
    core->set_timer(true);

    // 加载数据 6张图转成vector<data指针>
    auto images = fastnode.load_images(data);

    // nv::format替换路径  
    // nv::Tensor::load 用于从文件中加载Tensor对象的。它会读取一个二进制文件，
    // 解析其中的头部信息，然后根据头部信息中的维度、数据类型和形状等信息创建一个Tensor对象
    auto valid_c_idx = nv::Tensor::load(nv::format("%s/valid_c_idx.tensor", data), false);
    auto valid_x = nv::Tensor::load(nv::format("%s/x.tensor", data), false);
    auto valid_y = nv::Tensor::load(nv::format("%s/y.tensor", data), false);

    // valid_c_idx.print("Tensor", 0, 6, 2);
    valid_x.print("Tensor", 0, 6, 2);
    valid_y.print("Tensor", 0, 6, 2);

    float valid_c_idx_arr[160000 * 6];
    for(auto i = 0; i < 160000 * 6; ++i)
    {
      valid_c_idx_arr[i] = valid_c_idx.ptr<float>()[i];
    }

    printf("%ld,  %ld ", valid_c_idx.size(0),valid_c_idx.size(1));

    // update虚函数 给valid_c_idx，valid_x，valid_y分配内存
    // core->update(valid_c_idx.ptr<float>(), valid_x.ptr<int64_t>(), valid_y.ptr<int64_t>(), stream);
    core->update(valid_c_idx_arr, valid_x.ptr<int64_t>(), valid_y.ptr<int64_t>(), stream);
    
    // warmup 推理
    // std::vector<fastbev::post::transbbox::BoundingBox> bboxes
    auto bboxes = core->forward((const unsigned char**)images.data(), stream);
    
    // evaluate inference time 评价推理时间 可以不需要
    // for (int i = 0; i < 5; ++i) 
    // {
    //   core->forward((const unsigned char**)images.data(), stream);
    // }

    std::string save_file_name = Save_Dir + ".txt";
    fastnode.SaveBoxPred(bboxes, save_file_name);

    fastnode.draw_boxes(bboxes);
    std::cout << bboxes.size() << std::endl;
    // for(auto bbox : bboxes)
    // { 
    //   // id x y z w l h yaw vx vy score
    //   std::cout << bbox.id << "  ";
    //   std::cout << bbox.position.x << "  " << bbox.position.y << "  " << bbox.position.z << "  ";
    //   std::cout << bbox.size.w << "  " << bbox.size.l << "  " << bbox.size.h << "  ";
    //   std::cout << bbox.z_rotation << "  ";
    //   std::cout << bbox.score << "  ";
    //   // std::cout << bbox.velocity.vx << "  " << bbox.velocity.vy << "  ";
    //   std::cout << std::endl;
    // }

    // destroy memory
    fastnode.free_images(images);
    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}