
# 生成模型

~~~python
# 1 修改修改environment.sh文件下TensorRT, CUDA, cudnn路径
主要是TensorRT的路径 , cuda和cudnn一般都安装在/usr/local,基本不用修改

# 2 选择模型和精度  模型resnet18/resnet18int8/resnet18int8head 精度fp16/int8 
# 修改environment.sh下的DEBUG_MODEL和DEBUG_PRECISION变量的值
# export DEBUG_MODEL=resnet18   
# export DEBUG_PRECISION=fp16

# 3 生成模型 在该工程下执行
./tool/build_trt_engine.sh
~~~

+ 执行完`build_trt_engine.sh`脚本时,耗时比较久, 跟显卡算力有关。选择`resnet18`模型生成目录结构如下:

~~~python
...
├──model
    ├──resnet18
        ├── build
            ├── fastbev_post_trt_decode.json
            ├── fastbev_post_trt_decode.log
            ├── fastbev_post_trt_decode.plan
            ├── fastbev_pre_trt.json
            ├── fastbev_pre_trt.log
            └── fastbev_pre_trt.plan
...
~~~

~~~
/usr/bin/ld: 找不到 -lspconv
~~~


~~~python
/usr/bin/ld: 找不到 -lnvinfer
cmakelists.txt中添加：
link_directories( ${TensorRT_Lib}) 
# libnvinfer.so

/usr/bin/ld: 找不到 -lcublasLt
cmakelists.txt中添加：
link_directories( ${CUDA_Lib})
# libcublasLt.so

# 报错3  编译2次就好
Error generating
/root/share/bevfusion_ws/src/FastBEV-ROS-TensorRT/build/CMakeFiles/fastbev.dir/third_party/cuOSD/src/./fastbev_generated_cuosd_kernel.cu.o
~~~

# 可视化

~~~python
python tool/draw.py
~~~