~~~python
# 1 拉取docker镜像
docker pull hylin123/conda-cuda11.3-cudnn8.6.0:v2.0

# 2 创建docker容器
sudo docker run -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" -v ~/Documents/share:/root/share --name bev hylin123/conda-cuda11.3-cudnn8.6.0:v2.0 /bin/bash

# 3 更新apt列表
apt update

# 4 激活虚拟环境
conda activate bev

# 5 查看bev环境中包
pip list

# 显示如下
Package           Version
----------------- --------------
numpy             1.24.4
Pillow            10.0.1
pip               23.2.1
setuptools        68.0.0
torch             1.10.0+cu113
torchaudio        0.10.0+rocm4.1
torchvision       0.11.0+cu113
typing_extensions 4.8.0
wheel             0.38.4
~~~

~~~python
# 1 安装 git
apt install git

# 2 下载源码   目前默认是dev2.1
git clone https://github.com/HuangJunJie2017/BEVDet.git

# 3 查看代码版本号 显示dev2.1版本
git checkout

# 4 安装openlib相关库 mmcv-full安装耗时比较长，只要鼠标没卡住都是正常
pip install mmcv-full==1.5.3 onnxruntime-gpu==1.8.1 mmdet==2.25.1 mmsegmentation==0.25.0

# 5 安装mmdet3d
pip install -e -v .

# 6 安装其他依赖
pip install pycuda lyft_dataset_sdk networkx==2.2 numba==0.53.0 numpy==1.23.4 nuscenes-devkit plyfile scikit-image tensorboard trimesh==2.35.39 -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~