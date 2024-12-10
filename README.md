# 简介
* python使用多线程异步推理rknn模型
* 在[rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)该项目基础上添加yolov8支持，添加rknn-toolkit2 v2.3.0支持，rknn driver v0.9.6

# 更新说明
* 2024-12-10 添加yolov8_seg支持


# 使用说明
### 演示
  * 板子上的python环境中配置好rknn-toolkit-lite2、opencv、numpy依赖
  * 将仓库拉取至板子上, 在项目根目录下存放一个测试视频(test.mp4), 运行python main.py
  * 运行rkcat.sh可以查看当前温度与NPU占用
### 部署应用
  * 修改main.py下的modelPath为你自己的模型所在路径
  * 修改main.py下的cap为你想要运行的视频/摄像头
  * 修改main.py下的TPEs为你想要的线程数

# Acknowledgements
* https://github.com/leafqycc/rknn-multi-threaded
* https://github.com/ultralytics/ultralytics
* https://github.com/airockchip/ultralytics_yolov8
* https://github.com/airockchip/rknn-toolkit2
* https://github.com/airockchip/rknn_model_zoo
