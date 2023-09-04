侧信道分析接口源码为C++，使用python调用动态链接库libresreach3.so
## 编译
执行CMake_Research3下的./build.sh可生成动态链接库libresreach3.so，将Build下的libresreach3.so拷贝到CMake_Research3文件夹下，确保libresreach3.so与Trs文件夹在同一目录下

`cd CMake_Research3
./build.sh
cp Build/libresreach3.so ./`

## 文件目录介绍
Build --编译文件夹
Example --示例文件夹
Inc --接口文件夹，interface.h中记录所有对外的接口，使用extern暴露相关接口
Src --源码文件夹
Tests --测试用例文件夹
## python 调用 C++ 说明
call.py文件中，通过ctypes的对.so的文件进行实例化，使得c++中的方法可以在python引用，类比为一个函数，同时对这个实例化的方法进行输入输出的类型定义，使用ctypes中的类型去规定这个函数的输入输出接口