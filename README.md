# Tensorflow C++ GPU API example and tutorial 

## Hardware configurations/platform(s) tested
1. OS:   Windows 10 (10.0.17134 N/A Build 17134)  
   CPU:	Intel(R) Core(TM) i7-8086K CPU @ 4.00GHz, 4008 Mhz, 6 Core(s), 12 Logical Processor(s)  
   GPU:	NVIDIA GeForce GTX 1070  
## Software configurations
Visual studio 2015 (14.0.25431 update3)  
Git 2.15.0  
SWIG 3.0.12  
Cmake 2.13.0 rc3
CUDA 9.0  
cuDNN 7.4.1
Anaconda 4.2.9
Tensorflow 1.5.0

## Step 1: build tensorflow C++ static library
Reference [SHI Weili@meidum](https://medium.com/@shiweili/building-tensorflow-c-shared-library-on-windows-e79c90e23e6e)[Joe Antognini@github](https://joe-antognini.github.io/machine-learning/build-windows-tf)
### 1.1 setup python enviroment
In *anaconda prompt*, execute the following code to setup a new enviroment with name "tf_cplusplus" with python 3.5, I recommend using **pip** to install numpy because sometimes **conda install** gives weird version error.
```
conda create -n tf_cplusplus pip python=3.5
activate tensorflow
conda install numpy
```
### 1.2 Acquire tensorflow 1.5.0 source code
In *vs2015 native tools command prompt* (elevated access recommended), do
```
git clone https://github.com/tensorflow/tensorflow.git v1.5.0
cd v1.5.0
git checkout tags/v1.5.0
```
### 1.3 Create working directory
Continue inputting the following code:
```
cd tensorflow\contrib\cmake
mkdir build
cd build
```
