# Tensorflow-GPU C++ API example and tutorial 
## Summary
Tested the capability of Tensorflow 1.10.0/1.5.0 running with C++ static library (build with CMake) in Visual Studio 2015.
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
Tensorflow 1.5.0/1.10.0  
Note that the version after 1.10.0 of tensorflow no longer supports CMake [source](https://github.com/tensorflow/tensorflow/issues/23679)


## Step 1: build tensorflow C++ static library
References:  [SHI Weili @ medium](https://medium.com/@shiweili/building-tensorflow-c-shared-library-on-windows-e79c90e23e6e) and [Joe Antognini @ github](https://joe-antognini.github.io/machine-learning/build-windows-tf)
### 1.1 Setup python environment
In *anaconda prompt*, execute the following code to setup a new environment with name "tf_cplusplus" with python 3.5, I recommend using **pip** to install numpy because sometimes **conda install** gives weird version error.
```
conda create -n tf_cplusplus pip python=3.5
activate tf_cplusplus
pip install numpy
```
### 1.2 Acquire tensorflow 1.5.0 source code
In *vs2015 native tools command prompt* (elevated access recommended), for tensorflow (TF) 1.5.0, do
```
git clone https://github.com/tensorflow/tensorflow.git v1.5.0
cd v1.5.0
git checkout tags/v1.5.0
```

For 1.10.0, do
```
git clone https://github.com/tensorflow/tensorflow.git v1.10.0
cd v1.10.0
git checkout tags/v1.10.0
```
### 1.3 Create working directory
Continue executing the following code in *vs command prompt*:
```
cd tensorflow\contrib\cmake
mkdir build
cd build
```
### 1.4 Setup CMake
Continue executing the following code in *vs command prompt*, be aware that the location of the swig, python environment, CUDA, and vs installation location may vary on your station, **please change those locations**, for TF 1.5.0, do:
```
cmake .. -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-DSWIG_EXECUTABLE=C:\swigwin-3.0.12\swig.exe ^
-DPYTHON_EXECUTABLE=C:\Users\xzhou\AppData\Local\Continuum\Anaconda3\envs\cplusplus_tf\python.exe ^
-DPYTHON_LIBRARIES=C:\Users\xzhou\AppData\Local\Continuum\Anaconda3\envs\cplusplus_tf\libs\python35.lib ^
-Dtensorflow_ENABLE_GPU=ON ^
-DCUDNN_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0" ^
-Dtensorflow_BUILD_PYTHON_BINDINGS=OFF ^
-Dtensorflow_ENABLE_GRPC_SUPPORT=OFF ^
-Dtensorflow_BUILD_SHARED_LIB=ON ^
-DCUDA_HOST_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/amd64/cl.exe" ^
```
for TF 1.10.0, do:
```
cmake .. -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-DPYTHON_EXECUTABLE=C:\Users\xzhou\AppData\Local\Continuum\Anaconda3\envs\cplusplus_tf\python.exe ^
-Dtensorflow_ENABLE_GPU=ON ^
-DCUDNN_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0" ^
-Dtensorflow_BUILD_PYTHON_BINDINGS=OFF ^
-Dtensorflow_ENABLE_GRPC_SUPPORT=ON ^
-Dtensorflow_BUILD_SHARED_LIB=ON ^
-DCUDA_HOST_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/amd64/cl.exe" ^
-DCUDA_SDK_ROOT_DIR="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0"
```
Notes: I tried to build TF 1.10.0 with -Dtensorflow_ENABLE_GRPC_SUPPORT=OFF, error of “grpcpp/grpcpp.h”: No such file or directory was given, could be a bug [issue](https://github.com/tensorflow/tensorflow/issues/19893) and [potential solution in Chinese](https://blog.csdn.net/whunamikey/article/details/82143334). But the option of GRPC ON works for me, so I did not proceed.

### 1.5 Build tensorflow
The next step would be actually building tensorflow, it is recommended to disable parallelism to minimize the chance of running of heap spaces by setting m to 1. You should also consider to increase the virtual memory of your machine to over 15G. Continue executing the following code in *vs command prompt* (for both TF versions):
```
“C:\Program Files (x86)\MSBuild\14.0\Bin\amd64\MSBuild.exe” ^
/m:1 ^
/p:CL_MPCount=1 ^
/p:Configuration=Release ^
/p:Platform=x64 ^
/p:PreferredToolArchitecture=x64 ALL_BUILD.vcxproj ^
/filelogger
```
The whole building process might take more than 2 hours to finish. If success, it should come with no error and lots of warnings.
### 1.6 Test tensorflow
In any command propmt, locate the directory of {tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\Release and run 
```
./tf_tutorials_example_trainer.exe
```
You should have results like
```
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2019-01-09 16:31:40.067433: I D:\DSM_project\DSM_dependencies\v1.5.0\tensorflow\core\common_runtime\gpu\gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
000000/000001 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000009 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000000 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000006 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000007 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000005 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000002 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000008 lambda = 2.283615 x = [0.773146 0.634228] y = [3.587894 -0.773146]
000000/000000 lambda = 2.660952 x = [0.977561 -0.210652] y = [2.511379 -0.977561]
000000/000001 lambda = 2.660952 x = [0.977561 -0.210652] y = [2.511379 -0.977561]
```
which indicates that tensorflow with GPU works in your station, the next step would be building your own project in a seperate directory.

## Step 2: Building a standalone C++ Tensorflow-GPU program
### 2.1 Create a visual studio C++ console project
Make sure the configuration is x64 and release
### 2.2 Source code
Refer to [Tebesu @ github](https://tebesu.github.io/posts/Training-a-TensorFlow-graph-in-C++-API), you can test your program with the following source code with a pre-trained file "mlp.pb" (available on [github link](https://github.com/tebesu/Tensorflow-Cpp-API-Training)):
```
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
using namespace tensorflow;

int main(int argc, char* argv[]) {

    std::string graph_definition = "mlp.pb";
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;
    std::vector<Tensor> outputs; // Store outputs
    TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

    // Set GPU options
    graph::SetDefaultDevice("/gpu:0", &graph_def);
    opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
    opts.config.mutable_gpu_options()->set_allow_growth(true);

    // create a new session
    TF_CHECK_OK(NewSession(opts, &session));

    // Load graph into session
    TF_CHECK_OK(session->Create(graph_def));

    // Initialize our variables
    TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

    Tensor x(DT_FLOAT, TensorShape({100, 32}));
    Tensor y(DT_FLOAT, TensorShape({100, 8}));
    auto _XTensor = x.matrix<float>();
    auto _YTensor = y.matrix<float>();

    _XTensor.setRandom();
    _YTensor.setRandom();

    for (int i = 0; i < 10; ++i) {

        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs)); // Get cost
        float cost = outputs[0].scalar<float>()(0);
        std::cout << "Cost: " <<  cost << std::endl;
        TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr)); // Train
        outputs.clear();
    }


    session->Close();
    delete session;
    return 0;
}
```
### 2.3 Setup additional include directories
The header files you needed would be located on the following locations, make sure they are all included, please be noted **the location on your station may vary**, for 1.10.0, change those directories accordingly: 

{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\protobuf\src\protobuf\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc_ops.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc_framework.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_cpu.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_direct_session.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_framework.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_kernels.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_lib.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_ops.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc_while_loop.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\nsync\src\nsync\Release;

### 2.4 Setup linked libraries
The needed libraries are listed as follows:
```
zlib\install\lib\zlibstatic.lib
gif\install\lib\giflib.lib
png\install\lib\libpng12_static.lib
jpeg\install\lib\libjpeg.lib
lmdb\install\lib\lmdb.lib
jsoncpp\src\jsoncpp\src\lib_json\$(Configuration)\jsoncpp.lib
farmhash\install\lib\farmhash.lib
fft2d\\src\lib\fft2d.lib
highwayhash\install\lib\highwayhash.lib
nsync\install\lib\nsync.lib
protobuf\src\protobuf\$(Configuration)\libprotobuf.lib
re2\src\re2\$(Configuration)\re2.lib
sqlite\install\lib\sqlite.lib
snappy\src\snappy\$(Configuration)\snappy.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cudart_static.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cuda.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cublas.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cublas_device.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cufft.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\curand.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64\cupti.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cusolver.lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\cudnn.lib
tf_cc.lib
tf_cc_ops.lib
tf_cc_framework.lib
tf_core_direct_session.lib
tf_core_framework.lib
tf_core_kernels.lib
tf_core_lib.lib
tf_core_ops.lib
tf_core_cpu.lib
tf_protos_cc.lib
tf_core_gpu_kernels.lib
kernel32.lib
user32.lib
gdi32.lib
winspool.lib
shell32.lib
ole32.lib
oleaut32.lib
uuid.lib
comdlg32.lib
advapi32.lib
wsock32.lib
ws2_32.lib
shlwapi.lib
snappy\src\snappy\Release\snappy.lib
tf_cc_while_loop.dir\Release\tf_cc_while_loop.lib
tf_checkpoint_ops.dir\Release\tf_checkpoint_ops.lib
tf_io_ops.dir\Release\tf_io_ops.lib
tf_ctc_ops.dir\Release\tf_ctc_ops.lib
tf_data_flow_ops.dir\Release\tf_data_flow_ops.lib
tf_array_ops.dir\Release\tf_array_ops.lib
tf_stream_executor.dir\Release\tf_stream_executor.lib
```
which located in the following *Additional library directories*:

{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\protobuf\src\protobuf\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc_ops.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc_framework.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_cpu.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_direct_session.dir\Release;   {tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_framework.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_kernels.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_lib.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_core_ops.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\tf_cc_while_loop.dir\Release;  
{tensorflow}\v1.5.0\tensorflow\contrib\cmake\build\nsync\src\nsync\Release;  

2.4 Setup command line options

[why do I need to setup them](https://github.com/tensorflow/tensorflow/issues/4242#issuecomment-245436151)  [another explanation](https://joe-antognini.github.io/machine-learning/windows-tf-project)
```
/machine:x64
/ignore:4049 /ignore:4197 /ignore:4217 /ignore:4221
/WHOLEARCHIVE:tf_cc.lib
/WHOLEARCHIVE:tf_cc_framework.lib
/WHOLEARCHIVE:tf_cc_ops.lib
/WHOLEARCHIVE:tf_core_cpu.lib
/WHOLEARCHIVE:tf_core_direct_session.lib
/WHOLEARCHIVE:tf_core_framework.lib
/WHOLEARCHIVE:tf_core_kernels.lib
/WHOLEARCHIVE:tf_core_lib.lib
/WHOLEARCHIVE:tf_core_ops.lib   
/WHOLEARCHIVE:tf_stream_executor.lib
/WHOLEARCHIVE:libjpeg.lib
/FORCE:MULTIPLE
```

2.5 Compile and run the program
You should have a similar result as mine:
```
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 8.00GiB freeMemory: 6.63GiB
2019-01-09 17:00:29.682091: I D:\DSM_project\DSM_dependencies\v1.5.0\tensorflow\core\common_runtime\gpu\gpu_device.cc:1195] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
Cost: 264.194
Cost: 256.987
Cost: 249.912
Cost: 242.962
Cost: 236.132
Cost: 229.422
Cost: 222.836
Cost: 216.377
Cost: 210.05
Cost: 203.86
```
