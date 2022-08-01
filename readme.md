An unstable SPCBPT code

requirement：

* optix 5.0, incompatible with later versions
* cuda 10.0+
* Visual Studio 2015 (toolset v140)
* Cmake

How to Build:  

* Start up cmake-gui from the Start Menu.

* Select the "src" directory and the source code
* Create a build directory that isn't the same as the source directory. 
* Press "Configure" button and select the version of Visual Studio, our code is compatible with VS 2015 only (for its v140 toolset).
* Select "x64" as the platform
* Press "OK".
* Set OptiX_INSTALL_DIR to wherever you installed OptiX, e.g., C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.0.1
* Set CUDA_SDK_ROOT_DIR to the path to cuda samples, e.g., C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1
* Press "generate" and then "open Project"

I am struggling with Cmake. So you need to manually compile the hello_cuda.cu file currently. What you need to do is:

* Find the optixPathTracer project
  * Right click - Build dependencies - Build Customizations
  * Enable cuda
* Right click “Cuda Files” directory
  * Add-Existing item
  * find "hello_cuda.cu" in src/optixPathTracer directory and add it to the project
* Build the optixPathTracer project in release

Visual Studio may ask you to "Reload All" the project. Agree and do the above operations again to compile the "hello_cuda.cu" file.

Finally, you can Right click the optixPathTracer project and set it as Startup project and run the renderer program.

———————————————————————————————————————————————————————————————————————

The code corresponding to the key parts of our algorithms

## pre-processing

###  optixPathTracer.cpp: void pre_processing( )

* Determines $\Gamma$ and Q using **train_api.data.get_data**, measure the time for preprocessing, and transports $\Gamma$ and $Q$ to the device. The classification function $\kappa$ is determined and transferred to the device by **train_api.data.get_data**
  * **train_api.data.get_data**
    * Pre-traces the full paths used for trainning. We use LVCBPT for the pre-tracing but it may introduce some correlation problem and make a less effective $\Gamma$, we will consider replace it by a simple path tracer.
    * Determine the classification function $\kappa$ (and devide the sub-path space into subspaces thereby)
      * The labeled sub-paths used for decision tree come from **get_rough_weighted_sample**
        * Cut the prefix and suffix sub-paths by **classification_data_get_flat**
        * Sample the centroid sub-paths and label others by **HS_algorithm::stra_hs**, here we use a stratified sampled method to get the centroid sub-paths, but its optimization is slight compared to sampling the centroid based on the contribution directly.
      * Build the decision trees (with depth limitation of 12) and transfer ithem to the device
    * Label each each light sub-path, prefix and sufffix sub-path of full paths with its subspace ID
      * Get Q
      * Train the MIS-aware $\Gamma$
      * Trick: Starting training from the contribution-based $\Gamma$ can make a better result in the same time budget. 

* Light sub-path tracing

  * device
    * Function **light_vertex_launch** in path_trace_camera.cu
    * Trace the light sub-path in the GPU

  * host
    * Function **light_cache_process_ZGCBPT** in path_trace_camera.cu
    * Make statistics on necessary parameters
    * Build the light subspace for the second stage sampling

* Eye sub-path tracing

  * Function **ZGCBPT_pinhole_camera** in path_trace_camera.cu
  * Trace an eye sub-path for each pixel
  * Two stage sampling for each prefix eye sub-path 

* Hit program
  * hit program.cu
    * **BDPT_closest_hit** for eye sub-path
    * **BDPT_L_closest_hit** for light sub-path

* MIS computation
  * rmis.h
  * Use a RMIS style and is difficult to understand.



