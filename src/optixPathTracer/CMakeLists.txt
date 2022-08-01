#
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  

#enable_language(CUDA)
 
#CUDA_ADD_EXECUTABLE(optixPathTracer 
#	hello_cuda.cu)
#file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
#file(GLOB CURRENT_SOURCES  *.cpp *.cu)

#MESSAGE("ERRRR")
#MESSAGE(CUDA_NVCC_FLAGS)
#find_package(CUDA)
#CUDA_ADD_EXECUTABLE(optixPathTracer 
#	hello_cuda.cu)

include_directories(${SAMPLES_INCLUDE_DIR})
include_directories(${IL_INCLUDE_DIR})

#add_subdirectory(cudaCode)
#include_directories(cudaCode)
# See top level CMakeLists.txt file for documentation of OPTIX_add_sample_executable.
OPTIX_add_sample_executable( optixPathTracer
    optixPathTracer.cpp
	sceneLoader.cpp
	Picture.cpp
	Texture.cpp
	sceneLoader.h
	material_parameters.h
	properties.h
	light_parameters.h
	state.h
	prd.h
	rt_function.h
	MyAssert.h
	Picture.h
	Texture.h
	BDPT.h
	BDPT_STRUCT.h
	select.h
	ZGC.h
	ZGC_common.h
	ZGC_device.h
	frame_estimator.h
	kd_tree.h
	BDenv_common.h
	BDenv_device.h
	BDenv_host.h
	subspace_common.h
	subspace_device.h
	subspace_host.h
	rmis.h
	BDPTVertex.h
	global_setting.h

	hello_cuda.h
	#hello_cuda.cu

	PG_common.h
	PG_device.h
	PG_host.h

	SVM_common.h
	SVM_device.h
	SVM_host.h

	classTree_common.h
	classTree_host.h
	classTree_device.h

	MLP_common.h
	MLP_host.h
	MLP_device.h
	
	gamma_common.h
	gamma_device.h
	gamma_host.h

	#hello_cuda.h 
	#hello_cuda.cu

    path_trace_camera.cu
    quad_intersect.cu
	disney.cu
	lambert.cu
	glass.cu
	hit_program.cu
	light_hit_program.cu
	light_sample.cu
	background.cu
    triangle_mesh.cu
	sphere_intersect.cu

    # common headers
    ${SAMPLES_INCLUDE_DIR}/commonStructs.h
    ${SAMPLES_INCLUDE_DIR}/helpers.h
    ${SAMPLES_INCLUDE_DIR}/random.h
    ) 


#message("message works")

#	target_sources(optixPathTracer PRIVATE hello_cuda.cu)
#add_executable(optixPathTracer 
#	hello_cuda.cu)
target_link_libraries(optixPathTracer PUBLIC 
cublas.lib
cuda.lib
cudadevrt.lib
cudart.lib
cudart_static.lib
cufft.lib
cufftw.lib
curand.lib
cusolver.lib
cusparse.lib
)

#target_link_libraries( optixPathTracer 
#hello_cuda
#)

	