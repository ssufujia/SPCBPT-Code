/* 
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixPathTracer: A path tracer using the disney brdf.
//
//-----------------------------------------------------------------------------

#ifndef __APPLE__
#  include <GL/glew.h>
#  if defined( _WIN32 )
#    include <GL/wglew.h>
#  endif
#endif

#include <GLFW/glfw3.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "commonStructs.h"
#include "sceneLoader.h"
#include "light_parameters.h"
#include "properties.h"
#include <IL/il.h>
#include <Camera.h>
#include <OptiXMesh.h>
#include"BDPT_STRUCT.h"
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include<random>
#include<fstream>
#include"select.h"
#include "ZGC.h"
#include"frame_estimator.h"
#include"kd_tree.h"
#include"BDenv_host.h"
#include"subspace_host.h"
#include<set>
#include"PG_host.h"
#include"SVM_host.h"
#include"classTree_host.h"
#include"MLP_host.h"
#include"hello_cuda.h"
#include"gamma_host.h"

using namespace optix;
using std::default_random_engine;

const char* const SAMPLE_NAME = "optixPathTracer";

const int NUMBER_OF_BRDF_INDICES = 3;
const int NUMBER_OF_LIGHT_INDICES = LightTypeNum;
optix::Buffer m_bufferBRDFSample;
optix::Buffer m_bufferBRDFEval;
optix::Buffer m_bufferBRDFPdf;

optix::Buffer m_bufferLightSample;
optix::Buffer m_bufferMaterialParameters;
optix::Buffer m_bufferLightParameters;
Buffer denoisedBuffer;
Buffer emptyBuffer;

default_random_engine random_generator;
CommandList denoise_command;
CommandList heat_command;
PostprocessingStage denoiserStage;
PostprocessingStage tonemapStage;
PostprocessingStage heatStage;
DivTris div_tris;
lightSelectionFunction_host gamma;
double elapsedTime = 0;
double lastTime = 0;
double subspaceDivTime;
int num_triangles = 0;
int num_samplers = 0;
bool standrdImageIsLoad = false;
bool postprocessing_needs_init = true;
//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------
Properties properties;
Context      context = 0;
Scene* scene;
Aabb scene_aabb;

//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

template<typename T>
void randomSelectVector(std::vector<T>& v, int selectNum); 
int buildVDirectorBuffer();
void buildPhotonMap(Buffer& LTVBuffer, Buffer& KDBuffer, int kd_n);
void minProjectProcess(Context& context, float3 dir, float3 min, float3 max)
{
    std::vector<float3> points, project2;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                float3& a = i == 0 ? min : max;
                float3& b = j == 0 ? min : max;
                float3& c = k == 0 ? min : max;
                points.push_back(make_float3(a.x, b.y, c.z));
            }
        }
    }
    auto onb = optix::Onb(dir);
    Aabb aabb;
    for (auto p = points.begin(); p != points.end(); p++)
    {
        aabb.include(make_float3(dot(onb.m_binormal, *p), dot(onb.m_tangent, *p), 0.0f));
    }
    context["project_box_max"]->setFloat(aabb.m_max);
    context["project_box_min"]->setFloat(aabb.m_min);
    float3 aDelta = (aabb.m_max - aabb.m_min).x * onb.m_binormal;
    float3 bDelta = (aabb.m_max - aabb.m_min).y * onb.m_tangent;
    context["DirProjectionArea"]->setFloat(length(cross(aDelta, bDelta)));

}
void context_initial_value_set()
{
    context["EVC_frame"]->setUint(0);
}
void frame_estimate(int flag = 0)
{
#ifdef ESTIMATE_INVALID
    return;
#endif // ESTIMATE_INVALID

    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();

    if (flag == 1  &&false )
    {
        int current_frame = context["frame"]->getUint();
        int estimate_frame_array[100] = ESTIMATE_FRAME;
        for (int i = 0; i < 100; i++)
        {
            if (estimate_frame_array[i] == 0)
                return;
            else if (estimate_frame_array[i] == current_frame)
                break;
            else
                continue;
        }
    }    
    RTsize OW, OH;
    context["output_buffer"]->getBuffer()->getSize(OW, OH);
    if (OW != 1920 ||OH != 1001)
    {
        std::cout << "estimate fall for wrong screen size" << std::endl;
        return;
    }
    char4* p1 = reinterpret_cast<char4*>(context["output_buffer"]->getBuffer()->map());
    char4* p2 = reinterpret_cast<char4*>(context["standrd_buffer"]->getBuffer()->map());

    float error = 0.0f;
    float re_error = 0.0f;
    if (!standrdImageIsLoad)
    {
        int i = 0;
        int a, b, c, d;
        std::ifstream inFile;
        inFile.open(REFERENCE_FILE_PATH);
        while (inFile >> a >> b >> c >> d)
        {
            p2[i].x = char(a);
            p2[i].y = char(b);
            p2[i].z = char(c);
            p2[i].w = char(d);
            i++;
        }
        standrdImageIsLoad = true;
        inFile.close();
    }
    for (int i = 0; i < OW * OH; i++)
    {
        float de = 0.0f;
        int a, b, c, d;
        a = p1[i].x - p2[i].x;
        b = p1[i].y - p2[i].y;
        c = p1[i].z - p2[i].z;
        error += float(a*a + b * b + c * c) / 256 / 256;

        float ra, rb, rc;
        ra = p2[i].x != 0 ? abs(a / p2[i].x) : 0;
        rb = p2[i].y != 0 ? abs(b / p2[i].y) : 0;
        rc = p2[i].z != 0 ? abs(c / p2[i].z) : 0;
        float t = (ra + rb + rc) / 3;

    }
    error /= OW * OH;
    re_error /= OW * OH;
    //std::cout << "mse and mape in frame " << context["frame"]->getUint()
    //    << " and time " << elapsedTime << " is " << error << " and " << re_error * 100 << "%" << std::endl;

    std::cout << "" << context["frame"]->getUint()
        << "  " << elapsedTime << " " << error << " " << re_error * 100 << "%" << std::endl;
    context["standrd_buffer"]->getBuffer()->unmap();
    context["output_buffer"]->getBuffer()->unmap();

    sutil::writeBufferToFile("./theImageYouLoad.png", context["standrd_buffer"]->getBuffer());
    
    lastTime = sutil::currentTime();
}
void frame_estimate_float(int flag = 0)
{
#ifdef ESTIMATE_INVALID
    if (!standrdImageIsLoad)
    {
        float4* p2 = reinterpret_cast<float4*>(context["standrd_float_buffer"]->getBuffer()->map());
        int i = 0;
        float a, b, c, d;
        std::ifstream inFile;
        inFile.open(REFERENCE_FILE_PATH);
        while (inFile >> a >> b >> c >> d)
        {

            p2[i] = optix::make_float4(a, b, c, d);

            float lum = 0.3 * a + 0.6 * b + 0.1 * c;
            float limit = 1.5;
            float4 tone = p2[i] * 1.0f / (1.0f + lum / limit);
            float kInvGamma = 1.0f / 2.2f;
            float3 gamma_color = make_float3(pow(tone.x, kInvGamma), pow(tone.y, kInvGamma), pow(tone.z, kInvGamma));
            gamma_color.x = fminf(1.0f, gamma_color.x);
            gamma_color.y = fminf(1.0f, gamma_color.y);
            gamma_color.z = fminf(1.0f, gamma_color.z); 
            i++;

        }

        standrdImageIsLoad = true;
        inFile.close();
        context["standrd_float_buffer"]->getBuffer()->unmap();
    }
    return;
#endif // ESTIMATE_INVALID

    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();

    if (flag == 1 && false)
    {
        int current_frame = context["frame"]->getUint();
        int estimate_frame_array[100] = ESTIMATE_FRAME;
        for (int i = 0; i < 100; i++)
        {
            if (estimate_frame_array[i] == 0)
                return;
            else if (estimate_frame_array[i] == current_frame)
                break;
            else
                continue;
        }
    }
    RTsize OW, OH;
    context["standrd_float_buffer"]->getBuffer()->getSize(OW, OH);
    if (OW != 1920 || OH != 1001)
    {
        std::cout << "estimate fall for wrong screen size" << std::endl;
        return;
    }
    float4* p1 = reinterpret_cast<float4*>(context["accum_buffer"]->getBuffer()->map());
    float4* p2 = reinterpret_cast<float4*>(context["standrd_float_buffer"]->getBuffer()->map());
    uchar4*  p3 = reinterpret_cast<uchar4*>(context["standrd_buffer"]->getBuffer()->map());
    float error = 0.0f;
    float re_error = 0.0f; 
    float mae = 0.0f;
    if (!standrdImageIsLoad)
    {
        int i = 0;
        float a, b, c, d;
        std::ifstream inFile;
        inFile.open(REFERENCE_FILE_PATH);
        while (inFile >> a >> b >> c >> d)
        {
            
            p2[i] = optix::make_float4(a, b, c, d);
            
            float lum = 0.3 * a + 0.6 * b + 0.1 * c;
            float limit = 1.5;
            float4 tone = p2[i] * 1.0f / (1.0f + lum / limit);
            float kInvGamma = 1.0f / 2.2f;
            float3 gamma_color = make_float3(pow(tone.x, kInvGamma), pow(tone.y, kInvGamma), pow(tone.z, kInvGamma));
            gamma_color.x = fminf(1.0f, gamma_color.x);
            gamma_color.y = fminf(1.0f, gamma_color.y);
            gamma_color.z = fminf(1.0f, gamma_color.z);
            uchar4 color = make_uchar4(uint(gamma_color.z * 255), uint(gamma_color.y * 255), uint(gamma_color.x * 255), 255);
            p3[i] = color;
            i++;
             
        }
        
        standrdImageIsLoad = true;
        inFile.close();
    } 
    for (int i = 0; i < OW * OH  ; i++)
    {
        float minLimit = 0.000000001;
        float3 a = make_float3(p1[i]);
        float3 b = make_float3(p2[i]);
        float3 bias = a - b;
        float3 r_bias = (a - b) / (b + make_float3(minLimit));
        float diff = length(a-b) / 3.0;
        float diff_b = length((a-b) / (b + make_float3(minLimit))) / 3; 
        float diff_c = (abs(bias.x) + abs(bias.y) + abs(bias.z)) / 3;
        float diff_d = (abs(r_bias.x) + abs(r_bias.y) + abs(r_bias.z)) / 3;
        //warning : this option may be buggy
        if ((b.x + b.y + b.z)>5)
        {
            diff = diff_b = diff_c = diff_d = 0;

        }
        
        error += diff * diff;
       // re_error += diff_b * diff_b;
        re_error += diff_d;// < 1.0 ? diff_d : 1.0;
        mae += diff_c;
    }
    error /= OW * OH;
    re_error /= OW * OH;
    mae /= OW * OH;
    float rmae = sqrtf(mae);
    //re_error = sqrtf(re_error);
    //std::cout << "mse and mape in frame " << context["frame"]->getUint()
    //    << " and time " << elapsedTime << " is " << error << " and " << re_error * 100 << "%" << std::endl;

    std::cout << "" << context["frame"]->getUint()
        << "  " << elapsedTime << " " << error << " "<<rmae <<" " << re_error * 100 << "%" << std::endl;
    context["standrd_buffer"]->getBuffer()->unmap();
    context["accum_buffer"]->getBuffer()->unmap();
    context["standrd_float_buffer"]->getBuffer()->unmap();

    sutil::writeBufferToFile("./theImageYouLoad.png", context["standrd_buffer"]->getBuffer());

    lastTime = sutil::currentTime();
}
void postprocessing_init(sutil::Camera &camera)
{
    if (!postprocessing_needs_init)
        return;
    postprocessing_needs_init = false;
    if (denoise_command)
        denoise_command->destroy();
    denoiserStage["input_albedo_buffer"]->set(context["input_albedo_buffer"]->getBuffer());
    denoiserStage["input_normal_buffer"]->set(context["input_normal_buffer"]->getBuffer());

    denoise_command = context->createCommandList();
    denoise_command->appendLaunch(pinholeCamera, camera.width(), camera.height());
    //denoise_command->appendPostprocessingStage(tonemapStage, camera.width(), camera.height());
    denoise_command->appendPostprocessingStage(denoiserStage, camera.width(), camera.height());
    denoise_command->finalize();

    heat_command = context->createCommandList();
    heat_command->appendLaunch(pinholeCamera, camera.width(), camera.height());
    //heat_command->appendPostprocessingStage(tonemapStage, camera.width(), camera.height());
    //heat_command->appendPostprocessingStage(heatStage, int(ceilf(camera.width() / 16.0)), int(ceilf(camera.height() / 16.0)));
    heat_command->finalize();
    
}
static std::string ptxPath( const std::string& cuda_file )
{
    return
        std::string(sutil::samplesPTXDir()) +
        "/" + std::string(SAMPLE_NAME) + "_generated_" +
        cuda_file +
        ".ptx";
}
template<typename T>
void randomSelectVector(std::vector<T>& v,int selectNum)
{
    if (selectNum >= v.size())
    {
        return ;
    }
    for (int i = 0; i < selectNum; i++)
    {
        int rndInt = random_generator() % (v.size() - i) + i;
        T tmp = v[rndInt];
        v[i] = v[rndInt];
        v[rndInt] = tmp;
    }
}
template<typename T>
void buildKDTree(T** directors, int start, int end, int depth, T* kd_tree, int current_root,
    splitChoice split_choice, float3 bbmin, float3 bbmax);
optix::GeometryInstance createSphere(optix::Context context,
	optix::Material material,
	float3 center,
	float radius)
{
	optix::Geometry sphere = context->createGeometry();
	sphere->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("sphere_intersect.cu");
	sphere->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	sphere->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "sphere_intersect_robust"));

	sphere["center"]->setFloat(center);
	sphere["radius"]->setFloat(radius);

	optix::GeometryInstance instance = context->createGeometryInstance(sphere, &material, &material + 1);
	return instance;
}

optix::GeometryInstance createQuad(optix::Context context,
	optix::Material material,
	float3 v1, float3 v2, float3 anchor, float3 n)
{
	optix::Geometry quad = context->createGeometry();
	quad->setPrimitiveCount(1u);
	const std::string ptx_path = ptxPath("quad_intersect.cu");
	quad->setBoundingBoxProgram(context->createProgramFromPTXFile(ptx_path, "bounds"));
	quad->setIntersectionProgram(context->createProgramFromPTXFile(ptx_path, "intersect2"));

	float3 normal = normalize(cross(v1, v2));
	float4 plane = make_float4(normal, dot(normal, anchor));
    float3 r_v = v1, r_u=v2;
    v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	quad["v1"]->setFloat(v1);
	quad["v2"]->setFloat(v2);
	quad["anchor"]->setFloat(anchor);
	quad["plane"]->setFloat(plane);

	optix::GeometryInstance instance = context->createGeometryInstance(quad, &material, &material + 1);
	return instance;
}


static Buffer getOutputBuffer()
{
    return context[ "output_buffer" ]->getBuffer();
}

void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}
bool zoneLVCInitFlag = false;

void zoneAlloc()
{

    int zoneSum = SUBSPACE_NUM - MAX_LIGHT;
    /*visibility test*/
    triangleStruct *triangleBufferP = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
    triangleStruct *triangleTargetP = reinterpret_cast<triangleStruct *>(context["triangle_targets"]->getBuffer()->map());
    float sceneArea = 0.0f;
    int areaDivPoint_vis[VISIBILITY_TEST_NUM];
    float targetStep = float(num_triangles - 1) / VISIBILITY_TEST_NUM;
    for (int i = 0; i < num_triangles; i++)
    {
        triangleStruct &tmp = triangleBufferP[i];
        sceneArea += tmp.area();
    }
    {
        //按面积为依据，平均选取出vis的flag点和最初聚类点
        float currentArea = 0.0f;
        int vis_step = 0;
        int zone_step = 0;
        for (int i = 0; i < num_triangles; i++)
        {
            triangleStruct &tmp = triangleBufferP[i];
            currentArea += tmp.area();
            while (currentArea > sceneArea * vis_step / VISIBILITY_TEST_NUM && vis_step < VISIBILITY_TEST_NUM)
            {
                areaDivPoint_vis[vis_step] = i;
                vis_step++;
            }
        }
        vis_step++;
    }
    for (int i = 0; i < VISIBILITY_TEST_NUM; i++)
    {
        triangleTargetP[i] = triangleBufferP[areaDivPoint_vis[i]];
    }
    context["triangle_targets"]->getBuffer()->unmap();
    context["triangle_samples"]->getBuffer()->unmap();
    for (int i = 0; i < VISIBILITY_TEST_SLICE; i++)
    {
#ifdef RAW_CLUSTER
        break;
#endif // RAW_CLUSTER
        context["frame"]->setUint(i);
        context->launch(visibilityTestProg, VISIBILITY_TEST_NUM, num_triangles);

    }
    context["frame"]->setUint(0);
    triangleBufferP = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
    float * visibilityP = reinterpret_cast<float *>(context["visibility_buffer"]->getBuffer()->map());


    context["visibility_buffer"]->getBuffer()->unmap();
#ifdef RAW_CLUSTER
    context["triangle_samples"]->getBuffer()->unmap();
    return;
#endif // RAW_CLUSTER

    std::ofstream saveFile;
    saveFile.open("zone_Alloc.txt");
    for (int i = 0; i < num_triangles; i++)
    {
        saveFile << triangleBufferP[i].zoneNum << " ";
    }
    saveFile.close();
    context["triangle_samples"]->getBuffer()->unmap();
    return;

}
void zoneRawAlloc()
{
    if (LABEL_BY_STREE)return;
    int triZoneSum = SUBSPACE_NUM - MAX_LIGHT;
    triangleStruct *p = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
    ZoneMatrix *M2 = reinterpret_cast<ZoneMatrix *>(context["M2_buffer"]->getBuffer()->map());
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        M2[i].area = 0;
    }
    float area = 0.0f;
    for (int i = 0; i < num_triangles; i++)
    {
        float3 v1 = p[i].position[0] - p[i].position[1];
        float3 v2 = p[i].position[0] - p[i].position[2];
        area += length(cross(v1, v2)) / 2;
    } 
    float area_c = 0.0f;
    for (int i = 0; i < num_triangles; i++)
    {
        p[i].zoneNum = int(area_c / area * triZoneSum);
        p[i].zoneNum = p[i].zoneNum == triZoneSum ? triZoneSum - 1 : p[i].zoneNum;
        float3 v1 = p[i].position[0] - p[i].position[1];
        float3 v2 = p[i].position[0] - p[i].position[2];
        area_c += length(cross(v1, v2)) / 2; 
        M2[p[i].zoneNum].area += length(cross(v1, v2)) / 2;
    }
    printf("real area %f\n", area);
    context["triangle_samples"]->getBuffer()->unmap();
    context["M2_buffer"]->getBuffer()->unmap();
}
float triDiff(rawTriangle &a, rawTriangle &b)
{
#define objectDiff 1000.0f

    static float scene_max_dis = length(scene_aabb.m_max - scene_aabb.m_min);
    float3 diffPos = a.center - b.center;
    float ds = 2 * length(diffPos) / scene_max_dis;

    float3 diffNormal = a.normal - b.normal;
    float dc = length(diffNormal);

    float dobject = a.objectId == b.objectId ? 0 : objectDiff;
    float ans = sqrt(dc * dc / b.m / b.m + ds * ds + dobject * dobject);
    if (isnan(ans))
    {
        return 100000.0f;
    }
    return ans;
}
void zoneSLICAlloc()
{
    int zoneSum = SUBSPACE_NUM - MAX_LIGHT;
    int evc_width = sqrt(zoneSum);

    {
        context["EVC_width"]->setUint(evc_width);
        context["EVC_height"]->setUint(evc_width);
        context["EVC_mex_depth"]->setUint(10);
        context->launch(EVCLaunch, evc_width, evc_width);
    }

    triangleStruct *triangleBufferP = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
    rawTriangle *p = new rawTriangle[num_triangles];
    rawTriangle *t = new rawTriangle[zoneSum];

    float scene_area = 0.0f;
    for (int i = 0; i < num_triangles; i++)
    {
        p[i].center = make_float3(0.0f);
        for (int j = 0; j < 3; j++)
        {
            p[i].position[j] = triangleBufferP[i].position[j];
            p[i].center += triangleBufferP[i].position[j] / 3;
        }
        p[i].normal = normalize(
            cross(
                p[i].position[0] - p[i].position[1], p[i].position[0] - p[i].position[2]
            ));
        p[i].objectId = triangleBufferP[i].objectId;
        p[i].area = length(cross(
            p[i].position[0] - p[i].position[1], p[i].position[0] - p[i].position[2]
        )) / 2;

        scene_area += p[i].area;
    }

    // random Sample
    if (false)
    {
        float c_area = 0.0f;
        int t_index = 0;
        for (int i = 0; i < num_triangles; i++)
        {
            int z0 = c_area / scene_area * zoneSum;
            c_area += p[i].area;
            int z1 = c_area / scene_area * zoneSum;
            if (z1 != z0)
            {
                t[t_index] = p[i];
                t_index++;
            }
        }
        printf("%d  sssdasd\n", zoneSum - t_index);
        for (; t_index < zoneSum; t_index++)
        {
            t[t_index] = p[random_generator() % num_triangles];
        }
    }
    else
    {
        int ZoneT1 = zoneSum * 3 / 4;
        RAWVertex * vertex_cache = reinterpret_cast<RAWVertex*> (context["raw_LVC"]->getBuffer()->map());
        std::vector<rawTriangle> initPoint;
        for (int i = 0; i < evc_width *evc_width * 10; i++)
        {
            if (vertex_cache[i].valid)
            {
                rawTriangle z;
                z.objectId = vertex_cache[i].v.zoneId;
                z.normal = vertex_cache[i].v.normal;
                z.center = vertex_cache[i].v.position;
                initPoint.push_back(z);
                vertex_cache[i].valid = false;
            }
        }
        if (initPoint.size() > ZoneT1)
        {
            for (int i = 0; i < ZoneT1; i++)
            {
                t[i] = initPoint[i * initPoint.size() / ZoneT1];
            }
        }
        int ZoneT2 = zoneSum - min(ZoneT1, initPoint.size());
        printf("%d\n", ZoneT2);
        for (int i = ZoneT1; i < zoneSum; i++)
        {
            t[i] = p[random_generator() % num_triangles];
        }

        context["raw_LVC"]->getBuffer()->unmap();
    }

    for (int i = 0; i < zoneSum; i++)
    {
        t[i].m = 1.0f;
    }
    for (int k = 0; k < KMEANS_ITER_NUM; k++)
    {
        for (int i = 0; i < num_triangles; i++)
        {
            int zone_id = 0;
            float min_dis = triDiff(p[i], t[0]);
            for (int j = 0; j < zoneSum; j++)
            {
                float dis = triDiff(p[i], t[j]);
                if (dis < min_dis)
                {
                    min_dis = dis;
                    zone_id = j;
                }
            }
            p[i].zoneNum = zone_id;
        }
        for (int i = 0; i < zoneSum; i++)
        {
            t[i].area = 0;
            t[i].center = make_float3(0.0f);
            t[i].normal = make_float3(0.0f);
        }
        for (int i = 0; i < num_triangles; i++)
        {
            int zone_id = p[i].zoneNum;
            t[zone_id].area += p[i].area;
            t[zone_id].center += p[i].area * p[i].center;
            t[zone_id].normal += p[i].normal * p[i].area;
            t[zone_id].n_max = fmaxf(t[zone_id].n_max, p[i].normal);
            t[zone_id].n_min = fminf(t[zone_id].n_min, p[i].normal);
        }
        for (int i = 0; i < zoneSum; i++)
        {
            if (t[i].area < DBL_EPSILON)
                continue;
            t[i].center /= t[i].area;
            t[i].normal = normalize(t[i].normal);
            t[i].m = max(length(t[i].n_max - t[i].n_min), 0.1f);
        }
    }
    for (int i = 0; i < num_triangles; i++)
    {
        triangleBufferP[i].zoneNum = p[i].zoneNum;
    }
    delete[] p;
    delete[] t;

    std::ofstream saveFile;
    saveFile.open("zone_Alloc.txt");
    for (int i = 0; i < num_triangles; i++)
    {
        saveFile << triangleBufferP[i].zoneNum << " ";
    }
    saveFile.close();
    context["triangle_samples"]->getBuffer()->unmap();
    return;
}

static float BU_maxdiff = 1000.0f;
struct BU_cluster
{
    optix::float3 p_min;
    optix::float3 p_max;
    optix::float3 n_min;
    optix::float3 n_max;
    int left_child;
    int right_child;
    triangleStruct *origin;
    float area;
    int objectId;
    BU_cluster() {}
    BU_cluster(triangleStruct &a)
    {
        p_min = fminf(a.position[0], fminf(a.position[1], a.position[2]));
        p_max = fmaxf(a.position[0], fmaxf(a.position[1], a.position[2]));
        float3 n = normalize(cross(a.position[0] - a.position[1], a.position[0] - a.position[2]));
        n_min = n_max = n;
        area = length(cross(a.position[0] - a.position[1], a.position[0] - a.position[2])) / 2;
        if (isnan(area))
        {
            area = 0.001f;
        }
        origin = &a;
        left_child = right_child = -1;
    }
    BU_cluster combine(BU_cluster &b)
    {

        BU_cluster c;
        c.p_min = fminf(b.p_min, p_min);
        c.p_max = fmaxf(b.p_max, p_max);
        c.n_min = fminf(b.n_min, n_min);
        c.n_max = fmaxf(b.n_max, n_max);
        c.area = area + b.area;
        c.objectId = objectId;
        return c;
    }
    float cost()
    {
        float c = 10.0f;
        float d = 100.0f;
        float3 mid = normalize(n_min + n_max);
        float beta = dot(mid, normalize(n_min));
        float dig = length(p_max - p_min);

        beta = length(n_max - n_min);
        float AL_ratio = sqrt(area / (dig*dig)) + 0.1;
        return d / AL_ratio + c * (1 - beta);
    }
    float diff(BU_cluster &b)
    {
        if (objectId != b.objectId)
        {
            return BU_maxdiff;
        }
        else
            return combine(b).cost();
    }

    void allocZone(int zoneNum, std::vector<BU_cluster> &b)
    {
        if (left_child == -1)
        {
            origin->zoneNum = zoneNum;
        }
        else
        {
            b[left_child].allocZone(zoneNum, b);
            b[right_child].allocZone(zoneNum, b);
        }
    }
};

void zoneBUAlloc()
{
    int zoneSum = SUBSPACE_NUM - MAX_LIGHT;
    triangleStruct *triangleBufferP = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
    std::vector<BU_cluster> b;
    std::set<int> p;
    b.clear();
    p.clear();
    for (int i = 0; i < num_triangles; i++)
    {
        b.push_back(BU_cluster(triangleBufferP[i]));
        p.insert(i);
    }
    for (int i = num_triangles; i > zoneSum; i--)
    {
        int min_area_p = *p.begin();
        float min_area = b[min_area_p].area;
        for (std::set<int>::iterator pt = p.begin(); pt != p.end(); pt++)
        {
            if (b[*pt].area < min_area)
            {
                min_area_p = *pt;
                min_area = b[*pt].area;
            }
        }

        int pb;
        float minCost = BU_maxdiff;
        for (std::set<int>::iterator pt = p.begin(); pt != p.end(); pt++)
        {
            if (*pt == min_area_p)
            {
                continue;
            }
            float cost = b[min_area_p].diff(b[*pt]);
            if (cost < minCost)
            {
                pb = *pt;
                minCost = cost;
            }
        }
        p.insert(b.size());
        BU_cluster new_one = b[min_area_p].combine(b[pb]);
        new_one.left_child = min_area_p;
        new_one.right_child = pb;
        b.push_back(new_one);
        p.erase(pb);
        p.erase(min_area_p);
    }

    auto pt = p.begin();
    for (int i = 0; i < zoneSum; i++)
    {
        b[*pt].allocZone(i, b);
        pt++;
    }

    for (int i = 0; i < num_triangles; i++)
    {
        printf("%d\n", triangleBufferP[i].zoneNum);
    }

    std::ofstream saveFile;
    saveFile.open("zone_Alloc.txt");
    for (int i = 0; i < num_triangles; i++)
    {
        saveFile << triangleBufferP[i].zoneNum << " ";
    }
    saveFile.close();
    context["triangle_samples"]->getBuffer()->unmap();
    return;
}

void loadZoneSetting()
{
    triangleStruct *triangleBufferP = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
    std::ifstream saveFile;
    saveFile.open("zone_Alloc.txt");
    int i = 0;
    while (saveFile >> triangleBufferP[i].zoneNum)
    {
        i++;
    }
    saveFile.close();
    context["triangle_samples"]->getBuffer()->unmap();
    return;
}
int LT_trace_old(int core = PCPT_CORE_NUM,int core_size = LIGHT_VERTEX_PER_CORE, bool M_FIX = false)
{
    int target_raw_size = core * core_size;
    RTsize raw_size;
    context["raw_LVC"]->getBuffer()->getSize(raw_size);
    if (target_raw_size > raw_size)
    {
        context["raw_LVC"]->getBuffer()->setSize(target_raw_size);
    }


    int LVC_frame = 0;
    LVC_frame = context["LVC_frame"]->getInt();
    context["LVC_frame"]->setInt(++LVC_frame);
    //int origin_M = context["M_FIX"]->getInt();
    
    context["LT_CORE_NUM"]->setInt(core);
    context["LT_CORE_SIZE"]->setInt(core_size);
    context["M_FIX"]->setInt(M_FIX == true ? 1 : 0);
    context->launch(lightBufferGen, core, 1);
    
    return core * core_size;
    //context["M_FIX"]->setInt(origin_M);
}
void LTC_process()
{
#ifndef LTC_STRA
    return;
#endif // !LTC_STRA
    if(true)
    {
        float t = sutil::currentTime();
        context->launch(LTCLaunchProg, LTC_CORE_NUM, 1);

        printf("LTC launch custom %f time\n", sutil::currentTime() - t);
        t = sutil::currentTime();

        RTsize LTCSIZE; 
        context["LTC"]->getBuffer()->getSize(LTCSIZE);
        //LTC process
        float4 * LT_result_buffer_p = reinterpret_cast<float4 *>(context["LT_result_buffer"]->getBuffer()->map());
        LightTraceCache* LT_P = reinterpret_cast<LightTraceCache*>(context["LTC"]->getBuffer()->map());


        printf("LTC map custom %f time\n", sutil::currentTime() - t);
        t = sutil::currentTime();

        RTsize W, H;
        context["LT_result_buffer"]->getBuffer()->getSize(W, H);
        for (int i = 0; i < W*H; i++)
        {
            LT_result_buffer_p[i] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }
        int pathCount = 0;
        for (int i = 0; i < LTCSIZE; i++)
        {
            LightTraceCache & ltc = LT_P[i];
            if (ltc.valid == true && ltc.origin == true)
            {
                pathCount++;
                ltc.valid = false;
                ltc.origin = false;
            }
        }
        for (int i = 0; i < LTCSIZE; i++)
        {
            LightTraceCache & ltc = LT_P[i];
            if (ltc.valid == true && ltc.pixiv_loc.x!= 0 && ltc.pixiv_loc.y!=0)
            {
                LT_result_buffer_p[ltc.pixiv_loc.y * W + ltc.pixiv_loc.x] += make_float4(ltc.result, 0) / pathCount;
            }
        }

        context["LT_result_buffer"]->getBuffer()->unmap();
        context["LTC"]->getBuffer()->unmap(); 
        printf("LTC ADD custom %f time\n", sutil::currentTime() - t); 
        t = sutil::currentTime();
    }
    if (false)
    {
        LT_trace(context,LTC_CORE_NUM, LTC_SPC, false);
        BDPTVertex * PM = reinterpret_cast<BDPTVertex*>(context["PM"]->getBuffer()->map());
        RAWVertex * raw_LVC = reinterpret_cast<RAWVertex*>(context["raw_LVC"]->getBuffer()->map());
        int LTC_path_count = 0;
        int valid_PM_count = 0;
        for (int i = 0; i < LTC_SAVE_SUM; i++)
        {
            if (raw_LVC[i].valid == true && raw_LVC[i].v.depth == 0)
            {
                LTC_path_count++;
                raw_LVC[i].valid = false;
            }
            else  
            {
                PM[valid_PM_count] = raw_LVC[i].v;
                valid_PM_count++;
            }
        }
        context["PM"]->getBuffer()->unmap();
        context["raw_LVC"]->getBuffer()->unmap();
        buildPhotonMap(context["PM"]->getBuffer(), context["KdPM"]->getBuffer(), valid_PM_count);
        context["LTC_path_count"]->setInt(LTC_path_count);
    }

}
void light_cache_process_LVCBPT()
{
    static bool is_init = true;
    {
        int M1 = 10 * 10000;
        int M2 = RIS_M2;
        int lvc_path_buffer_length = 15;
    }

    int n_size = PCPT_CORE_NUM * LIGHT_VERTEX_PER_CORE;// M1* lvc_path_buffer_length;
    if (is_init)
    {
        is_init = false;
        context["LVC_frame"]->setInt(0); 
        RTsize raw_size, lvc_size;
        context["raw_LVC"]->getBuffer()->getSize(raw_size);
        context["LVC"]->getBuffer()->getSize(lvc_size);
        if (raw_size < n_size)context["raw_LVC"]->getBuffer()->setSize(n_size);
        if (lvc_size < n_size)context["LVC"]->getBuffer()->setSize(n_size);
    }
    LT_trace(context);
    int path_count = 0;
    int path_vertex = 0;
    static int pathSum = 0;
    static int vertexSum = 0;
    float average_length;

    auto raw_p = thrust::device_pointer_cast(context["raw_LVC"]->getBuffer()->getDevicePointer(0));
    auto lvc_p = thrust::device_pointer_cast(context["LVC"]->getBuffer()->getDevicePointer(0)); 
    MLP::data_obtain_cudaApi::LVC_process_simple(raw_p, lvc_p, n_size, path_count, path_vertex);



    //parameter setting
    vertexSum += path_vertex;
    pathSum += path_count;
    printf("lvc_size %d\n", n_size);

    average_length = float(vertexSum) / pathSum;
    context["light_vertex_count"]->setInt(path_vertex);
    context["light_path_count"]->setInt(path_count); 
    context["average_light_length"]->setFloat(average_length);
    printf("aver_path%f \n", average_length);
    context["light_path_sum"]->setInt(pathSum); 
#ifdef LTC_STRA
    context["LTC_weight"]->setFloat(1.0f);
#else
    context["LTC_weight"]->setFloat(0.0f);
#endif
}


void light_cache_process_RISBPT()
{
    static bool is_init = true; 
    int M1 = RIS_M2;
    int M2 = RIS_M2;
    int lvc_path_buffer_length = LIGHT_VERTEX_PER_CORE; 

    int n_size = M1 * lvc_path_buffer_length;// M1* lvc_path_buffer_length;
    if (is_init)
    {
        is_init = false;
        context["LVC_frame"]->setInt(0);
        RTsize raw_size, lvc_size;
        context["raw_LVC"]->getBuffer()->getSize(raw_size);
        context["LVC"]->getBuffer()->getSize(lvc_size);
        if (raw_size < n_size)context["raw_LVC"]->getBuffer()->setSize(n_size);
        if (lvc_size < n_size)context["LVC"]->getBuffer()->setSize(n_size);

        light_cache_process_RISBPT();
    }
    LT_trace(context, M1, lvc_path_buffer_length, true);
    int path_count = 0;
    int path_vertex = 0;
    static int pathSum = 0;
    static int vertexSum = 0;
    float average_length;

    auto raw_p = thrust::device_pointer_cast(context["raw_LVC"]->getBuffer()->getDevicePointer(0));
    auto lvc_p = thrust::device_pointer_cast(context["LVC"]->getBuffer()->getDevicePointer(0));
    MLP::data_obtain_cudaApi::LVC_process_simple(raw_p, lvc_p, n_size, path_count, path_vertex);

    if (false)
    {
        auto vpls = MLP::data_obtain_cudaApi::get_light_cut_sample(lvc_p, path_vertex);
        auto light_tree = classTree::lightTree(vpls);
        classTree::light_tree_api light_tree_dev = MLP::data_obtain_cudaApi::light_tree_to_device(light_tree.v.data(), light_tree.v.size());
        context["classTree::light_tree_dev"]->setUserData(sizeof(classTree::light_tree_api), &light_tree_dev);
        printf("light_tree build complete\n");
    }


    //parameter setting
    vertexSum += path_vertex;
    pathSum += path_count;
    printf("lvc_size %d\n", n_size);

    average_length = float(vertexSum) / pathSum;
    context["light_vertex_count"]->setInt(path_vertex);
    context["light_path_count"]->setInt(path_count);
    context["average_light_length"]->setFloat(average_length);
    printf("aver_path%f \n", average_length);
    context["light_path_sum"]->setInt(pathSum);
#ifdef LTC_STRA
    context["LTC_weight"]->setFloat(1.0f);
#else
    context["LTC_weight"]->setFloat(0.0f);
#endif


    static bool KD_SET = false;
    // printf("A\n"); 
    if (true)
    {  
        RTsize PDSizeW, PDSizeH, PDSizeD;
        context["PMFCaches"]->getBuffer()->getSize(PDSizeW, PDSizeH, PDSizeD);
        context["EVC_height"]->setUint(PDSizeH);
        context["EVC_width"]->setUint(PDSizeW);
        context["EVC_max_depth"]->setUint(PDSizeD);
         
        context->launch(EVCLaunch, PDSizeW, PDSizeH);
          

        int validKd_count = buildVDirectorBuffer();
        printf("kd build finished %d\n",validKd_count);
        if (!KD_SET)
        {
            context["KDPMFCaches"]->getBuffer()->setSize(2 * validKd_count);
        }
         
        RTsize kd_size;
        context["KDPMFCaches"]->getBuffer()->getSize(kd_size);
        KDPos* kd = reinterpret_cast<KDPos*>(context["Kd_position"]->getBuffer()->map());
        PMFCache* kdp = reinterpret_cast<PMFCache*>(context["KDPMFCaches"]->getBuffer()->map());
          
        static int cFrame = 0;
        int valid_c = 0;
        if (KD_SET)
            cFrame++;
        for (int i = 0; i < kd_size; i++)
        {
            auto& a = kd[i];
            auto& b = kdp[i];
            b.valid = a.valid;
            b.position = a.position;
            b.normal = a.normal;
            b.in_direction = a.in_direction;
            b.axis = a.axis;
            if (KD_SET)
            {
                //   b.Q = lerp(b.Q, b.sum, 1.0 / cFrame);
            }
            if (b.valid)
            {
                valid_c++;
                //printf("Q %d:%f\n", i, b.Q);
            }
        } 

        context["Kd_position"]->getBuffer()->unmap();
        context["KDPMFCaches"]->getBuffer()->unmap(); 

        printf("PMFCache launch . vertex count %d\n", path_vertex);
        context->launch(PMFCacheLaunch, kd_size, 1);
        printf("PMFCache launch . vertex count finish %d\n", path_vertex);
         
        // printf("B\n");

        kd = reinterpret_cast<KDPos*>(context["Kd_position"]->getBuffer()->map());
        KDPos* kd2 = reinterpret_cast<KDPos*>(context["last_kd_position"]->getBuffer()->map());
        kdp = reinterpret_cast<PMFCache*>(context["KDPMFCaches"]->getBuffer()->map()); 
        for (int i = 0; i < kd_size; i++)
        {
            auto& a = kd[i];
            auto& b = kdp[i];
            auto& c = kd2[i];
            a.Q = b.Q;
            a.Q_variance = b.Q_variance;
            a.shadow_success = b.shadow_success; 
            c = a;  
        }
        //virtual_pmf_id
        kdp[kd_size - 1] = PMFCache(path_vertex);
        context["virtual_pmf_id"]->setInt(kd_size - 1);
         
        context["Kd_position"]->getBuffer()->unmap();
        context["KDPMFCaches"]->getBuffer()->unmap();
        context["last_kd_position"]->getBuffer()->unmap(); 
         
        if (KD_SET == false)
        {
            KD_SET = true;
            context["KD_SET"]->setInt(1);
        }
        RTsize cache_size;
        context["KDPMFCaches"]->getBuffer()->getSize(cache_size);
        int pmf_one_size = context["KDPMFCaches"]->getBuffer()->getElementSize();
        printf("CacheSize %d %d %d\n", cache_size, pmf_one_size, cache_size * pmf_one_size);
         
    } 
}
void pre_processing()
{
    static std::vector<float> optimal_E(1000000, 1.0 / SUBSPACE_NUM);
    static std::vector<float> optimal_Q(SUBSPACE_NUM, 1.0);
    float t_c = sutil::currentTime();
    float pre_process_time = 0;
#ifdef ZGCBPT

    ////////////////////////////////////////////
    //////////////pre_processing stage//////////
    ////////////////////////////////////////////
    optimal_E = train_api.data.get_data(context, optimal_Q, pre_process_time);
    subspaceDivTime = pre_process_time;
#endif
    printf("data generate time :%f %f\n\n", sutil::currentTime() - t_c, pre_process_time);

    svm_api.load_optimal_E(context, optimal_E);
    load_Q(context, optimal_Q);
    gamma.setup(optimal_E, optimal_Q);
    ////////////////////////////////////pre process complete///////////////////////

}
void light_cache_process_ZGCBPT() 
{
    static int frame_count = 0;

    int n_size = PCPT_CORE_NUM * LIGHT_VERTEX_PER_CORE;// M1* lvc_path_buffer_length;
    static bool is_init = true;
    if (is_init)
    {
        is_init = false;
        RTsize raw_size, lvc_size;
        context["raw_LVC"]->getBuffer()->getSize(raw_size);
        context["LVC"]->getBuffer()->getSize(lvc_size);
        if (raw_size < n_size)context["raw_LVC"]->getBuffer()->setSize(n_size);
        if (lvc_size < n_size)context["LVC"]->getBuffer()->setSize(n_size);
    } 
    
    //tracing sub-paths, store sub-paths in raw_LVC    
    LT_trace(context);


    int path_count = 0;
    int path_vertex = 0;
    static int pathSum = 0;
    static int vertexSum = 0;
    float average_length;

    auto raw_p = thrust::device_pointer_cast(context["raw_LVC"]->getBuffer()->getDevicePointer(0));
    auto lvc_p = thrust::device_pointer_cast(context["LVC"]->getBuffer()->getDevicePointer(0));
    //process the sub-paths in raw_LVC, delete the empty sub-paths (pre alloc for GPU tracing but do not get a sub-path)
    //also count the statistic information for estimation
    MLP::data_obtain_cudaApi::LVC_process_simple(raw_p, lvc_p, n_size, path_count, path_vertex);

    //statistics update
    vertexSum += path_vertex;
    pathSum += path_count;  
    average_length = float(vertexSum) / pathSum;
    context["light_vertex_count"]->setInt(path_vertex);
    context["light_path_count"]->setInt(path_count);
    context["average_light_length"]->setFloat(average_length); 
    context["light_path_sum"]->setInt(pathSum); 
    context["LTC_weight"]->setFloat(0.0f); //disable the light tracing strategy (s = 0), because it is of low efficiency in most of the cases
     

    //move light sub-paths to their light subspaces
    subspaces_api.buildSubspace(context);//details to be done



    context["M3_buffer"]->setBuffer(context["M2_buffer"]->getBuffer()); 

}
 
void light_cache_process()
{ 
#ifdef RRPT
    context["average_light_length"]->setFloat(1);
    return;
#endif
    static int vertexSum = 0;
    static int pathSum = 0;
    float average_length;
    static bool isFirstFrame = true;
    static bool uber_static_state = false;
    static float idea_energy = 1;
    static ZoneMatrix *StableP = new ZoneMatrix[SUBSPACE_NUM];
    static ZoneMatrix *eyeM2P = new ZoneMatrix[SUBSPACE_NUM];
    Q_update(context,pathSum);
    //if(context["frame"]->getUint()<5)
    E_update(context, StableP, eyeM2P);
    //svm_api.load_optimal_E(context,mlp_debug_api.ans);
    static std::vector<float> optimal_E(SUBSPACE_NUM * SUBSPACE_NUM, 1.0 / SUBSPACE_NUM);
    static std::vector<float> optimal_Q(SUBSPACE_NUM,1.0);
#ifndef EYE_DIRECT
    svm_api.load_optimal_E(context, optimal_E);
    load_Q(context, optimal_Q);
#endif // !EYE_DIRECT
    //svm_api.load_optimal_E(context, optimal_E);


    if (isFirstFrame)
    {
        context["LVC_frame"]->setInt(0);
        float t_c = sutil::currentTime();
        float pre_process_time = 0;
#ifdef ZGCBPT
        optimal_E = train_api.data.get_data(context, optimal_Q, pre_process_time);
        subspaceDivTime = pre_process_time;
#endif
        printf("data generate time :%f %f\n\n", sutil::currentTime() - t_c,pre_process_time);

        isFirstFrame = false;
        zoneRawAlloc();

        div_tris.rawAlloc(context);
        div_tris.validation(context);

        //tri_alloc_by_grid(context, div_tris);
#ifdef SLIC_CLASSIFY 
        float begintime = sutil::currentTime();
        //tri_alloc_by_simpleSLIC(context, div_tris);
        slic_gpu_api.segmentation(context, div_tris);
        //subspaceDivTime = sutil::currentTime() - begintime;
        printf("simple slic custom %f second\n", sutil::currentTime() - begintime);
        //svm_api.getGamma(context);
        
        //classTree::tree t; 
        //svm_api.GetPathInfo(context); 
        
        //classTree::tree_node* dev_t = device_to(t.v, t.size);

#endif
        //zoneSLICAlloc();
        //zoneBUAlloc();
        ZoneMatrix * M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            for (int j = 0; j < SUBSPACE_NUM; j++)
            {
                M2[i].r[j] = 0;
                M2[i].m[j] = 0;
                StableP[i].r[j] = 0.001 / SUBSPACE_NUM;
                StableP[i].m[j] = (j + 1) * 0.001 / SUBSPACE_NUM;
                eyeM2P[i].r[j] = 1.0 / SUBSPACE_NUM;
                eyeM2P[i].m[j] = (j + 1) * 1.0 / SUBSPACE_NUM;
            }
            M2[i].sum = 0;
            StableP[i].sum = 0;
            eyeM2P[i].sum = 1;
        }
        context["M2_buffer"]->getBuffer()->unmap();


        ZoneSampler *zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            zoneLVC[i].Q = 0.0f;
            zoneLVC[i].Q_old = 0.0f;
            zoneLVC[i].sum = 0.01f;
            zoneLVC[i].size = 0;
        }
        context["zoneLVC"]->getBuffer()->unmap();
        light_cache_process();
        Q_update(context,pathSum);
        zoneLVCInitFlag = true;
    }  
    
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();
#ifdef PCBPT
    LT_trace(context, PCPT_CORE_NUM, LIGHT_VERTEX_PER_CORE, true);
    //light_cache_process_risbpt();
#else

    LT_trace(context, PCPT_CORE_NUM, LIGHT_VERTEX_PER_CORE, false); 

#endif // PCBPT 


    printf("Light trace custom %f s\n", sutil::currentTime() - lastTime);
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();
    //如果是第二帧，那么在追踪后把zoneLVC清零再统计
    if (zoneLVCInitFlag == true)
    {
        zoneLVCInitFlag = false;
        ZoneSampler *zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            zoneLVC[i].size = 0;
        }
        context["zoneLVC"]->getBuffer()->unmap();
        pathSum = 0;
        vertexSum = 0;
#ifdef ZGCBPT
#ifndef SLIC_CLASSIFY
        //ZoneLVCInit(context, context["triangle_sum"], StableP, div_tris);
#endif
        pathSum = context["light_path_sum"]->getInt();

        LT_trace(context, PCPT_CORE_NUM, LIGHT_VERTEX_PER_CORE, false); 
        uber_static_state = true;
#endif // ZGCBPT
        //ZoneLVCInit(context, num_triangles, StableP);
        //pathSum = context["light_path_sum"]->getInt();
        
    }
     
    RAWVertex * RAWLVC = reinterpret_cast<RAWVertex*>(context["raw_LVC"]->getBuffer()->map()); 
    BDPTVertex* LVC = reinterpret_cast<BDPTVertex*>(context["LVC"]->getBuffer()->map());
    ZoneMatrix * M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
    ZoneSampler *zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());

    printf("bufferMap custom %f s\n", sutil::currentTime() - lastTime);
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();

    int vertexCount = 0;
    int pathCount = 0;
    RTsize LVCSIZE;
    context["LVC"]->getBuffer()->getSize(LVCSIZE);

    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        zoneLVC[i].realSize = 0;
    }
    for (int i = 0; i < LVCSIZE; i++)
    {
        if (RAWLVC[i].valid && RAWLVC[i].v.isBrdf == false)
        {
            BDPTVertex &vertex = RAWLVC[i].v;
            if (RAWLVC[i].v.depth == 0)
            {
                pathCount++;
#ifdef INDIRECT_ONLY
                continue;
#endif // INDIRECT_ONLY
            }

            float3 flux = vertex.flux / vertex.pdf;
            float weight = flux.x + flux.y + flux.z;

            if (isinf(weight) || isnan(weight))
            {
                continue;
            }
#ifndef ZGCBPT
#endif
            LVC[vertexCount] = vertex;
            vertexCount++;
#
#ifndef ZGCBPT
            continue;
#endif // !ZGCBPT
            if (vertex.depth >= 1 )
            {
                //M2[vertex.zoneId].r[RAWLVC[i].lastZoneId] += weight;
                //M2[vertex.zoneId].sum += weight; 
                StableP[vertex.zoneId].r[RAWLVC[i].lastZoneId] += weight;
                StableP[vertex.zoneId].sum += weight;
            }
            
            ZoneSampler & vZone = zoneLVC[vertex.zoneId];
            vZone.v[vZone.realSize % MAX_LIGHT_VERTEX_PER_TRIANGLE] = vertex;
            vZone.Q_old += weight;
            vZone.size++;
            vZone.realSize++; 
        }
    }


    printf("Lightvertex copy custom %f s\n", sutil::currentTime() - lastTime);
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();
#ifdef ZGCBPT


    int ValidZoneNum = 0;
    float averageVertexInZone = 0;
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        if (zoneLVC[i].size > 0)
        {
            ValidZoneNum++;
        }
        averageVertexInZone += min(zoneLVC[i].size, MAX_LIGHT_VERTEX_PER_TRIANGLE);
    }
    averageVertexInZone /= ValidZoneNum;
    printf("ValidZone: %d  averNum: %f\n", ValidZoneNum, averageVertexInZone);
#ifdef EYE_DIRECT


    RTsize RecordW, RecordH, RecordD;
    context["result_record_buffer"]->getBuffer()->getSize(RecordD, RecordH, RecordW);
    eyeResultRecord* recordP = reinterpret_cast<eyeResultRecord*>(context["result_record_buffer"]->getBuffer()->map());
    static bool recordInit = true;
    static int countnum = 0;
    countnum++;
    if (countnum >= EYE_DIRECT_FRAME)
    {
    }
    else if (recordInit)
    {
        recordInit = false;
        for (int i = 0; i < RecordW * RecordD * RecordH; i++)
        {
            recordP[i].valid = false;
        }
    }
    else
    {
        for (int i = 0; i < RecordW * RecordD * RecordH; i++)
        {
            eyeResultRecord &r = recordP[i];
            if (r.valid) {
                float weight = r.result.x + r.result.y + r.result.z;
                if (!isinf(weight) && !isnan(weight))
                {
                    eyeM2P[r.eyeZone].r[r.lightZone] += weight;
                    eyeM2P[r.eyeZone].sum += weight;
                }
            }
            r.valid = false;
        }
    } 
    context["result_record_buffer"]->getBuffer()->unmap();

    printf("eyeDataGet custom %f s\n", sutil::currentTime() - lastTime);
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();
#endif  
    static int countttt = 100;
    countttt--;
    if (countttt == 0)
    {
        for (int i = 0; i < 1000; i++)
        {
            printf("Q: %d %f\n", i, zoneLVC[i].Q / pathSum);
        }
    }

    float bias_rate = 0;
    static float  uniform_rate = 1;
    static int uniform_count = 0;
    uniform_rate -= 0.05 ;
    uniform_rate = 0.1;
    uniform_rate = uniform_rate < 0.1 ? 0.1 : uniform_rate;
    //uniform_rate = 0.5;W
    if (uniform_count > 0)
    {
        uniform_count -= 1;
        uniform_rate = 1;
    }
    else
    {
        uniform_rate = 0.2;
    }
    printf("%f\n", uniform_rate); 
 
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        zoneLVC[i].lum_mr_set();
        if (context["frame"]->getUint() > 500)break;
        StableP[i].m[0] = StableP[i].r[0];
        eyeM2P[i].m[0] = eyeM2P[i].r[0];
        for (int j = 1; j < SUBSPACE_NUM; j++)
        {
#ifdef INDIRECT_ONLY
            if (j >= SUBSPACE_NUM - MAX_LIGHT)
            {
                StableP[i].r[j] = 0;
                eyeM2P[i].r[j] = 0;
            }
#endif
            StableP[i].m[j] = StableP[i].r[j] + StableP[i].m[j - 1];
            eyeM2P[i].m[j] = eyeM2P[i].r[j] + eyeM2P[i].m[j - 1];
        }
        StableP[i].sum = StableP[i].m[SUBSPACE_NUM - 1];
        eyeM2P[i].sum = eyeM2P[i].m[SUBSPACE_NUM - 1]; 
        if (StableP[i].sum == 0.0f)
            StableP[i].sum = 0.000001f;
        //M2[i] = StableP[i];
       // M2[i].sum = 0.0;
        for (int j = 0; j < SUBSPACE_NUM; j++)
        {
            float lerp_rate = 1.0;
#ifdef EYE_DIRECT
            lerp_rate = 0.1;
#endif // EYE_DIRECT


           // M2[i].r[j] = StableP[i].r[j] / StableP[i].sum * lerp_rate + eyeM2P[i].r[j] / eyeM2P[i].sum * (1.0 - lerp_rate); 
           // M2[i].r[j] = lerp(M2[i].r[j], 1.0 / SUBSPACE_NUM, uniform_rate);
            

            //M2[i].r[j] = 1;
        } 
       // M2[i].validation();
        //printf("%d  %f\n", i, M2[i].sum);
        //  printf("%f\n", eyeM2P[i].sum);
    }
    //M2 Blur
    if (false)
    {
        ZoneMatrix * tmpM2 = new ZoneMatrix[SUBSPACE_NUM];
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            tmpM2[i].init();
        }
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            for (int j = -2; j <= 2; j++)
            {
                int t = i + j;
                if (i + j < 0 || i + j >= SUBSPACE_NUM)
                    continue;
                tmpM2[i].raw_add(M2[t], 3 - abs(j));
            }
            tmpM2[i].blur(2);
        }
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            M2[i] = tmpM2[i];

        }
        delete[] tmpM2;
    }
    //  printf("biasrate:%f\n", bias_rate * 100 / SUBSPACE_NUM / SUBSPACE_NUM ); 
    ZoneMatrix* M3_buffer = reinterpret_cast<ZoneMatrix*>(context["M3_buffer"]->getBuffer()->map());
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        for(int j=0;j<SUBSPACE_NUM;j++)
        {
            if (zoneLVC[j].realSize == 0)
            {
                M3_buffer[i].r[j] = 0;
#ifdef LIGHTVERTEX_REUSE 
                M3_buffer[i].r[j] = M2[i].r[j]; 
#endif 
            }
            else
            {
                M3_buffer[i].r[j] = M2[i].r[j];
            }
        }
        M3_buffer[i].validation();
    }
    context["M3_buffer"]->getBuffer()->unmap();

    printf("M2Process custom %f s\n", sutil::currentTime() - lastTime);
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();
#endif // ZGCBPT
    vertexSum += vertexCount;
    pathSum += pathCount;
     
    average_length = float(vertexCount) / pathCount;
    context["light_vertex_count"]->setInt(vertexCount);
    context["light_path_count"]->setInt(pathCount);
    printf("path traced M %d ,%d light vertex generated\n", pathCount,vertexCount);
    //context["light_path_count"]->setInt(1);
    context["average_light_length"]->setFloat(average_length);
    printf("aver_path%f\n",average_length);
    context["light_path_sum"]->setInt(pathSum);
    //context["LTC_weight"]->setFloat(float(pathCount) / W / H * average_length);
#ifdef LTC_STRA
    context["LTC_weight"]->setFloat(1.0f);
#else
    context["LTC_weight"]->setFloat(0.0f);
#endif

    context["raw_LVC"]->getBuffer()->unmap();
#ifndef ZGCBPT
#endif
    context["LVC"]->getBuffer()->unmap();
    context["M2_buffer"]->getBuffer()->unmap();
    context["zoneLVC"]->getBuffer()->unmap();
    
#ifdef PCBPT
    static bool KD_SET = false;
    // printf("A\n"); 
    if (!isFirstFrame)
    {
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        RTsize PDSizeW, PDSizeH, PDSizeD;
        context["PMFCaches"]->getBuffer()->getSize(PDSizeW, PDSizeH, PDSizeD);
        context["EVC_height"]->setUint(PDSizeH);
        context["EVC_width"]->setUint(PDSizeW);
        context["EVC_max_depth"]->setUint(PDSizeD);



        context->launch(EVCLaunch, PDSizeW, PDSizeH);


        printf("EVC process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        int validKd_count = buildVDirectorBuffer();
        if (!KD_SET)
        {
            context["KDPMFCaches"]->getBuffer()->setSize(2 * validKd_count);
        }

        printf("kd_build process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        RTsize kd_size;
        context["KDPMFCaches"]->getBuffer()->getSize(kd_size);
        KDPos* kd = reinterpret_cast<KDPos*>(context["Kd_position"]->getBuffer()->map());
        PMFCache* kdp = reinterpret_cast<PMFCache*>(context["KDPMFCaches"]->getBuffer()->map());


        printf("kd_map process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        static int cFrame = 0;
        int valid_c = 0;
        if (KD_SET)
            cFrame++;
        for (int i = 0; i < kd_size; i++)
        {
            auto& a = kd[i];
            auto& b = kdp[i];
            b.valid = a.valid;
            b.position = a.position;
            b.normal = a.normal;
            b.in_direction = a.in_direction;
            b.axis = a.axis;
            if (KD_SET)
            {
                //   b.Q = lerp(b.Q, b.sum, 1.0 / cFrame);
            }
            if (b.valid)
                valid_c++;
        }

        printf("kd_copy process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();


        context["Kd_position"]->getBuffer()->unmap();
        context["KDPMFCaches"]->getBuffer()->unmap();
        printf("kd_unmap process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();
         
        context->launch(PMFCacheLaunch, kd_size, 1);

        printf("shaodw_ray process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();
        // printf("B\n");

        kd = reinterpret_cast<KDPos*>(context["Kd_position"]->getBuffer()->map());
        KDPos* kd2 = reinterpret_cast<KDPos*>(context["last_kd_position"]->getBuffer()->map());
        kdp = reinterpret_cast<PMFCache*>(context["KDPMFCaches"]->getBuffer()->map());

        printf("kd_map2 process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        for (int i = 0; i < kd_size; i++)
        {
            auto& a = kd[i];
            auto& b = kdp[i];
            auto& c = kd2[i];
            a.Q = b.Q;
            a.Q_variance = b.Q_variance;
            a.shadow_success = b.shadow_success;
            //printf("%f\n", a.Q);
            c = a;
            //if (b.valid && b.shadow_success != 0)
            //{ 
            //}

        }
        //virtual_pmf_id
        kdp[kd_size - 1] = PMFCache(vertexCount);
        context["virtual_pmf_id"]->setInt(kd_size - 1);


#ifdef PCBPT_OPTIMAL
        elapsedTime += sutil::currentTime() - lastTime;
        std::vector<KDPos> New_kd_pos;
        KD_tree New_kd;
        for (int i = 0; i < kd_size; i++)
        {
            auto& a = kdp[i];
            if (a.valid == true && a.shadow_success != 0)
                New_kd_pos.push_back(KDPos(a.position,i));
        }
        New_kd.construct(New_kd_pos);

        for (int i = 0;i < kd_size; i++)
        {
            auto& a = kdp[i];
            a.fix_init();
            if (a.valid)
            {
                auto k_pair = New_kd.find(a.position, FIX_RANGE);
                for (auto p = k_pair.begin(); p != k_pair.end(); p++)
                {
                    a.merge_with_other(kdp[*p]);
                }
                a.fix_validation();
            }
        } 
        lastTime = sutil::currentTime();
#endif


        printf("kd_copy process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();
        
        context["Kd_position"]->getBuffer()->unmap();
        context["KDPMFCaches"]->getBuffer()->unmap();
        context["last_kd_position"]->getBuffer()->unmap();
        printf("kd_unmap process run for %f second\n", sutil::currentTime() - lastTime);
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        if (!KD_SET)
        { 
            KD_SET = true;
            context["KD_SET"]->setInt(1);
            LT_trace(context, PCPT_CORE_NUM, LIGHT_VERTEX_PER_CORE, true);
            RAWVertex * RAWLVC = reinterpret_cast<RAWVertex*>(context["raw_LVC"]->getBuffer()->map());
            BDPTVertex * LVC = reinterpret_cast<BDPTVertex*>(context["LVC"]->getBuffer()->map());

            vertexCount = 0;
            for (int i = 0; i < LVCSIZE; i++)
            {
                if (RAWLVC[i].valid)
                {
#ifdef INDIRECT_ONLY
                    if (RAWLVC[i].v.depth == 0)
                        continue;
#endif
                    BDPTVertex &vertex = RAWLVC[i].v;
                    LVC[vertexCount++] = vertex;
                }
            }

            context["light_vertex_count"]->setInt(vertexCount);

            context["raw_LVC"]->getBuffer()->unmap();
            context["LVC"]->getBuffer()->unmap();

            context["PMFCaches"]->getBuffer()->getSize(PDSizeW, PDSizeH, PDSizeD);
            context["EVC_height"]->setUint(PDSizeH);
            context["EVC_width"]->setUint(PDSizeW);
            context["EVC_max_depth"]->setUint(PDSizeD);
            context->launch(EVCLaunch, PDSizeW, PDSizeH);
            buildVDirectorBuffer();
            context["KDPMFCaches"]->getBuffer()->getSize(kd_size);
            kd = reinterpret_cast<KDPos*>(context["Kd_position"]->getBuffer()->map());
            kdp = reinterpret_cast<PMFCache*>(context["KDPMFCaches"]->getBuffer()->map());
            for (int i = 0; i < kd_size; i++)
            {
                auto & a = kd[i];
                auto & b = kdp[i];
                b.valid = a.valid;
                b.position = a.position;
                b.normal = a.normal;
                b.in_direction = a.in_direction;
                b.axis = a.axis;
            }
            context["Kd_position"]->getBuffer()->unmap();
            context["KDPMFCaches"]->getBuffer()->unmap();
            context->launch(PMFCacheLaunch, kd_size, 1);

            kd = reinterpret_cast<KDPos*>(context["Kd_position"]->getBuffer()->map());
            kd2 = reinterpret_cast<KDPos*>(context["last_kd_position"]->getBuffer()->map());
            kdp = reinterpret_cast<PMFCache*>(context["KDPMFCaches"]->getBuffer()->map());
            for (int i = 0; i < kd_size; i++)
            {
                auto & a = kd[i];
                auto & b = kdp[i];
                auto & c = kd2[i];
                a.Q = b.Q;
                a.Q_variance = b.Q_variance;
                a.shadow_success = b.shadow_success;
                c = a;
            }

            std::vector<KDPos> New_kd_pos;
            KD_tree New_kd;
            for (int i = 0; i < kd_size; i++)
            {
                auto& a = kdp[i];
                if (a.valid == true && a.shadow_success != 0)
                    New_kd_pos.push_back(KDPos(a.position, i));
            }
            New_kd.construct(New_kd_pos);

            for (int i = 0; i < kd_size; i++)
            {
                auto& a = kdp[i];
                a.fix_init();
                if (a.valid)
                {
                    auto k_pair = New_kd.find(a.position, 10);
                    for (auto p = k_pair.begin(); p != k_pair.end(); p++)
                    {
                        a.merge_with_other(kdp[*p]);
                    }
                    a.fix_validation();
                }
            }

 
            context["Kd_position"]->getBuffer()->unmap();
            context["last_kd_position"]->getBuffer()->unmap();
            context["KDPMFCaches"]->getBuffer()->unmap();
            // printf("KD_SET\n");
        }
    }
    // printf("C\n");
#endif // PCBPT
#ifdef ZGCBPT 
    static int E_init_count = 1;
    if (E_init_count > 0)
    {
        E_update(context, StableP, eyeM2P);
        E_init_count--;
    }
    subspaces_api.buildSubspace(context);
#endif
    if(uber_static_state)
    {
        elapsedTime += sutil::currentTime() - lastTime;
        lastTime = sutil::currentTime();

        UberVertexGen(context, UBER_VERTEX_NUM, random_generator);
        lastTime = sutil::currentTime();
    }
    LTC_process(); 
}


void createContext(bool use_pbo)
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount(RayTypeCount);
    context->setEntryPointCount(rayGenProNum);
    context->setPrintEnabled(true);
    context->setPrintBufferSize(100);
    context->setStackSize(8000);
    
    // Note: high max depth for reflection and refraction through glass
    context["max_depth"]->setInt(1 );
    context["cutoff_color"]->setFloat(0.0f, 0.0f, 0.0f);
    context["frame"]->setUint(0u);
    context["scene_epsilon"]->setFloat(1.e-3f);

    Buffer buffer = //sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height, use_pbo);
        context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height);
    context["output_buffer"]->set(buffer);
    Buffer false_buffer = //sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height, use_pbo);
        context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height);
    context["false_buffer"]->set(false_buffer);

    Buffer false_mse_buffer = //sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height, use_pbo);
        context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height);
    context["false_mse_buffer"]->set(false_mse_buffer);
    corputBufferGen(context);

    {//buffer for denoise 
        Buffer tonemappedBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
        //sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
        context["tonemapped_buffer"]->set(tonemappedBuffer);

        Buffer albedoBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
        context["input_albedo_buffer"]->set(albedoBuffer);
        Buffer normalBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
        context["input_normal_buffer"]->set(normalBuffer);

        denoisedBuffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
        context["denoised_buffer"]->set(denoisedBuffer);
        emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);

        Buffer LT_result_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
        context["LT_result_buffer"]->set(LT_result_buffer);

        Buffer heat_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4,
            int(ceilf(scene->properties.width / 16.0)), int(ceilf(scene->properties.height / 16.0)));
        context["heat_buffer"]->set(heat_buffer);

    }
    Buffer standrdImageBuffer = //sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height, use_pbo);
        context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height);
    context["standrd_buffer"]->set(standrdImageBuffer);
    Buffer standrd_float_buffer = //sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height, use_pbo);
        context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
    context["standrd_float_buffer"]->set(standrd_float_buffer);

    // Accumulation buffer
    Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,
        RT_FORMAT_FLOAT4, scene->properties.width, scene->properties.height);
    context["accum_buffer"]->set(accum_buffer);

    Buffer light_vertex_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, max(LTC_SAVE_SUM, max(LIGHT_VERTEX_NUM,
        2 * int(PMFCaches_RATE * scene->properties.width)* int(PMFCaches_RATE * scene->properties.height) * PMF_DEPTH)));
    light_vertex_buffer->setElementSize(sizeof(RAWVertex));
    context["raw_LVC"]->set(light_vertex_buffer);

    Buffer light_vertex_cache = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, LIGHT_VERTEX_NUM);
    light_vertex_cache->setElementSize(sizeof(BDPTVertex));
    context["LVC"]->set(light_vertex_cache);
    
     

    Buffer light_vertex_in_triangle_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, SUBSPACE_NUM);
    light_vertex_in_triangle_buffer->setElementSize(sizeof(ZoneSampler));
    context["zoneLVC"]->set(light_vertex_in_triangle_buffer);

    Buffer Mesh2Mesh_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, SUBSPACE_NUM);
    Mesh2Mesh_buffer->setElementSize(sizeof(ZoneMatrix));
    context["M2_buffer"]->set(Mesh2Mesh_buffer);

    Buffer M3_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, SUBSPACE_NUM);
    M3_buffer->setElementSize(sizeof(ZoneMatrix));
    context["M3_buffer"]->set(M3_buffer);

    Buffer result_record_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER,
        scene->properties.width, scene->properties.height, RECORD_DEPTH);
    result_record_buffer->setElementSize(sizeof(eyeResultRecord));
    context["result_record_buffer"]->set(result_record_buffer);

    Buffer light_trace_cache = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, LTC_SAVE_SUM);
    light_trace_cache->setElementSize(sizeof(LightTraceCache));
    context["LTC"]->set(light_trace_cache);

#ifdef PCBPT
    Buffer PMFCache_save_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER,
        PMFCaches_RATE * scene->properties.width, PMFCaches_RATE * scene->properties.height, PMF_DEPTH);
    PMFCache_save_buffer->setElementSize(sizeof(PMFCache));
    context["PMFCaches"]->set(PMFCache_save_buffer);

    Buffer PMFCache_kd_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER,
        1);
    PMFCache_kd_buffer->setElementSize(sizeof(PMFCache));
    context["KDPMFCaches"]->set(PMFCache_kd_buffer);

    Buffer Kd_position = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER,
        2 * int(PMFCaches_RATE * scene->properties.width)* int(PMFCaches_RATE * scene->properties.height) * PMF_DEPTH);
    Kd_position->setElementSize(sizeof(KDPos));
    context["Kd_position"]->set(Kd_position);
    Buffer last_kd_position = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER,
        2 * int(PMFCaches_RATE * scene->properties.width)* int(PMFCaches_RATE * scene->properties.height) * PMF_DEPTH);
    last_kd_position->setElementSize(sizeof(KDPos));
    
    //RTsize tmp_size;
    //PMFCache_save_buffer->getSize(tmp_size);
    //printf("pmf_init size%d\n", 2 * int(PMFCaches_RATE * scene->properties.width) * int(PMFCaches_RATE * scene->properties.height) * PMF_DEPTH);

    context["last_kd_position"]->set(last_kd_position);

#else
    Buffer PMFCache_save_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1, 1, 0);
    PMFCache_save_buffer->setElementSize(sizeof(PMFCache)); 
    context["PMFCaches"]->set(PMFCache_save_buffer); 

    Buffer PMFCache_kd_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1);
    PMFCache_kd_buffer->setElementSize(sizeof(PMFCache));
    context["KDPMFCaches"]->set(PMFCache_kd_buffer);

    Buffer Kd_position = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
    Kd_position->setElementSize(sizeof(KDPos));
    context["Kd_position"]->set(Kd_position);
    Buffer last_kd_position = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
    last_kd_position->setElementSize(sizeof(KDPos));
    context["last_kd_position"]->set(last_kd_position);
#endif

    Buffer KdPM = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 2 * LTC_SAVE_SUM);
    KdPM->setElementSize(sizeof(KDPos));
    context["KdPM"]->set(KdPM);
    Buffer PM = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 2 * LTC_SAVE_SUM);
    PM->setElementSize(sizeof(BDPTVertex));
    context["PM"]->set(PM);
    Buffer triangle_sample_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, MAX_TRIANGLE);
    triangle_sample_buffer->setElementSize(sizeof(triangleStruct));
    context["triangle_samples"]->set(triangle_sample_buffer);
    {//初始化tringle_samples数组，主要目的是让未处理时的zone编号为默认三角形编号
        triangleStruct *p = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map());
        for (int i = 0; i < MAX_TRIANGLE; i++)
        {
            int step = MAX_TRIANGLE / (SUBSPACE_NUM - 5);
            p[i].zoneNum = i / step;
        }
        context["triangle_samples"]->getBuffer()->unmap();
    }
    div_tris.createBuffer(context);

    //Buffer visibility_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, VISIBILITY_TEST_NUM, MAX_TRIANGLE);
    Buffer visibility_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, 1, 1);
    context["visibility_buffer"]->set(visibility_buffer);

    Buffer triangle_targets = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, VISIBILITY_TEST_NUM);
    triangle_targets->setElementSize(sizeof(triangleStruct));
    context["triangle_targets"]->set(triangle_targets);


    Buffer uberLVC = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1);
    uberLVC->setElementSize(sizeof(UberZoneLVC));
    context["uberLVC"]->set(uberLVC);

    {
        Buffer test_setting = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1);
        test_setting->setElementSize(sizeof(TestSetting));
        context["test_setting"]->set(test_setting);
        TestSetting *p = reinterpret_cast<TestSetting*>( context["test_setting"]->getBuffer()->map());
        p[0].vpZone = 0;
        context["test_setting"]->getBuffer()->unmap();
    }
#ifdef USE_ML_MATRIX
    Buffer ML_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, scene->properties.width, scene->properties.height, STACKSIZE);
    ML_buffer->setElementSize(sizeof(MLRecord));
    context["ML_buffer"]->set(ML_buffer);
#endif // USE_ML_MATRIX

    subspaces_api.subspace_sampler_init(context); 
    slic_gpu_api.init(context);
    svm_api.init(context);
    class_tree_api.init(context);
    mlp_api.init(context);

    // Ray generation program
    std::string ptx_path( ptxPath( "path_trace_camera.cu" ) );
    //0:渲染主程序
#ifdef USE_PCPT
    Program ray_gen_program = context->createProgramFromPTXFile( ptx_path, "GPCPT_pinhole_camera" );
#else
    Program ray_gen_program = context->createProgramFromPTXFile(ptx_path, TRACER_PROGRAM_NAME);
#endif
    context->setRayGenerationProgram( 0, ray_gen_program );
    //1:light vertex生成程序
    ray_gen_program = context->createProgramFromPTXFile(ptx_path, "light_vertex_launch");
    context->setRayGenerationProgram(lightBufferGen, ray_gen_program);

    //2:标准PCPT的标识点PMF cache的生成程序pin
    ray_gen_program = context->createProgramFromPTXFile(ptx_path, "PMFCaches_launch");
    context->setRayGenerationProgram(PMFCacheLaunch, ray_gen_program);
    //3:可见性程序，用于生成场景可见性矩阵用于kmeans聚类
    //ray_gen_program = context->createProgramFromPTXFile(ptx_path, "visiblity_test");
    //context->setRayGenerationProgram(visibilityTestProg, ray_gen_program);
    //4:格式转换，用于将降噪后的浮点denoised图像转换为4byte输出图像
    ray_gen_program = context->createProgramFromPTXFile(ptx_path, "format_transform");
    context->setRayGenerationProgram(FormatTransform, ray_gen_program);
    
    ray_gen_program = context->createProgramFromPTXFile(ptx_path, "EVC_launch");
    context->setRayGenerationProgram(EVCLaunch, ray_gen_program);

    ray_gen_program = context->createProgramFromPTXFile(ptx_path, "PG_training");
    context->setRayGenerationProgram(PGTrainingProg, ray_gen_program);
     
    auto slic_prog = context->createProgramFromPTXFile(ptx_path, "quick_slic");
    context->setRayGenerationProgram(SlicProg, slic_prog);

    //auto gamma_compute_prog = context->createProgramFromPTXFile(ptx_path, "gamma_compute");
    //context->setRayGenerationProgram(GammaComputeProg, gamma_compute_prog);

    auto mlp_path_construct_prog = context->createProgramFromPTXFile(ptx_path, "MLP_path_construct");
    context->setRayGenerationProgram(MLPPathConstructProg, mlp_path_construct_prog);

    auto OPTP_prog = context->createProgramFromPTXFile(ptx_path, "get_OPT_Info_NEE");
    context->setRayGenerationProgram(OPTPProg, OPTP_prog);
    
    //auto MLP_forward_prog = context->createProgramFromPTXFile(ptx_path, "MLP_forward");
    //context->setRayGenerationProgram(forwardProg, MLP_forward_prog);
    //auto MLP_backward_prog = context->createProgramFromPTXFile(ptx_path, "MLP_backward");
    //context->setRayGenerationProgram(backwardProg, MLP_backward_prog);
   // ray_gen_program = context->createProgramFromPTXFile(ptx_path, "Uber_vertex_C");
  //  context->setRayGenerationProgram(uberVertexProcessProg, ray_gen_program);

    ray_gen_program = context->createProgramFromPTXFile(ptx_path, "LTC_launch");
    context->setRayGenerationProgram(LTCLaunchProg, ray_gen_program);
    // Exception program
    Program exception_program = context->createProgramFromPTXFile( ptx_path, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    ptx_path = ptxPath( "background.cu" );
    context->setMissProgram( PTRay, context->createProgramFromPTXFile( ptx_path, "miss_env" ) );
    context->setMissProgram(PT_RR_RAY, context->createProgramFromPTXFile(ptx_path, "pt_miss_env"));
    context->setMissProgram(BDPTRay, context->createProgramFromPTXFile(ptx_path, "miss_env"));
    context->setMissProgram(BDPT_L_Ray, context->createProgramFromPTXFile(ptx_path, "miss"));
	const std::string texture_filename = std::string(sutil::samplesDir()) + "/data/" + scene->env_file;
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texture_filename, optix::make_float3(1.0f)));
    env_info.hostSetting(context);

	Program prg;
	// BRDF sampling functions.
	m_bufferBRDFSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfSample = (int*) m_bufferBRDFSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Sample");
	brdfSample[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Sample");
	brdfSample[1] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Sample");
	brdfSample[2] = prg->getId();
	m_bufferBRDFSample->unmap();
	context["sysBRDFSample"]->setBuffer(m_bufferBRDFSample);
	
	// BRDF Eval functions.
	m_bufferBRDFEval = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfEval = (int*) m_bufferBRDFEval->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Eval");
	brdfEval[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Eval");
	brdfEval[1] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Eval");
	brdfEval[2] = prg->getId();
	m_bufferBRDFEval->unmap();
	context["sysBRDFEval"]->setBuffer(m_bufferBRDFEval);
	
	// BRDF Pdf functions.
	m_bufferBRDFPdf = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_BRDF_INDICES);
	int* brdfPdf = (int*) m_bufferBRDFPdf->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("disney.cu"), "Pdf");
	brdfPdf[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("glass.cu"), "Pdf");
	brdfPdf[1] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("lambert.cu"), "Pdf");
	brdfPdf[2] = prg->getId();
	m_bufferBRDFPdf->unmap();
	context["sysBRDFPdf"]->setBuffer(m_bufferBRDFPdf);

	// Light sampling functions.
	m_bufferLightSample = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_PROGRAM_ID, NUMBER_OF_LIGHT_INDICES);
	int* lightsample = (int*)m_bufferLightSample->map(0, RT_BUFFER_MAP_WRITE_DISCARD);
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "sphere_sample");
	lightsample[0] = prg->getId();
	prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "quad_sample");
	lightsample[1] = prg->getId();
    prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "direction_sample");
    lightsample[DIRECTION] = prg->getId();
    prg = context->createProgramFromPTXFile(ptxPath("light_sample.cu"), "env_sample");
    lightsample[ENV] = prg->getId();
	m_bufferLightSample->unmap();
	context["sysLightSample"]->setBuffer(m_bufferLightSample);

    //降噪指令声明
    denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
    denoiserStage["input_buffer"]->set(context["tonemapped_buffer"]->getBuffer());
    denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
    denoiserStage->declareVariable("blend")->setFloat(0.0f);
    denoiserStage->declareVariable("input_albedo_buffer");
    denoiserStage->declareVariable("input_normal_buffer");

    tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");
    tonemapStage->declareVariable("input_buffer")->set(context["accum_buffer"]->getBuffer());
    tonemapStage->declareVariable("output_buffer")->set(context["tonemapped_buffer"]->getBuffer());
    tonemapStage->declareVariable("exposure")->setFloat(0.25f);
    tonemapStage->declareVariable("gamma")->setFloat(2.2f);

    //heatStage = context->createBuiltinPostProcessingStage("DLSSIMPredictor");
    //heatStage->declareVariable("input_buffer")->set(context["tonemapped_buffer"]->getBuffer());
    //heatStage->declareVariable("output_buffer")->set(context["heat_buffer"]->getBuffer());
    context_initial_value_set();
}

Material createMaterial(const MaterialParameter &mat, int index)
{
	const std::string ptx_path = ptxPath( "hit_program.cu" );
	Program ch_program      = context->createProgramFromPTXFile( ptx_path, "closest_hit" );
    Program rr_ch_program = context->createProgramFromPTXFile(ptx_path, "rr_closest_hit");
	Program ah_program      = context->createProgramFromPTXFile(ptx_path, "any_hit");
    Program BDPT_ch_program = context->createProgramFromPTXFile(ptx_path, "BDPT_closest_hit");
    Program BDPT_L_ch_program = context->createProgramFromPTXFile(ptx_path, "BDPT_L_closest_hit");

	Material material = context->createMaterial();
    material->setAnyHitProgram(ShadowRay, ah_program);

    ah_program = context->createProgramFromPTXFile(ptx_path, "trans_any_hit");
	material->setClosestHitProgram( PTRay, ch_program );
    material->setAnyHitProgram(PTRay, ah_program);
    material->setClosestHitProgram(PT_RR_RAY, rr_ch_program);
    material->setAnyHitProgram(PT_RR_RAY, ah_program);
     
    material->setClosestHitProgram(BDPTRay, BDPT_ch_program);
    material->setAnyHitProgram(BDPTRay, ah_program);
    material->setClosestHitProgram(BDPT_L_Ray, BDPT_L_ch_program);
    material->setAnyHitProgram(BDPT_L_Ray, ah_program);

	material["materialId"]->setInt(index);
	material["programId"]->setInt(mat.brdf);

	return material;
}

Material createLightMaterial(const LightParameter &mat, int index)
{
	std::string ptx_path = ptxPath("light_hit_program.cu");
	Program ch_program = context->createProgramFromPTXFile(ptx_path, "closest_hit");
    Program BDPT_ch_program = context->createProgramFromPTXFile(ptx_path, "BDPT_closest_hit");
    Program BDPT_L_ch_program = context->createProgramFromPTXFile(ptx_path, "BDPT_L_closest_hit");
    ptx_path = ptxPath("hit_program.cu");
    Program ah_program = context->createProgramFromPTXFile(ptx_path, "light_any_hit");
	Material material = context->createMaterial();
	material->setClosestHitProgram(PTRay, ch_program);
    material->setClosestHitProgram(PT_RR_RAY, ch_program);
    material->setAnyHitProgram(ShadowRay, ah_program);
    material->setClosestHitProgram(BDPTRay, BDPT_ch_program);
    material->setClosestHitProgram(BDPT_L_Ray, BDPT_L_ch_program);

	material["lightMaterialId"]->setInt(index); 
	return material;
}

void updateMaterialParameters(const std::vector<MaterialParameter> &materials)
{
	MaterialParameter* dst = static_cast<MaterialParameter*>(m_bufferMaterialParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
	for (size_t i = 0; i < materials.size(); ++i, ++dst) {
		MaterialParameter mat = materials[i];
		
		dst->color = mat.color;
		dst->emission = mat.emission;
		dst->metallic = mat.metallic;
		dst->subsurface = mat.subsurface;
		dst->specular = mat.specular;
		dst->specularTint = mat.specularTint;
		dst->roughness = mat.roughness;
		dst->anisotropic = mat.anisotropic;
		dst->sheen = mat.sheen;
		dst->sheenTint = mat.sheenTint;
		dst->clearcoat = mat.clearcoat;
		dst->clearcoatGloss = mat.clearcoatGloss;
		dst->brdf = mat.brdf;
		dst->albedoID = mat.albedoID;
#ifdef DIFFUSE_ONLY

        dst->metallic = 0.0;
        dst->specular = 0.0;
        dst->roughness = 1.0;
#endif // DIFFUSE_ONLY
#ifdef GLOSSY_ONLY

       
        dst->specular = 1.0;
        dst->roughness = 0.1;
        dst->metallic = 0.5;
#endif // DIFFUSE_ONLY

	}
	m_bufferMaterialParameters->unmap();
}

void updateLightParameters(const std::vector<LightParameter> &lightParameters)
{
    static int div_base = SUBSPACE_NUM - 1;
#ifdef BD_ENV
    div_base -= 100;
#endif
	LightParameter* dst = static_cast<LightParameter*>(m_bufferLightParameters->map(0, RT_BUFFER_MAP_WRITE_DISCARD));
    for (size_t i = 0; i < lightParameters.size(); ++i, ++dst) {
        LightParameter mat = lightParameters[i];

        dst->position = mat.position;
        dst->emission = mat.emission;
        dst->radius = mat.radius;
        dst->area = mat.area;
        dst->u = mat.u;
        dst->v = mat.v;
        dst->normal = mat.normal;
        dst->direction = mat.direction;
        dst->lightType = mat.lightType;
#ifdef BD_ENV
        if (dst->lightType == DIRECTION)
        { 
            float env_factor = scene->env_factor;
            //env_info.set_env_lum(context, env_factor);
            context["env_lum"]->setFloat(env_factor);
            dst->lightType = ENV;
            env_info.add_direction_light(context, -mat.direction, mat.emission / env_factor);
        }
#endif
        dst->divLevel = mat.divLevel;
#ifndef ZGCBPT
        dst->divLevel = 1;
#endif
        dst->divBase = div_base;
        div_base -= dst->divLevel * dst->divLevel;
        dst->id = i;
	}
	m_bufferLightParameters->unmap();
}

optix::Aabb createGeometry(
        // output: this is a Group with two GeometryGroup children, for toggling visibility later
        optix::Group& top_group
        )
{ 
    const std::string ptx_path = ptxPath( "triangle_mesh.cu" );

    top_group = context->createGroup();
    top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );

	size_t i,j;
    optix::Aabb aabb;
    {
        GeometryGroup geometry_group = context->createGeometryGroup();
        geometry_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
        top_group->addChild( geometry_group );
		
        for (i = 0,j=0; i < scene->mesh_names.size(); ++i,++j) {
            OptiXMesh mesh;
            mesh.context = context;

            // override defaults
            mesh.intersection = context->createProgramFromPTXFile( ptx_path, "mesh_intersect_refine" );
            mesh.bounds = context->createProgramFromPTXFile( ptx_path, "mesh_bounds" );
            mesh.material = createMaterial(scene->materials[i], i);

            loadMesh( scene->mesh_names[i], mesh, scene->transforms[i] ); 
            
            //reset the uvmap for mesh(important for our algorithm)
            //note that uv and texcoord are different in this program, uv is used for subspace compute when texcoord is used to get the alebdo color
            if(false)
            {
                OptiXMesh uv_mesh;
                uv_mesh.context = context;
                loadMesh(scene->uv_mesh_names[i], uv_mesh, scene->transforms[i]);

                mesh.geom_instance["uv_buffer"]->setBuffer(uv_mesh.geom_instance->getGeometry()["texcoord_buffer"]->getBuffer());


                int3* indexP = reinterpret_cast<int3*>    (mesh.geom_instance->getGeometry()["index_buffer"]->getBuffer()->map());
                float3* vertexP = reinterpret_cast<float3*>    (mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->map()); 


                int3* indexP2 = reinterpret_cast<int3*>    (uv_mesh.geom_instance->getGeometry()["index_buffer"]->getBuffer()->map());
                float3* vertexP2 = reinterpret_cast<float3*>    (uv_mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->map());

                RTsize ss;
                RTsize ss2;
                mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->getSize(ss);
                uv_mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->getSize(ss2);
                printf("%d %d\n", ss, ss2);
                for (int i = 0; i < mesh.num_triangles; i++)
                { 
                    if (indexP[i].x + indexP[i].y + indexP[i].z != indexP2[i].x + indexP2[i].y + indexP2[i].z)
                    {
                        printf("\n");
                        printf("%d %d\n", indexP[i].x, indexP2[i].x);
                        printf("%d %d\n", indexP[i].y, indexP2[i].y);
                        printf("%d %d\n", indexP[i].z, indexP2[i].z);
                        printf("\n");


                        printf("\n");
                        printf("%f %f\n", vertexP[indexP[i].x].x, vertexP[indexP2[i].x].x);
                        printf("%f %f\n", vertexP[indexP[i].x].y, vertexP[indexP2[i].x].y);
                        printf("%f %f\n", vertexP[indexP[i].x].z, vertexP[indexP2[i].x].z);


                        printf("%f %f\n", vertexP[indexP[i].y].x, vertexP[indexP2[i].y].x);
                        printf("%f %f\n", vertexP[indexP[i].y].y, vertexP[indexP2[i].y].y);
                        printf("%f %f\n", vertexP[indexP[i].y].z, vertexP[indexP2[i].y].z);


                        printf("%f %f\n", vertexP[indexP[i].z].x, vertexP[indexP2[i].z].x);
                        printf("%f %f\n", vertexP[indexP[i].z].y, vertexP[indexP2[i].z].y);
                        printf("%f %f\n", vertexP[indexP[i].z].z, vertexP[indexP2[i].z].z);
                        printf("\n");
                    }
                }
                mesh.geom_instance->getGeometry()["index_buffer"]->getBuffer()->unmap();
                uv_mesh.geom_instance->getGeometry()["index_buffer"]->getBuffer()->unmap();
                mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->unmap();
                uv_mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->unmap();
                uv_mesh.geom_instance->destroy();
            }
            else
            {
                mesh.geom_instance["uv_buffer"]->setBuffer(mesh.geom_instance->getGeometry()["texcoord_buffer"]->getBuffer());

            }
            //loadMesh(scene->mesh_names[i], mesh, Matrix4x4::identity().translate(make_float3(0.0,1.0,0.0)));
            mesh.geom_instance["base_PrimIdx"]->setInt(num_triangles);
            geometry_group->addChild( mesh.geom_instance );
            
            {//将单个mesh的信息转移到全局缓存中
                static int triangleCount = 0;
                static int objectId = 0;
                float mesh_area = 0;
                if (scene->use_geometry_normal == true)
                {
                    mesh.geom_instance->getGeometry()["normal_buffer"]->getBuffer()->setSize(0);
                    printf("disable shading normal\n");
                }
                triangleStruct *p = reinterpret_cast<triangleStruct *>(context["triangle_samples"]->getBuffer()->map()); 
                int3 * indexP = reinterpret_cast<int3 *>    (mesh.geom_instance->getGeometry()["index_buffer"]->getBuffer()->map());
                float3* vertexP = reinterpret_cast<float3 *>    (mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->map());
                float2* uvP = reinterpret_cast<float2 *>    (mesh.geom_instance->getGeometry()["texcoord_buffer"]->getBuffer()->map());
                RTsize uvsize;
                mesh.geom_instance->getGeometry()["texcoord_buffer"]->getBuffer()->getSize(uvsize); 
                for (int i = 0; i < mesh.num_triangles; i++)
                {

                    p[triangleCount].position[0] = vertexP[indexP[i].x];
                    p[triangleCount].position[1] = vertexP[indexP[i].y];
                    p[triangleCount].position[2] = vertexP[indexP[i].z];
                    p[triangleCount].objectId = objectId;
                    p[triangleCount].zoneNum = objectId;
                    if (uvsize > 0)
                    {
                        p[triangleCount].uv[0] = uvP[indexP[i].x];
                        p[triangleCount].uv[1] = uvP[indexP[i].y];
                        p[triangleCount].uv[2] = uvP[indexP[i].z];
                    }
                    triangleStruct& tri = p[triangleCount];
                    div_tris.add(tri);
                    triangleCount++;
                    mesh_area += tri.area();
                }
                uv_grid.push_back(mesh_area); 
                objectId++;
                context["triangle_samples"]->getBuffer()->unmap(); 
                mesh.geom_instance->getGeometry()["index_buffer"]->getBuffer()->unmap();
                mesh.geom_instance->getGeometry()["vertex_buffer"]->getBuffer()->unmap();
                mesh.geom_instance->getGeometry()["texcoord_buffer"]->getBuffer()->unmap();
                mesh.geom_instance->getGeometry()["objMat_id"]->setInt(scene->materials[i].albedoID);
            }

            aabb.include( mesh.bbox_min, mesh.bbox_max );
            
            std::cerr << scene->mesh_names[i] << ": " << mesh.num_triangles << std::endl;
            num_triangles += mesh.num_triangles;
        }
        std::cerr << "Total triangle count: " << num_triangles <<" + "<<div_tris.count << std::endl;
    }
	//Lights
	{
		GeometryGroup geometry_group = context->createGeometryGroup();
		geometry_group->setAcceleration(context->createAcceleration("NoAccel"));
		
		for (i = 0; i < scene->lights.size(); ++i)
		{
			GeometryInstance instance;
            if (scene->lights[i].lightType == QUAD)
                instance = createQuad(context, createLightMaterial(scene->lights[i], i), scene->lights[i].u, scene->lights[i].v, scene->lights[i].position, scene->lights[i].normal);
            else if (scene->lights[i].lightType == SPHERE)
                instance = createSphere(context, createLightMaterial(scene->lights[i], i), scene->lights[i].position, scene->lights[i].radius);
            else if (scene->lights[i].lightType == DIRECTION)
                continue;
            else
                continue;
            geometry_group->addChild(instance);
		}
        if(geometry_group->getChildCount()!=0)
            top_group->addChild(geometry_group);
        printf("%d\n", scene->lights.size());
		//GeometryInstance instance = createSphere(context, createMaterial(materials[j], j), optix::make_float3(150, 80, 120), 80);
		//geometry_group->addChild(instance);
	}

	

    float sceneMaxLength = length(aabb.m_min - aabb.m_max);
    context[ "top_object" ]->set( top_group ); 
    context["sceneMaxLength"]->setFloat(sceneMaxLength);
    context["min_box"]->setFloat(aabb.m_min);
    context["max_box"]->setFloat(aabb.m_max);
    printf("scene bounding box max: %f %f %f\n", aabb.m_max.x, aabb.m_max.y, aabb.m_max.z);
    printf("scene bounding box min: %f %f %f\n", aabb.m_min.x, aabb.m_min.y, aabb.m_min.z);
    context["scene_center"]->setFloat((aabb.m_max + aabb.m_min) / 2);
    
    context["DirProjectionArea"]->setFloat((M_PIf * sceneMaxLength* sceneMaxLength) / 4.0);
    minProjectProcess(context, scene->dirLightDir, aabb.m_min, aabb.m_max);

    env_info.setAABB(context,aabb);
    uv_grid.table_write(context, 0, SUBSPACE_NUM - MAX_LIGHT);
    return aabb;
}

//------------------------------------------------------------------------------
//
//  GLFW callbacks
//
//------------------------------------------------------------------------------

struct CallbackData
{
    sutil::Camera& camera;
    unsigned int& accumulation_frame;
};

void keyCallback( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    bool handled = false;

    if( action == GLFW_PRESS )
    {
        switch( key )
        {
            case GLFW_KEY_Q:
            case GLFW_KEY_ESCAPE:
                if( context )
                    context->destroy();
                if( window )
                    glfwDestroyWindow( window );
                glfwTerminate();
                exit(EXIT_SUCCESS);

            case(GLFW_KEY_C):
            {
                printf("Camera_info:\n");
                
                float3 eye = context["eye"]->getFloat3();
                float3 W = context["W"]->getFloat3();
                float3 V = context["V"]->getFloat3();
                W = eye + normalize(W) * 0.1;
                printf("eye position %f %f %f\n",eye.x,eye.y,eye.z);
                printf("lookat %f %f %f\n",W.x,W.y,W.z);
                printf("up %f %f %f\n", V.x, V.y, V.z);
                break;
            }
            case( GLFW_KEY_S ):
            {
                const std::string outputImage = std::string(SAMPLE_NAME) + ".png";
                std::cerr << "Saving current frame to '" << outputImage << "'\n";
                sutil::writeBufferToFile( outputImage.c_str(), getOutputBuffer() );
                handled = true;

                /**/
#ifndef ACCM_VAL_ESTIMATE
                char4* p = reinterpret_cast<char4*>(context["output_buffer"]->getBuffer()->map());
                RTsize OW, OH;
                context["output_buffer"]->getBuffer()->getSize(OW,OH);
                std::ofstream outFile;
                outFile.open("./standrd.txt");
            
                for (int i = 0; i < OW * OH; i++)
                {
                    outFile << int(p[i].x) << " ";
                    outFile << int(p[i].y) << " ";
                    outFile << int(p[i].z) << " ";
                    outFile << int(p[i].w) << std::endl;
                    
                }
                outFile.close();
                context["output_buffer"]->getBuffer()->unmap();
#else

                float4* p = reinterpret_cast<float4*>(context["accum_buffer"]->getBuffer()->map());
                RTsize OW, OH;
                context["accum_buffer"]->getBuffer()->getSize(OW, OH);
                std::ofstream outFile;
                outFile.open("./standrd.txt");

                for (int i = 0; i < OW * OH; i++)
                {
                    outFile << p[i].x << " ";
                    outFile << p[i].y << " ";
                    outFile << p[i].z << " ";
                    outFile << p[i].w << std::endl;

                }
                outFile.close();
                context["accum_buffer"]->getBuffer()->unmap();
#endif

                break;
            }
            case(GLFW_KEY_E):
            {
#ifdef  ACCM_VAL_ESTIMATE

                frame_estimate_float();
#else
                frame_estimate();

#endif //   ACCM_VAL_ESTIMATE

                 
                
                break;
            }
            case( GLFW_KEY_F ):
            {
               CallbackData* cb = static_cast<CallbackData*>( glfwGetWindowUserPointer( window ) );
               cb->camera.reset_lookat();
               cb->accumulation_frame = 0;
               handled = true;
               break;
            }
        }
    }

    if (!handled) {
        // forward key event to imgui
        ImGui_ImplGlfw_KeyCallback( window, key, scancode, action, mods );
    }
}

void windowSizeCallback( GLFWwindow* window, int w, int h )
{
    if (w < 0 || h < 0) return;

    const unsigned width = (unsigned)w;
    const unsigned height = (unsigned)h;

    CallbackData* cb = static_cast<CallbackData*>( glfwGetWindowUserPointer( window ) );
    if ( cb->camera.resize( width, height ) ) {
        cb->accumulation_frame = 0;
    }

    sutil::resizeBuffer( getOutputBuffer(), width, height );
    sutil::resizeBuffer( context[ "accum_buffer" ]->getBuffer(), width, height );
    sutil::resizeBuffer(context["standrd_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["false_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["false_mse_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["standrd_float_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["tonemapped_buffer"]->getBuffer(), width, height); 
    sutil::resizeBuffer(context["LT_result_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["heat_buffer"]->getBuffer(), int(ceilf(width)), int(ceilf(height)));
    sutil::resizeBuffer(context["denoised_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["input_albedo_buffer"]->getBuffer(), width, height);
    sutil::resizeBuffer(context["input_normal_buffer"]->getBuffer(), width, height);

    sutil::resizeBuffer(denoisedBuffer, width, height);
    postprocessing_needs_init = true;
#ifdef USE_PCPT
    sutil::resizeBuffer(context["PMFCaches"]->getBuffer(), PMFCaches_RATE * width, PMFCaches_RATE * height);
    context["KDPMFCaches"]->getBuffer()->setSize(2 * int(PMFCaches_RATE * width) * int(PMFCaches_RATE * height));
#endif // USE_PCPT

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glViewport(0, 0, width, height);
}


//------------------------------------------------------------------------------
//
// GLFW setup and run 
//
//------------------------------------------------------------------------------

GLFWwindow* glfwInitialize( )
{
    GLFWwindow* window = sutil::initGLFW();

    // Note: this overrides imgui key callback with our own.  We'll chain this.
    glfwSetKeyCallback( window, keyCallback );

    glfwSetWindowSize( window, (int)scene->properties.width, (int)scene->properties.height);
    glfwSetWindowSizeCallback( window, windowSizeCallback );

    return window;
}
void gl_stage_init()
{
    // Initialize GL state
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, scene->properties.width, scene->properties.height);

}

void glfwRun( GLFWwindow* window, sutil::Camera& camera, const optix::Group top_group )
{ 
    gl_stage_init();
    unsigned int frame_count = 0;
    unsigned int accumulation_frame = 0; 
    int max_depth = 1 ;
    int vp_x = 500;
    int vp_y = 500;
	lastTime = sutil::currentTime();

    // Expose user data for access in GLFW callback functions when the window is resized, etc.
    // This avoids having to make it global.
    CallbackData cb = { camera, accumulation_frame };
    glfwSetWindowUserPointer( window, &cb );

    while (!glfwWindowShouldClose(window))
    {

        glfwPollEvents();

        ImGui_ImplGlfw_NewFrame();

        ImGuiIO& io = ImGui::GetIO();

        // Let imgui process the mouse first
        if (!io.WantCaptureMouse) {

            double x, y;
            glfwGetCursorPos(window, &x, &y);

            if (camera.process_mouse((float)x, (float)y, ImGui::IsMouseDown(0), ImGui::IsMouseDown(1), ImGui::IsMouseDown(2))) {
                accumulation_frame = 0;
            }
        }

        // imgui pushes
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.6f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 2.0f);


        sutil::displayFps(frame_count++);
        sutil::displaySpp(accumulation_frame);

        {
            static const ImGuiWindowFlags window_flags =
                ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_AlwaysAutoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoScrollbar;

            ImGui::SetNextWindowPos(ImVec2(2.0f, 70.0f));
            ImGui::Begin("controls", 0, window_flags);
            if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderInt("max depth", &max_depth, 1, 10)) {
                    context["max_depth"]->setInt(max_depth);
                    accumulation_frame = 0;
                }
            }
            if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderInt("vp_x", &vp_x, 1, camera.width())) {
                    context["vp_x"]->setInt(vp_x);
                }
            }
            if (ImGui::CollapsingHeader("Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderInt("vp_y", &vp_y, 1, camera.height())) {
                    context["vp_y"]->setInt(vp_y);
                }
            }

            ImGui::End();
        }

        elapsedTime += sutil::currentTime() - lastTime;
        if (accumulation_frame == 2)
        {
            elapsedTime = subspaceDivTime;
            printf("elasp time reset subspace DivTime %f\n", subspaceDivTime);
        }
		sutil::displayElapsedTime(elapsedTime);
		lastTime = sutil::currentTime();

        // imgui pops
        ImGui::PopStyleVar( 3 );

        // Render main window
        context["frame"]->setUint( accumulation_frame++ );
        corputBufferGen(context);

////////////////////////////////////////////////
/////////////////algorithm begin////////////////
////////////////////////////////////////////////

        static bool pre_processing_end = false;
        if (pre_processing_end == false)
        {
            pre_processing_end = true;

#ifdef ZGCBPT

            context["LVC_frame"]->setInt(0);
            pre_processing();
#endif
        }

////////////////////////////////////////////////
//////////light sub-path tracing pass///////////
////////////////////////////////////////////////
#ifdef PCBPT
        light_cache_process_RISBPT();
#else

#ifdef LVCBPT
        light_cache_process();
#else 
#ifdef KITCHEN_DISCARD
        light_cache_process();
#else
        light_cache_process_ZGCBPT();
#endif

#endif

#endif
         

#ifdef USE_DENOISER

        if (accumulation_frame > DENOISE_BEGIN_FRAME)
        {
            context["denoised_buffer"]->set(denoisedBuffer);
            postprocessing_init(camera);
            static bool denoised = false;
            //printf("test A%d\n", count);
            if(denoised == false)
                denoise_command->execute();
            //denoised = true;
            //printf("test B%d\n", count);
            //sutil::displayBufferGL(context["tonemapped_buffer"]->getBuffer());
            context->launch(FormatTransform, camera.width(), camera.height());

        }
        else
        {

#ifdef VIEW_HEAT_post
            context["denoised_buffer"]->set(context["heat_buffer"]->getBuffer());
            postprocessing_init(camera);
            heat_command->execute();
            context->launch(FormatTransform, int(ceilf(camera.width()/16.0)), int(ceilf(camera.height()/16.0)));

#else

            elapsedTime += sutil::currentTime() - lastTime;
            lastTime = sutil::currentTime();
            printf("pinhole launch begin\n");
            context->launch(0, camera.width(), camera.height());
            printf("pinhole launch custom %f s\n", sutil::currentTime() - lastTime); 
#endif // VIEW_HEAT
        }
        sutil::displayBufferGL(getOutputBuffer());
//        frame_estimate(1);
#ifdef  ACCM_VAL_ESTIMATE

        frame_estimate_float(1);
#else
        frame_estimate(1);

#endif //   ACCM_VAL_ESTIMATE

#else

////////////////////////////////////////////////
//////////eye sub-path tracing and estimation///
////////////////////////////////////////////////
        context->launch( 0, camera.width(), camera.height() );
        sutil::displayBufferGL(getOutputBuffer());
#endif // USE_DENOISER


////////////////////////////////////////////////
//////////frame estimation//////////////////////
////////////////////////////////////////////////
        static estimate_status es(get_scene_name(),subspaceDivTime,PATH_M, context["average_light_length"]->getFloat());
        estimate(context, elapsedTime, lastTime, context["frame"]->getUint(),es);
        
        // Render gui over it
        ImGui::Render();

        glfwSwapBuffers( window );
    }
    
    destroyContext();
    glfwDestroyWindow( window );
    glfwTerminate();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr <<
        "App Options:\n"
        "  -h | --help                  Print this usage message and exit.\n"
        "  -f | --file <output_file>    Save image to file and exit.\n"
        "  -n | --nopbo                 Disable GL interop for display buffer.\n"
		"  -s | --scene                 Provide a scene file for rendering.\n"
        "App Keystrokes:\n"
        "  q  Quit\n"
        "  s  Save image to '" << SAMPLE_NAME << ".png'\n"
        "  f  Re-center camera\n"
        "\n"
        << std::endl;

    exit(1);
}

std::string scene_file;
std::string out_file;
void read_scene_metadata(int argc, char** argv)
{

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if (arg == "-h" || arg == "--help")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "-f" || arg == "--file")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            out_file = argv[++i];
        }
        else if (arg == "-scene")
        {
            if (i == argc - 1)
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit(argv[0]);
            }
            scene_file = argv[++i];
        }
        else if (arg[0] == '-')
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    if (scene_file.empty())
    {
        // Default scene
        //scene_file = sutil::samplesDir() + std::string("/data/cornell3.scene");
        scene_file = sutil::samplesDir() + std::string(SCENE_FILE_PATH);
        scene = LoadScene(scene_file.c_str());
    }
    else
    {
        scene = LoadScene(scene_file.c_str());
    }
}
void renderer_init(GLFWwindow* &window, sutil::Camera* &camera_p, optix::Group& top_group)
{
    bool use_pbo = true;
    window = glfwInitialize();

    GLenum err = glewInit();

    if (err != GLEW_OK)
    {
        std::cerr << "GLEW init failed: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    ilInit();

    createContext(use_pbo);

    // Load textures
    for (int i = 0; i < scene->texture_map.size(); i++)
    {
        Texture tex;
        Picture* picture = new Picture;
        std::string textureFilename = std::string(sutil::samplesDir()) + "/data/" + scene->texture_map[i];
        std::cout << textureFilename << std::endl;
        picture->load(textureFilename);
        tex.createSampler(context, picture);
        scene->textures.push_back(tex);
        delete picture;
    }

    // Set textures to albedo ID of materials
    for (int i = 0; i < scene->materials.size(); i++)
    {
        if (scene->materials[i].albedoID != RT_TEXTURE_ID_NULL)
        {
            scene->materials[i].albedoID = scene->textures[scene->materials[i].albedoID - 1].getId();
        }
    }

    m_bufferLightParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_bufferLightParameters->setElementSize(sizeof(LightParameter));
    m_bufferLightParameters->setSize(scene->lights.size());
    updateLightParameters(scene->lights);
    context["sysLightParameters"]->setBuffer(m_bufferLightParameters);
    context["Lights"]->setBuffer(m_bufferLightParameters);

    m_bufferMaterialParameters = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
    m_bufferMaterialParameters->setElementSize(sizeof(MaterialParameter));
    m_bufferMaterialParameters->setSize(scene->materials.size());
    updateMaterialParameters(scene->materials);
    context["sysMaterialParameters"]->setBuffer(m_bufferMaterialParameters);

    context["sysNumberOfLights"]->setInt(scene->lights.size());
    const optix::Aabb aabb = createGeometry(top_group);
    scene_aabb = aabb;
    PGTrainer_api.init(context);

    context->validate();

    optix::float3 camera_eye;
    optix::float3 camera_lookat;
    optix::float3 camera_up;
    float vfov;
    camera_eye = scene->eye;
    camera_lookat = scene->lookat;
    camera_up = scene->up;
    vfov = scene->fov;

    camera_p = new sutil::Camera(scene->properties.width, scene->properties.height,
        &camera_eye.x, &camera_lookat.x, &camera_up.x,
        context["eye"], context["U"], context["V"], context["W"], vfov);

}
int main( int argc, char** argv )
{ 
    try
    {
        GLFWwindow* window;
        sutil::Camera* camera_p;
        optix::Group top_group;

        read_scene_metadata(argc, argv);//scene_loading
        renderer_init(window, camera_p, top_group);//renderer_init
        load_standrd(context); //reference image loading
         
        glfwRun( window, *camera_p, top_group ); //run the algorithm
        return 0;
    }
    SUTIL_CATCH( context->get() )
}

int buildVDirectorBuffer()
{
    const splitChoice split_choice = RoundRobin;
    Buffer &pmf_caches_buffer = context["PMFCaches"]->getBuffer();
    Buffer &raw_director_buffer = context["raw_LVC"]->getBuffer();
    Buffer &kd_director_buffer = context["Kd_position"]->getBuffer();
    RAWVertex* raw_directors_data = reinterpret_cast<RAWVertex*>(raw_director_buffer->map());
    KDPos* kd_director_data = reinterpret_cast<KDPos*>(kd_director_buffer->map());

    RTsize kd_size,raw_w,raw_h,raw_d;
    pmf_caches_buffer->getSize(raw_w,raw_h,raw_d);
    kd_director_buffer->getSize(kd_size);
    for (unsigned int i = 0; i < (unsigned int)kd_size; ++i) {
        kd_director_data[i].valid = false;
    }

    unsigned int valid_directors = 0;
    KDPos** temp_directors = new KDPos*[raw_w * raw_h * raw_d];
    KDPos* temp_rawpos = new KDPos[raw_w * raw_h * raw_d];
    RTsize vc_size;
    raw_director_buffer->getSize(vc_size);
    for (unsigned int i = 0; i < (unsigned int)raw_w * raw_h * raw_d; ++i) {
        if ( raw_directors_data[i].valid) {
            BDPTVertex &tEVC = (raw_directors_data[i]).v;
            temp_rawpos[valid_directors].valid = true;
            temp_rawpos[valid_directors].normal = tEVC.normal;
            temp_rawpos[valid_directors].in_direction = normalize(tEVC.position - tEVC.lastPosition) ;
            temp_rawpos[valid_directors].position = tEVC.position;
            temp_directors[valid_directors] = &temp_rawpos[valid_directors];
            valid_directors++;
        }
    }
    // Make sure we aren't at most 1 less than power of 2
    valid_directors = (valid_directors >= (unsigned int)kd_size) ? (unsigned int)kd_size : valid_directors;
    printf("valid pmfchches count %d\n", valid_directors);
    float3 bbmin = make_float3(0.0f);
    float3 bbmax = make_float3(0.0f);
    if (split_choice == LongestDim) {
        bbmin = make_float3(std::numeric_limits<float>::max());
        bbmax = make_float3(-std::numeric_limits<float>::max());
        // Compute the bounds of the photons
        for (unsigned int i = 0; i < valid_directors; ++i) {
            float3 position = (*temp_directors[i]).position;
            bbmin = fminf(bbmin, position);
            bbmax = fmaxf(bbmax, position);
        }
    }

    // Now build KD tree
    buildKDTree<KDPos>(temp_directors, 0, valid_directors, 0, kd_director_data, 0, split_choice, bbmin, bbmax);
     
    delete[] temp_directors;
    delete[] temp_rawpos;
    raw_director_buffer->unmap();
    kd_director_buffer->unmap();
    return valid_directors;
}

//for the front kn_d BDPTVERTEX in LVCBUFFER,build a kdTREE in KDBuffer
void buildPhotonMap(Buffer& LTVBuffer,Buffer& KDBuffer,int kd_n)
{
    const splitChoice split_choice = RoundRobin; 

    BDPTVertex* raw_directors_data = reinterpret_cast<BDPTVertex*>(LTVBuffer->map());
    KDPos* kd_director_data = reinterpret_cast<KDPos*>(KDBuffer->map());

    RTsize kd_size;
    KDBuffer->getSize(kd_size);
    for (unsigned int i = 0; i < (unsigned int)kd_size; ++i) {
        kd_director_data[i].valid = false;
    }

    unsigned int valid_directors = 0;
    KDPos** temp_directors = new KDPos*[kd_n];
    KDPos* temp_rawpos = new KDPos[kd_n];
    for (unsigned int i = 0; i < (unsigned int)kd_n; ++i) {
        if (true) {
            BDPTVertex &tEVC = (raw_directors_data[i]);
            temp_rawpos[valid_directors].valid = true;
            temp_rawpos[valid_directors].normal = tEVC.normal;
            temp_rawpos[valid_directors].in_direction = normalize(tEVC.position - tEVC.lastPosition);
            temp_rawpos[valid_directors].position = tEVC.position;
            temp_rawpos[valid_directors].original_p = i;
            temp_directors[valid_directors] = &temp_rawpos[valid_directors];
            valid_directors++;
        }
    }
    // Make sure we aren't at most 1 less than power of 2
    valid_directors = (valid_directors >= (unsigned int)kd_size) ? (unsigned int)kd_size : valid_directors;
    printf("valid KDTree count %d\n", valid_directors);
    float3 bbmin = make_float3(0.0f);
    float3 bbmax = make_float3(0.0f);
    if (split_choice == LongestDim) {
        bbmin = make_float3(std::numeric_limits<float>::max());
        bbmax = make_float3(-std::numeric_limits<float>::max());
        // Compute the bounds of the photons
        for (unsigned int i = 0; i < valid_directors; ++i) {
            float3 position = (*temp_directors[i]).position;
            bbmin = fminf(bbmin, position);
            bbmax = fmaxf(bbmax, position);
        }
    }

    // Now build KD tree
    buildKDTree<KDPos>(temp_directors, 0, valid_directors, 0, kd_director_data, 0, split_choice, bbmin, bbmax);

    delete[] temp_directors;
    delete[] temp_rawpos;
    LTVBuffer->unmap();
    KDBuffer->unmap();
}