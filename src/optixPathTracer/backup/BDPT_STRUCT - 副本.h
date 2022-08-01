#pragma once 

#ifndef BDPT_STRUCT_H
#define AUTO_TEST_MOD
#define TOBEREWRITE
//#define SCENE_FILE_PATH "/data/white-room/white-room.scene"
//#define SCENE_FILE_PATH "/data/white-room/white-room-obj.scene"
//#define SCENE_FILE_PATH "/data/villa/villa-r.scene"
//#define SCENE_FILE_PATH "/data/breafast/breafast2.scene"
//#define SCENE_FILE_PATH "/data/breafast/breafast_direction3.scene"
//#define SCENE_FILE_PATH "/data/door_kitchen/door_kitchen_obj.scene"
//#define SCENE_FILE_PATH "/data/fireplace/fireplace.scene"
//#define SCENE_FILE_PATH "/data/conference/conference.scene"
//#define SCENE_FILE_PATH "/data/sibenik/sibenik.scene"
//#define SCENE_FILE_PATH "/data/bathroom/bathroom.scene"
//#define SCENE_FILE_PATH "/data/kitchen/kitchen.scene"
//#define SCENE_FILE_PATH "/data/stair/stair.scene"
//#define SCENE_FILE_PATH "/data/sponza_crytek/sponza_crytek.scene"

//#define SCENE_FILE_PATH "/data/fireplace/fireplace-2.0.scene"
//#define SCENE_FILE_PATH "/data/livingroom-2.0/livingroom-3.0.scene"
//#define SCENE_FILE_PATH "/data/bedroom11.scene"
//#define SCENE_FILE_PATH "/data/breafast_2.0/breafast_2.0.scene"

//#define SCENE_FILE_PATH "/data/gallery/gallery.scene"
//#define SCENE_FILE_PATH "/data/blender_bedroom/blender_bedroom.scene"
//#define SCENE_FILE_PATH "/data/pavillon_barcelone/pavillon_barcelone.scene"
//#define REFERENCE_FILE_PATH "./standard_float/door_kitchen.txt"

//#define LVCBPT
//#define PCBPT
#define ZGCBPT


//#define FIREPLACE
//#define CLASSROOM
//#define BATHROOM
//#define KITCHEN
//#define CONFERENCE
//#define HALLWAY
//#define HOUSE

#define HOUSE

#ifdef KITCHEN
#define SCENE_FILE_PATH "/data/kitchen/kitchen.scene"
#define REFERENCE_FILE_PATH "./standard_float/kitchen-50000.txt"
#define SCENE_PATH_AVER 3.6
#endif


#ifdef HALLWAY
#define SCENE_FILE_PATH "/data/hallway/hallway2.scene"
#define REFERENCE_FILE_PATH "./standard_float/hallway-300000.txt" 
#define SCENE_PATH_AVER 2.1
#endif
#ifdef CONFERENCE
#define SCENE_FILE_PATH "/data/conference/conference.scene"
#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt"
#define SCENE_PATH_AVER 2.4

#endif

#ifdef BATHROOM
#define SCENE_FILE_PATH "/data/bathroom/bathroom.scene"
#define REFERENCE_FILE_PATH "./standard_float/bathroom-25000.txt"
#define SCENE_PATH_AVER 2.8

#endif


#ifdef CLASSROOM
#define SCENE_FILE_PATH "/data/classroom/classroom.scene"
#define REFERENCE_FILE_PATH "./standard_float/classroom-90000.txt" 
#define SCENE_PATH_AVER 2.5

#endif

#ifdef FIREPLACE
#define SCENE_FILE_PATH "/data/fireplace/fireplace-3.0.scene"
#define REFERENCE_FILE_PATH "./standard_float/fireplace3.0-60000.txt"
#define SCENE_PATH_AVER 2.8
#endif


#ifdef HOUSE
#define SCENE_FILE_PATH "/data/house/house.scene"
#define REFERENCE_FILE_PATH "./standard_float/kitchen-50000.txt"
#define SCENE_PATH_AVER 2
#endif
//#define SCENE_FILE_PATH "/data/hallway/hallway2.scene"
//#define SCENE_FILE_PATH "/data/classroom/classroom.scene"
//#define SCENE_FILE_PATH "/data/conference/conference.scene"
//#define SCENE_FILE_PATH "/data/bathroom/bathroom.scene"
//#define SCENE_FILE_PATH "/data/fireplace/fireplace-3.0.scene"
//#define SCENE_FILE_PATH "/data/house/house.scene"

///shadingnormal
//#define REFERENCE_FILE_PATH "./standard_float/bedroom_30000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/stair-35000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/bedroom11-pt50000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt”
//#define REFERENCE_FILE_PATH "./standard_float/livingroom-3.0.txt"



///geo normal
//#define REFERENCE_FILE_PATH "./standard_float/fireplace_30000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/breafast_30000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/hallway-25000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/classroom-90000.txt" 


////#define REFERENCE_FILE_PATH "./standard_float/hallway-50000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/hallway-300000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/classroom-90000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/bathroom-50000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/fireplace3.0-60000.txt"

#define CONTINUE_RENDERING_BEGIN_FRAME 0

#include<vector>
//#define LTC_STRA

#define UNBIAS_RENDERING
//#define DIFFUSE_ONLY
//#define GLOSSY_ONLY
#define RR_MIN_LIMIT
#ifdef RR_MIN_LIMIT
#define MIN_RR_RATE 0.3f
#endif

#ifndef UNBIAS_RENDERING

#ifndef RR_MIN_LIMIT
#define RR_DISABLE
#endif
#endif

//200 500 1000 2000
#define MAX_ZONE 200
//PATH_M 10000 50000 100000  200000 500000 1000000
#define PATH_M 100000
#define iterNum int(SCENE_PATH_AVER + 1)
//#define BACK_ZONE
//#define LVC_RR

#define MAX_LIGHT_VERTEX_PER_TRIANGLE 50
#ifdef ZGCBPT
//iterNum 1 2 3 4 8
#define iterNum 2
#define SLIC_CLASSIFY

#define INDIRECT_ONLY
#define EYE_DIRECT
#define EYE_DIRECT_FRAME 2000
#define ZGC_SAMPLE_ON_LUM
#define TRACER_PROGRAM_NAME "ZGCBPT_pinhole_camera"


//#define UBER_RMIS
//#define TRACER_PROGRAM_NAME "ZGCBPT_test_pinhole_camera"
//
//#define PRIM_DIRECT_VIEW

#else

#ifdef PCBPT
//#define NOISE_DISCARD
//#define LTC_STRA
#define iterNum 1
#define INDIRECT_ONLY
#define PCBPT_MIS
#define PCBPT_STANDARD_MIS
#define KD_3
#define connectRate_kd3(r,ind) min(((r / KDPMFCaches[ind.x].Q + r / KDPMFCaches[ind.y].Q +r / KDPMFCaches[ind.z].Q)/3),1000.0)
#define TRACER_PROGRAM_NAME "PCBPT_pinhole_camera"
//#define PRIM_DIRECT_VIEW
#else
#ifdef BDPT
#define TRACER_PROGRAM_NAME "BDPT_pinhole_camera"
#else 
#ifdef LVCBPT
#define TRACER_PROGRAM_NAME "LVCBPT_pinhole_camera"
//#define CONTINUE_RENDERING
//#define CONTINUE_RENDERING_BEGIN_FRAME 25000
#else
#define TRACER_PROGRAM_NAME "pinhole_camera"
#define NO_COLOR
#ifdef RRPT
#define TRACER_PROGRAM_NAME "unbias_pt_pinhole_camera"
#endif
#endif
#endif

#endif // PCBPT
#endif

#define BDPT_STRUCT_H


//画面评估参数设置
#define ESTIMATE_INVALID
#define ESTIMATE_FRAME {10,20,40,80,160,320,640}
#define ACCM_VAL_ESTIMATE

//heat map is unavailable in this optix ver 
//#define VIEW_HEAT_pos 
//#define VIEW_HEAT 

//是否使用PCPT
//#define USE_PCPT

//是否重新划分区域
//#define ZONE_ALLOC

//是否使用视顶点矩阵
//#define USE_ML_MATRIX

//是否使用朴素方法划分区域
#define RAW_CLUSTER
#ifdef RAW_CLUSTER
#define KMEANS_ITER_NUM 1
#ifndef ZONE_ALLOC
#define ZONE_ALLOC
#endif // !ZONE_ALLOC
#else
#define KMEANS_ITER_NUM 10
#endif // RAW_CLUSTER


//场景中是否有透明介质
#define BRDF
#define STACKSIZE 24
#define M 1.0f
#define RR_RATE 1.0f
#define RR_BEGIN_DEPTH 2

//降噪参数设置
#define USE_DENOISER
#ifdef USE_DENOISER
#define DENOISE_BEGIN_FRAME 200000
#endif // USE_DENOISER
#ifdef PCBPT
#define PCPT_CORE_NUM 200
#define LIGHT_VERTEX_PER_CORE 15
#else
#define PCPT_CORE_NUM int(PATH_M / 100)
#define LIGHT_VERTEX_PER_CORE int(SCENE_PATH_AVER * 100)
#endif
#define LTC_CORE_NUM 1000
#define LTC_SPC 200
#define LTC_SAVE_SUM (LTC_CORE_NUM * LTC_SPC)

#define PMF_DEPTH 5
#define UberWidth 100
#define LIGHT_VERTEX_NUM (PCPT_CORE_NUM * LIGHT_VERTEX_PER_CORE) 
#define ZONE_OPTIMAL_DIRECTOR_NUM 1000
#define UBER_VERTEX_NUM (LIGHT_VERTEX_NUM / UberWidth *10) 
#define MAX_TRIANGLE 6000000
#define MAX_TRI_AREA 0.05
#define MGPT_GUIDE_RATE 0.9f
#define UNIFORM_GUIDE_RATE 0.25f
#define PMFCaches_RATE 0.063f
#define MAX_PATH_LENGTH 25
#define VISIBILITY_TEST_NUM 100
#define VISIBILITY_TEST_SLICE 10
#define MAX_LIGHT 20
#define RECORD_DEPTH 1 

#define BUFFER_WEIGHT 1.f 
#define DISCARD_VALUE 1000.0f
#define POWER_RATE 1.0f
#define  PPM_X         ( 1 << 0 )
#define  PPM_Y         ( 1 << 1 )
#define  PPM_Z         ( 1 << 2 )
#define  PPM_LEAF      ( 1 << 3 )
#define  PPM_NULL      ( 1 << 4 )
#define ENERGY_WEIGHT(a) (a.x + a.y + a.z)
#include <optixu/optixu_math_namespace.h>
#include "material_parameters.h"
#include "light_parameters.h"
using namespace optix;
enum splitChoice
{
    LongestDim, HighestVariance, RoundRobin
};
enum rayGenChoice
{
    pinholeCamera, FormatTransform,lightBufferGen, PMFCacheLaunch,EVCLaunch,uberVertexProcessProg,LTCLaunchProg,
   rayGenProNum,visibilityTestProg
};

struct BDPTVertex;
struct BDPTVertexStack;

struct BDPTVertex
{
    optix::float3 position;
    //should be normalized
    optix::float3 normal;
    
    //此处的flux当成contribute用
    optix::float3 flux;
    optix::float3 color;
    optix::float3 lastPosition;
    optix::float2 uv;
    float lastNormalProjection;
    //pdf存储由light path生成顶点的pdf，epdf存储由eye path生成顶点的pdf
    float pdf;
    float singlePdf;
    float lastSinglePdf;

    float d;
    float dLast;
#ifdef PCBPT

    float pmf_link_rate;
    float G;
    uint3 pmf_kd3;
    uint3 last_kd3;
    int last_pmf_id;
    int pmf_id;
    
#endif // PCBPT
#ifdef ZGC_SAMPLE_ON_LUM
    optix::float3 Dcolor;
    optix::float3 Dcolor_last;
    float SOL_rate;
    float zs_weight;

#endif // ZGC_SAMPLE_ON_LUM
#ifdef UBER_RMIS
    //the meaning of the var currentWeight is difference between light vertex and eye vertex
    // for light vertex ,it is the straight forward meaning 
    // for eye vertex ,it is the Weight of lastVertex.without the inverLastPdf item
    float currentWeight;
#endif // UBER_RMIS


    int materialId;

    int zoneId;
    int depth;
    int lastZoneId;
    int type = QUAD;

    bool isOrigin;
    bool inBrdf = false; 

};
struct BDPTVertexStack
{
    BDPTVertex v[STACKSIZE];
    int size;
};
struct RAWVertex
{
    BDPTVertex v;
    int lastZoneId;
    int valid = 0;
};
struct ZoneSampler
{
    BDPTVertex v[MAX_LIGHT_VERTEX_PER_TRIANGLE];
    int size;
    int realSize;
    float area;
    float Q;
    float sum;
    float m[MAX_LIGHT_VERTEX_PER_TRIANGLE];
    float r[MAX_LIGHT_VERTEX_PER_TRIANGLE];
    void lum_mr_set()
    {
        int valid_size = min(realSize, MAX_LIGHT_VERTEX_PER_TRIANGLE);
        m[0] = 0.0f;
        for (int i = 0; i < valid_size; i++)
        {
            float3 energy = v[i].flux / v[i].pdf;
            float lum = energy.x + energy.y + energy.z; 
            r[i] = lum;
            //r[i] = energy 1000000;
#ifdef ZGC_SAMPLE_ON_LUM

            v[i].zs_weight = r[i];
#endif // SAMPLE_ON_LUMINANCE

            if (i > 0)
            {
                m[i] = m[i - 1];
            }
            
            m[i] += r[i];
            
            //printf("%f %d\n", lum, i);
        }
        sum = m[valid_size - 1];
        //printf("sum %f\n",sum);
        //printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n");
    }
    
};
struct ZoneMatrix
{
    //acc
    float m[MAX_ZONE];
    
    //original
    float r[MAX_ZONE];
    float sum;

    float area;
    void init()
    {
        for (int i = 0; i < MAX_ZONE; i++)
        {
            r[i] = 0.0;
        }
        validation();
    }
    void raw_add(ZoneMatrix &a,float k = 1.0)
    {
        for (int i = 0; i < MAX_ZONE; i++)
        {
            
            r[i] += a.r[i] * k;
        }
        validation();
    }
    void blur(int k)
    {
        float* p = new float[MAX_ZONE];

        for (int i = 0; i < MAX_ZONE; i++)
        {
            p[i] = 0.0;
            int count = 0;
            for (int j = i - k / 2; j <= i + k / 2; j++)
            {
                if (j < 0 || j >= MAX_ZONE)
                    continue;
                p[i] += r[j];
                count++;
            }
            p[i] /= count;
        }
        for (int i = 0; i < MAX_ZONE; i++)
        {
            r[i] = p[i];
        }
        delete[] p;
        validation();
    } 
    void validation()
    {
        for (int i = 0; i < MAX_ZONE; i++)
        {
            if (i == 0)
            {
                m[i] = r[i];
            }
            else
            {
                m[i] = r[i] + m[i - 1];
            }
        }
        sum = m[MAX_ZONE - 1];
    }
};
struct eyeResultRecord
{
    float3 result;
    int eyeZone;
    int lightZone;
    bool valid;
};
struct PMFCache
{
    optix::float3 position;
    optix::float3 in_direction;
    optix::float3 normal;
    float m[LIGHT_VERTEX_NUM/4];
    float r[LIGHT_VERTEX_NUM/4];
    float sum;
    float Q;
    int axis;
    int zoneId;
    bool valid;
};
struct KDPos
{
    optix::float3 position;
    optix::float3 in_direction;
    optix::float3 normal;
    float Q;
    int axis;
    bool valid;

    int original_p;
};
struct triangleStruct
{
    optix::float3 position[3];
    optix::float2 uv[3];
    int objectId;
    int zoneNum;
    bool normalFlip;
    triangleStruct():normalFlip(false){}
    inline float area()
    {
        return length(cross(position[0] - position[1], position[0] - position[2])) / 2;
    }
    optix::float3 getPos(float u,float v)
    {
        float3 u1 = position[1] - position[0];
        float3 v1 = position[2] - position[0];
        return position[0] + u * u1 + v * v1;
    }

    optix::float3 center()
    {
        return (position[0] + position[1] + position[2]) / 3;
    }
    optix::float2 uvCenter()
    {
        return (uv[0] + uv[1] + uv[2]) / 3;
    }
    optix::float3 normal()
    {
        float3 u1 = position[1] - position[0];
        float3 v1 = position[2] - position[0];
        return normalize(cross(u1, v1));
    }
};
struct rawTriangle
{
    optix::float3 position[3];

    optix::float3 normal;
    optix::float3 center;
    float area;
    
    float m;
    
    float3 n_max;
    float3 n_min;

    int objectId;
    int zoneNum;

};
struct LightTraceCache
{
    float3 result;
    uint2 pixiv_loc;
    bool valid;
    bool origin;
};
__device__ int FindDivTriNum(int divLevel, float2 beta_gamma)
{
    int div_block = 0;
    int div_class = divLevel;
    float beta = beta_gamma.x;
    float gamma = beta_gamma.y;
    while (div_class != 0)
    {
        div_block = div_block << 1;
        div_class -= 1;
        float n_gamma = abs((beta + gamma - 1));
        float n_beta = abs((beta - gamma));
        if (beta < gamma)
        {
            div_block += 1;
            gamma = n_beta;
            beta = n_gamma;
        }
        else
        {
            beta = n_beta;
            gamma = n_gamma;

        }
    }
    return div_block;
}

struct SubLightVertex
{
    optix::float3 contri;
    optix::float3 dir;
    float distance;
    float pdf;
    float origin_pdf;
    float dLast;
    float lastNormalProjection;
    int zoneId;

    int lastZoneId;
    float pdf_00;
    int depth;
};
struct UberLightVertex
{

    optix::float3 position;
    //should be normalized
    optix::float3 normal;
     
    optix::float3 color;   
    float singlePdf; 
       

    int materialId;

    int zoneId;  
    bool inBrdf = false;

    int size;
    SubLightVertex son[UberWidth];
};
#define UBER_VERTEX_PER_ZONE  (UBER_VERTEX_NUM / MAX_ZONE + 1)
struct UberZoneLVC
{
    UberLightVertex v[UBER_VERTEX_PER_ZONE];
    int realSize = 0;
    int maxSize = UBER_VERTEX_PER_ZONE;
};
struct TestSetting
{
    int vpZone;
}; 
#endif