#pragma once 

#ifndef BDPT_STRUCT_H
#define BDPT_STRUCT_H
#include"global_setting.h"
#include <optixu/optixu_math_namespace.h>
#include<vector>
#include "material_parameters.h"
#include "light_parameters.h"
#include"rt_function.h"
#include"BDPTVertex.h"
using namespace optix;
enum splitChoice
{
    LongestDim, HighestVariance, RoundRobin
};
enum rayGenChoice
{
    pinholeCamera, FormatTransform,lightBufferGen, PMFCacheLaunch,EVCLaunch,LTCLaunchProg,PGTrainingProg,SlicProg, OPTPProg,
    MLPPathConstructProg,
    rayGenProNum, GammaComputeProg,
    forwardProg, backwardProg,
   
   visibilityTestProg, uberVertexProcessProg
};
 
struct BDPTVertexStack;

struct BDPTVertexStack
{
    BDPTVertex v[STACKSIZE];
    int size;
    RT_FUNCTION void clear()
    {
        size = 0;
        for (int i = 0; i < STACKSIZE; i++)
        {
            v[i].inBrdf = false;
        }
    }
    RT_FUNCTION BDPTVertexStack()
    {

    }
    RT_FUNCTION BDPTVertex& back(int i = 0)
    {
        return v[(size - 1 - i) % STACKSIZE];
    }
    RT_FUNCTION BDPTVertexStack(float3 origin_position,float3 origin_dir)//eye initial
    {
        size = 1;
        v[0].position = origin_position;
        v[0].flux = make_float3(1.0f);
        v[0].pdf = 1.0f;
        v[0].RMIS_pointer = 0;
        v[0].normal = origin_dir;
        v[0].isOrigin = true;
        v[0].depth = 0;

        v[1].singlePdf = 1.0f;

    }
};
struct ZoneSampler
{
    BDPTVertex v[MAX_LIGHT_VERTEX_PER_TRIANGLE];
    int size;
    int realSize;
    float area;
    double Q;
    double Q_old;
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
    float m[SUBSPACE_NUM];
    
    //original
    float r[SUBSPACE_NUM];
    float sum;

    float area;
    void init()
    {
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            r[i] = 0.0;
        }
        validation();
    }
    void raw_add(ZoneMatrix &a,float k = 1.0)
    {
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            
            r[i] += a.r[i] * k;
        }
        validation();
    }
    void blur(int k)
    {
        float* p = new float[SUBSPACE_NUM];

        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            p[i] = 0.0;
            int count = 0;
            for (int j = i - k / 2; j <= i + k / 2; j++)
            {
                if (j < 0 || j >= SUBSPACE_NUM)
                    continue;
                p[i] += r[j];
                count++;
            }
            p[i] /= count;
        }
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            r[i] = p[i];
        }
        delete[] p;
        validation();
    } 
    void validation()
    {
        for (int i = 0; i < SUBSPACE_NUM; i++)
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
        sum = m[SUBSPACE_NUM - 1];
        return;
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            if (r[i] / sum > 0.5)
            {
                float tar = sum - r[i];
                float delta = tar - r[i];
                sum += delta;
                r[i] += delta;
            }
        }

        for (int i = 0; i < SUBSPACE_NUM; i++)
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
        sum = m[SUBSPACE_NUM - 1];


    }
};
struct eyeResultRecord
{
    float3 result;
    int eyeZone;
    int lightZone;
    bool valid;
};
#define PMFSIZE 800
#ifdef PCBPT
#define PMFSIZE (LIGHT_VERTEX_NUM / 3)
#endif 

//#define PMFSIZE 16000
struct PMFCache
{
    optix::float3 position;
    optix::float3 in_direction;
    optix::float3 normal;
#ifdef KITCHEN
    float m[PMFSIZE];
    float r[PMFSIZE];
    int   re_direct[1];
    bool  hit_success[PMFSIZE];
    float stage_1_pdf[1];
#else
    float m[PMFSIZE];
    float r[PMFSIZE];
    int   re_direct[1];
    float stage_1_pdf[1];
    bool  hit_success[PMFSIZE];
#endif
    float sum;
    float sum_refine;
    float Q;
    float variance;
    float Q_variance;
    float sum_fix;
    int axis;
    int zoneId;
    int shadow_success;
    int size;
    bool valid;
    bool is_virtual;
    __host__ RT_FUNCTION
        PMFCache()
    {

    }
    __host__ RT_FUNCTION
        PMFCache(int s, bool is_virtual = true):size(s),is_virtual(is_virtual)
    {
        int min_range = min(size, LIGHT_VERTEX_NUM / 4);
        for (int i = 0; i < min_range; i++)
        {
            r[i] = 1.0 / size;
            m[i] = 1.0 / size * (i + 1);
        }
        sum = 1;
    }
    RT_FUNCTION BDPTVertex sample(BDPTVertex* lvc, float random, float& pmf, int& index2)
    {
        if (is_virtual)
        {

            index2 = optix::clamp(static_cast<int>(floorf(random * size)), 0, size - 1);
            pmf = 1.0 / size;
            return lvc[index2];
        }
        float index = random * sum;
        int mid = size / 2 - 1, l = 0, rr = size;
        while (rr - l > 1)
        {
            if (index < m[mid])
            {
                rr = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + rr) / 2 - 1;
        }
        pmf = r[l] / sum;// *stage_1_pdf[l];
        index2 = l;// re_direct[l];
        return lvc[index2];
    }
    void fix_init()
    {
        sum_fix = 0;
        for (int i = 0; i < size; i++)
        {
            m[i] = 0;
        }
    }
    void fix_validation()
    {
        float fix_rate = FIX_SAMPLE_RATE;
        fix_rate = (fix_rate) / (1 - fix_rate);
        for (int i = 0; i < size; i++)
        {
            if (!(sum_fix < 0.5))
            {
                r[i] += (m[i] / sum_fix) * fix_rate * sum;                
            }
            if (i == 0)
            {
                m[i] = r[i];
            }
            else
            {
                m[i] = r[i] + m[i - 1];
            }
        }
        sum = m[size - 1];
    }
    void validation()
    {
        for (int i = 0; i < size; i++)
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
        sum = m[size - 1];
    }
    void merge_with_other(PMFCache& b)
    {
        float min_clamp = 0.01;
        if (b.shadow_success == 0)
        {
            return;
        }
        for (int i = 0; i < size; i++)
        {
            if (hit_success[i] == false && b.hit_success[i] == true)
            {
                m[i] +=  b.r[i] / b.sum;
                sum_fix += m[i];
            }
        }
    }

};
struct KDPos
{
    optix::float3 position;
    optix::float3 in_direction;
    optix::float3 normal;
    float Q;
    float sum;
    float Q_variance; 
    float shadow_success;
    int axis;
    int label;
    int original_p;
    bool valid;
    KDPos() {}
    KDPos(float3 pos, int label) :position(pos), label(label) {}
};
struct triangleStruct
{
    optix::float3 position[3];
    optix::float2 uv[3];
    int objectId;
    int zoneNum;
    bool normalFlip;
    RT_FUNCTION triangleStruct():normalFlip(false){}
    RT_FUNCTION float area()
    {
        return length(cross(position[0] - position[1], position[0] - position[2])) / 2;
    }
    RT_FUNCTION optix::float3 getPos(float u,float v)
    {
        float3 u1 = position[1] - position[0];
        float3 v1 = position[2] - position[0];
        return position[0] + u * u1 + v * v1;
    }

    RT_FUNCTION optix::float3 center()
    {
        return (position[0] + position[1] + position[2]) / 3;
    }
    RT_FUNCTION optix::float2 uvCenter()
    {
        return (uv[0] + uv[1] + uv[2]) / 3;
    }
    RT_FUNCTION optix::float3 normal()
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
RT_FUNCTION int FindDivTriNum(int divLevel, float2 beta_gamma)
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
#define UBER_VERTEX_PER_ZONE  (UBER_VERTEX_NUM / SUBSPACE_NUM + 1)
struct UberZoneLVC
{
    UberLightVertex v[UBER_VERTEX_PER_ZONE];
    int realSize = 0;
    int maxSize = UBER_VERTEX_PER_ZONE;
};
struct TestSetting
{
    int vpZone;
    int vps_node;
    BDPTVertex v;
}; 
#endif