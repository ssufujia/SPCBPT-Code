#ifndef SVM_COMMON
#define SVM_COMMON
#include"rt_function.h"
#include <optixu/optixu_math_namespace.h>
using namespace optix;
#define GAMMA_DIM 100
struct GammaVec
{
    float2 surface;
    float2 dir;
    int objId;
    int default_lable;
    float contri;
    float gamma[GAMMA_DIM];
    float pdf[GAMMA_DIM];
    int begin_label[GAMMA_DIM];
    int end_label[GAMMA_DIM];
}; 
#define OPT_PATH_LENGTH 10

struct OptimizationPathInfo
{
    float contri;
    float actual_pdf;
    int path_length;
    float pixiv_lum;
    int ss_ids[OPT_PATH_LENGTH];
    float pdfs[OPT_PATH_LENGTH];
    float light_pdfs[OPT_PATH_LENGTH];
    float light_contris[OPT_PATH_LENGTH];
    float3 positions[OPT_PATH_LENGTH]; 
    float2 uvs[OPT_PATH_LENGTH]; 
    bool valid;
};

#endif