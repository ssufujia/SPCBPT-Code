#ifndef GAMMA_COMMON
#define GAMMA_COMMON


#include"rt_function.h"
#include"global_setting.h"
#include <optixu/optixu_math_namespace.h>
using namespace optix;
struct lightSelectionFunction
{
    float Matrix[SUBSPACE_NUM][SUBSPACE_NUM];
    float CMFs[SUBSPACE_NUM][SUBSPACE_NUM];
    float Q[SUBSPACE_NUM];


};
#endif