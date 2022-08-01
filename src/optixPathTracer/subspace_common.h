#ifndef SUBSPACE_COMMON
#define SUBSPACE_COMMON
#include"rt_function.h"
#include <optixu/optixu_math_namespace.h>
using namespace optix;

struct jumpUnit
{
    int p; 
    float cmf;
    float pmf;
};
struct subspaceSampler
{
    int size;
    int base;
    float Q;
    float energy_sum;
    int vertex_sum;
};



#endif