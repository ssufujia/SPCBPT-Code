#ifndef __BDPTVERTEX_H
#define __BDPTVERTEX_H


#include <optixu/optixu_math_namespace.h>
#include "material_parameters.h"
#include "light_parameters.h"
#include"rt_function.h"
#include"global_setting.h"


struct BDPTVertex
{
    optix::float3 position;
    //should be normalized
    optix::float3 normal;

    //flux is actually the local contribution term
    optix::float3 flux;
    optix::float3 color;

    //lastPosition: the position of the previous vertex, used to compute RMIS weight
    optix::float3 lastPosition;
    
    //uv: only used for path guiding, to save the uv coordinate of the light source
    optix::float2 uv;
    //pg_lightPdf is only used for path guiding, too. 
    optix::float2 pg_lightPdf;


    //used for RMIS computing, dot(incident_direction, normal of the previous vertex)
    float lastNormalProjection;

    //pdf for the sub-path corresponding to THIS vertex to be sampled.
    float pdf;
    //used for RMIS computing, pdf for (previous vertex) -> (this vertex)
    float singlePdf;
    //used for RMIS computing, the single pdf of the previous vertex.
    float lastSinglePdf;

    //float d;     //can be replaced by RIS_pointer, consider to remove


    int materialId;

    int zoneId;//subspace ID, consider to rename
    int depth;

    //used for RMIS computing
    int lastZoneId;
    int type = QUAD;


    bool isOrigin;
    bool inBrdf = false;
    bool lastBrdf = false;
    bool isBrdf = false;

    bool isLastVertex_direction;//if this vertex comes from the directional light 


    //cache the RMIS weight
    float3 RMIS_pointer_3;
    float RMIS_pointer;
    float last_lum;
    __host__ __device__ BDPTVertex() :isBrdf(false), lastBrdf(false) {}
    __host__ __device__ bool is_LL_DIRECTION() { return isLastVertex_direction; }
    __host__ __device__ bool is_DIRECTION()const { return type == DIRECTION||type == ENV; }
    __host__ __device__ float contri_float() { return flux.x + flux.y + flux.z; }
};

struct RAWVertex
{
    BDPTVertex v;
    int lastZoneId;
    int valid = 0;
};
#endif