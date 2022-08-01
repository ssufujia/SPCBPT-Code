#pragma once

#ifndef LIGHT_PARAMETER_H
#define LIGHT_PARAMETER_H

#include"rt_function.h"
#include <optixu/optixu_math_namespace.h>
using namespace optix;
enum LightType
{
	SPHERE, QUAD, DIRECTION,ENV,HIT_LIGHT_SOURCE,ENV_MISS,LightTypeNum
};
enum RayType
{
    PTRay,ShadowRay,BDPTRay, BDPT_L_Ray, PT_RR_RAY, RayTypeCount
};
struct LightParameter
{
	optix::float3 position;
	optix::float3 normal;
	optix::float3 emission;
	optix::float3 u;
	optix::float3 v;
    optix::float3 direction;
	LightType lightType;
	float area;
	float radius;
	int divBase;
	int divLevel;
	int id;
};

struct LightSample
{
	optix::float3 surfacePos;
	optix::float3 normal;
	optix::float3 emission;
	optix::float3 dir;
	optix::float2 uv;
	float pdf;
	float pdf_dir;
	int zoneId;
};

#endif
