#ifndef BDENV_COMMON
#define BDENV_COMMON
#include"rt_function.h"
#include <optixu/optixu_math_namespace.h>
using namespace optix;
struct envInfoBase
{
    int width;
    int height;
	float r;
	float3 center;

	int size; 
	RT_FUNCTION int coord2index(int2 coord)
	{
		return coord.x + coord.y * width;
	}
	RT_FUNCTION int2 index2coord(int index)
	{
		int w = index % width;
		int h = index / width;
		return make_int2(w, h);
	}
	RT_FUNCTION float2 coord2uv(int2 coord)
	{
		float u, v;
		u = coord.x / float(width);
		v = coord.y / float(height);
		return make_float2(u, v);
	}
	RT_FUNCTION int2 uv2coord(float2 uv)
	{
		int x = uv.x * width;
		int y = uv.y * height;
		x = min(x, width - 1);
		y = min(y, height - 1);
		return make_int2(x, y);
	}

	RT_FUNCTION float3 uv2dir(float2 uv)
	{
		float3 dir;

		float theta, phi, x, y, z;
		float u = uv.x;
		float v = uv.y;

		phi = asinf(2 * v - 1.0);
		theta = u / (0.5 * M_1_PIf) - M_PIf;

		dir.y = cos(M_PIf * 0.5f - phi);
		dir.x = cos(phi) * sin(theta);
		dir.z = cos(phi) * cos(theta);
		return dir;
	}
	RT_FUNCTION float2 dir2uv(float3 dir)
	{
		float2 uv;
		float theta = atan2f(dir.x, dir.z);
		float phi = M_PIf * 0.5f - acosf(dir.y);
		float u = (theta + M_PIf) * (0.5f * M_1_PIf);
		float v = 0.5f * (1.0f + sin(phi));
		uv = make_float2(u, v);

		return uv;
	}

};

#endif