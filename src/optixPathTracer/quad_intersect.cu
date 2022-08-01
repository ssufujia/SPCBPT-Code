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

#include <optix_world.h>
#include "intersection_refinement.h"

using namespace optix;

rtDeclareVariable(float4, plane, , );
rtDeclareVariable(float3, v1, , );
rtDeclareVariable(float3, v2, , );
rtDeclareVariable(float3, anchor, , );

rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float4, geometry_color, attribute geometry_color, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect(int primIdx)
{
	float3 n = make_float3(plane);
	float dt = dot(ray.direction, n);
	float t = (plane.w - dot(n, ray.origin)) / dt;
	if (t > ray.tmin && t < ray.tmax) {
		float3 p = ray.origin + ray.direction * t;
		float3 vi = p - anchor;
		float a1 = dot(v1, vi);
		if (a1 >= 0 && a1 <= 1) {
			float a2 = dot(v2, vi);
			if (a2 >= 0 && a2 <= 1) {
				if (rtPotentialIntersection(t)) {
					shading_normal = geometric_normal = n;
					texcoord = make_float3(a1, a2, 0);
					geometry_color = make_float4(1.0f);

					refine_and_offset_hitpoint(ray.origin + t * ray.direction, ray.direction,
						n, anchor,
						back_hit_point, front_hit_point);

					rtReportIntersection(0);
				}
			}
		}
	}
}

RT_PROGRAM void intersect2(int primIdx)
{
	float3 n = make_float3(plane);	
	float3 tv1 = v1 / dot(v1, v1);
	float3 tv2 = v2 / dot(v2, v2);
	float3 u = tv1,v = tv2,w = -ray.direction;
	float l2w_array[16] = {
		u.x,v.x,w.x,0,
		u.y,v.y,w.y,0,
		u.z,v.z,w.z,0,
		0  ,0  ,0  ,1};

	Matrix<4,4> L2W(l2w_array);
	Matrix<4,4> W2L = L2W.inverse();
	

	float4 W = make_float4(ray.origin - anchor,1); 
	float4 L = W2L * W;
	float t = L.z;
	if(L.x<1&&L.y<1&&L.x>0&&L.y>0)
	{
		if (rtPotentialIntersection(t)) {
			shading_normal = geometric_normal = n;
			texcoord = make_float3(L.x, L.y, 0);
			geometry_color = make_float4(1.0f);

			refine_and_offset_hitpoint(ray.origin + t * ray.direction, ray.direction,
				n, anchor,
				back_hit_point, front_hit_point);

			rtReportIntersection(0);
		}
	}
	
	
}

rtDeclareVariable(float3, r_v, , );
rtDeclareVariable(float3, r_u, , );
RT_PROGRAM void bounds(int, float result[6])
{
	// v1 and v2 are scaled by 1./length^2.  Rescale back to normal for the bounds computation.
	const float3 tv1 = v1 / dot(v1, v1);
	const float3 tv2 = v2 / dot(v2, v2);
	const float3 p00 = anchor;
	const float3 p01 = anchor + tv1;
	const float3 p10 = anchor + tv2;
	const float3 p11 = anchor + tv1 + tv2;
	const float  area = length(cross(tv1, tv2));

	optix::Aabb* aabb = (optix::Aabb*)result;

	if (area > 0.0f && !isinf(area)) {
		aabb->m_min = fminf(fminf(p00, p01), fminf(p10, p11)) - make_float3(1000.0f);
		aabb->m_max = fmaxf(fmaxf(p00, p01), fmaxf(p10, p11)) + make_float3(1000.0f);
	}
	else {
		aabb->invalidate();
	}
}