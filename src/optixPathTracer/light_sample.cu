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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu_matrix_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "rt_function.h"
#include "material_parameters.h"
#include "light_parameters.h"
#include "state.h"
#include "BDenv_device.h"
using namespace optix;
 
RT_FUNCTION float3 UniformSampleSphere(float u1, float u2)
{
	float z = 1.f - 2.f * u1;
	float r = sqrtf(max(0.f, 1.f - z * z));
	float phi = 2.f * M_PIf * u2;
	float x = r * cosf(phi);
	float y = r * sinf(phi);

	return make_float3(x, y, z);
}

RT_CALLABLE_PROGRAM void sphere_sample(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)
{
	const float r1 = rnd(prd.seed);
	const float r2 = rnd(prd.seed);
	sample.surfacePos = light.position + UniformSampleSphere(r1, r2) * light.radius;
	sample.normal = normalize(sample.surfacePos - light.position);
	sample.emission = light.emission ;//* sysNumberOfLights;
	sample.pdf = 1.0f / light.area;
	sample.zoneId = light.divBase;
}

RT_CALLABLE_PROGRAM void quad_sample(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)
{
	float r1 = rnd(prd.seed);
	float r2 = rnd(prd.seed);

	if (pg_lightSource_enable && rnd(prd.seed) < PG_LIGHTSOURCE_RATE)
	{
		float2 uv = pg_api.L_area_sample(prd.seed, light.id);
		r1 = uv.x;
		r2 = uv.y;
	}

	sample.uv = make_float2(r1, r2);

	if(pg_lightSource_enable && rnd(prd.seed) < PG_LIGHTSOURCE_RATE)
	{
		sample.dir = pg_api.L_dir_sample(prd.seed, light.id, sample.uv);
	}
	else 
	{
		float r1 = rnd(prd.seed);
		float r2 = rnd(prd.seed);
		optix::Onb onb(light.normal);

		cosine_sample_hemisphere(r1, r2, sample.dir);
		onb.inverse_transform(sample.dir);
	}

	sample.surfacePos = light.position + light.u * r1 + light.v * r2;
	sample.normal = light.normal;
	sample.emission = light.emission ;//* sysNumberOfLights;
	sample.emission *= dot(sample.dir, sample.normal) < 0 ? 0 : 1;
	sample.pdf = 1.0 / light.area;
	sample.zoneId = get_light_zone(light, make_float2(r1,r2));


	sample.pdf = quadLightPdf_area(light.id, sample.uv, sample.dir);
	sample.pdf_dir = quadLightPdf_dir(light.id, sample.uv, sample.dir);
}

RT_CALLABLE_PROGRAM void direction_sample(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)
{
	const float r1 = rnd(prd.seed);
	const float r2 = rnd(prd.seed);
	float3 dir;
	optix::Onb onb( light.direction );
    cosine_sample_hemisphere(r1, r2, dir);
	
	sample.surfacePos = sceneMaxLength * (-light.direction) + dir.x * sceneMaxLength / 2 * onb.m_tangent+ dir.y * sceneMaxLength / 2 * onb.m_binormal + (min_box + min_box)/2;
	
	if (true)
	{
		float3 unTouchPoint = 2 * (sceneMaxLength + length(max_box - min_box)) * (-light.direction);
		float3 projectDelta = project_box_max - project_box_min;
		float2 projectPoint = make_float2(projectDelta.x * r1, projectDelta.y * r2) + make_float2(project_box_min);
		float3 projectPoint_onb = projectPoint.x * onb.m_binormal + projectPoint.y * onb.m_tangent;
		sample.surfacePos = unTouchPoint + projectPoint_onb;
	}
	sample.normal = light.direction;
	//sample.pdf = 4.0 / (M_PIf * sceneMaxLength * sceneMaxLength);
	sample.pdf = 1.0;
	sample.emission = light.emission  ;//* sysNumberOfLights;
	sample.zoneId = light.divBase;
	sample.dir = light.direction; 
	sample.pdf_dir = 1.0 / DirProjectionArea;
}

RT_CALLABLE_PROGRAM void env_sample(LightParameter &light, PerRayData_radiance &prd, LightSample &sample)
{
	const float r1 = rnd(prd.seed);
	const float r2 = rnd(prd.seed);
	//float3 dir = UniformSampleSphere(r1, r2);
	float3 sample_dir = sky.sample(prd.seed);
	float3 dir = -sample_dir;

	float3 color = sky.getColor(sample_dir);

	
	
	sample.surfacePos = sky.sample_projectPos(dir,prd.seed);
	sample.normal = dir;
	sample.pdf = sky.pdf(sample_dir);
	sample.emission = color  ;//* sysNumberOfLights;
	sample.zoneId = sky.getLabel(sample_dir);
	sample.dir = dir;
	sample.pdf_dir = sky.projectPdf();

}

