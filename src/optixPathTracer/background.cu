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
#include "prd.h"
#include "BDenv_device.h"
//#include"rmis.h" 

 
using namespace optix;

rtDeclareVariable(float3, background_light, , ); // horizon color
rtDeclareVariable(float3, background_dark, , );  // zenith color
rtDeclareVariable(float3, up, , );               // global up vector

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, ); 

rtDeclareVariable(int, KD_SET, , ) = { 0 };

// -----------------------------------------------------------------------------

/*RT_PROGRAM void miss()
{
  const float t = max(dot(ray.direction, up), 0.0f);
  const float3 result = lerp(background_light, background_dark, t);

  prd_radiance.radiance = result;
  prd_radiance.done = true;
}*/

RT_PROGRAM void miss()
{
/*	float theta = atan2f(ray.direction.x, ray.direction.y);
	float phi = M_PIf * 0.5f - acosf(ray.direction.z);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	float3 result = make_float3(tex2D(envmap, u, v));
*/
	//if (prd_radiance.depth == 0)
		//prd_radiance.radiance = make_float3(0.0f);
	//else
	//prd.radiance += make_float3(0.5f,0.7,0.8f) * prd.throughput;
	//prd.radiance += make_float3(0.5f, 0.7, 0.8f) * prd.throughput * 0.0;
	prd.done = true;
//	prd.radiance += result * prd.throughput;
}


#ifdef BD_ENV
RT_PROGRAM void miss_env()
{
	prd.done = true;
	//return;
	BDPTVertex& MidVertex = prd.stackP->v[(prd.stackP->size) % STACKSIZE];
	BDPTVertex& LastVertex = prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];
	MidVertex.normal = -ray.direction;

	MidVertex.type = ENV_MISS;
	float lightPdf = 1.0 / sysNumberOfLights * sky.pdf(ray.direction);

	float pdf_G = abs(dot(MidVertex.normal, ray.direction) * dot(LastVertex.normal, ray.direction));
	if (LastVertex.isOrigin)
	{
		MidVertex.flux = LastVertex.flux * pdf_G * sky.color(ray.direction);
	}
	else
	{
		MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G * sky.color(ray.direction);
	}



	MidVertex.lastPosition = LastVertex.position;
	MidVertex.lastNormalProjection = abs(dot(LastVertex.normal, ray.direction));

	//MidVertex.zoneId = SUBSPACE_NUM - lightMaterialId - 1;
	MidVertex.zoneId = sky.getLabel(ray.direction);
	//MidVertex.zoneId = -1;
	MidVertex.lastZoneId = LastVertex.zoneId;


	MidVertex.singlePdf = MidVertex.singlePdf;
	MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;

	//MidVertex.dLast = LastVertex.d;

	MidVertex.depth = LastVertex.depth + 1;

	if (MidVertex.depth >= 2)
	{
		float inverLastPdf = pdf_G * sky.projectPdf();
		float t_rate = LastVertex.singlePdf / inverLastPdf;

		float currentWeight = MidVertex.inBrdf ? 0.0f : 1.0f;

#ifdef ZGCBPT
#ifdef ZGC_SAMPLE_ON_LUM 
		float3 lum = sky.color(ray.direction) / lightPdf;
		float3 SOL_rate = connectRate_SOL(LastVertex.zoneId, MidVertex.zoneId, lum);
		currentWeight *= SOL_rate.x + SOL_rate.y + SOL_rate.z;
		//if (MidVertex.depth == 2)
		//	if (isinf(currentWeight))
		//		currentWeight = connectRate(LastVertex.zoneId, MidVertex.zoneId);
//			rtPrintf("%f %f\n", currentWeight, connectRate(LastVertex.zoneId, MidVertex.zoneId));
#else
		currentWeight *= connectRate(LastVertex.zoneId, MidVertex.zoneId);
#endif
#endif 
		 
		MidVertex.RMIS_pointer = LastVertex.RMIS_pointer / t_rate + currentWeight;



#ifdef ZGC_SAMPLE_ON_LUM
		float3 a_lum = lum / sky.projectPdf();
		float3 tmp_dcolor = a_lum;// *LastVertex.Dcolor;
		//MidVertex.RMIS_pointer = (tmp_dcolor.x + tmp_dcolor.y + tmp_dcolor.z + LastVertex.d) / t_rate + currentWeight;
		 // tobe rewrite

		//if (MidVertex.depth == 2)
		//{
		//	MidVertex.d = LastVertex.d / t_rate + currentWeight;
		//}
#endif
	}
	else
	{
		MidVertex.RMIS_pointer = 0;
	}
	float t_rate = MidVertex.singlePdf / lightPdf;
	float currentWeight = 1.0;
	MidVertex.RMIS_pointer = MidVertex.RMIS_pointer / t_rate + currentWeight;
	//MidVertex.dLast = lightPdf;
	 

	prd.stackP->size++;
}

#else
RT_PROGRAM void miss_env()
{
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	float3 result = make_float3(tex2D(envmap, u, v));

	result = sky.getColor(ray.direction);
	//if (prd_radiance.depth == 0)
		//prd_radiance.radiance = make_float3(0.0f);
	//else
	//prd.radiance += make_float3(0.5f,0.7,0.8f) * prd.throughput;
	//prd.radiance += make_float3(0.5f, 0.7, 0.8f) * prd.throughput * 0.0;
	prd.done = true;
	if (prd.depth == 0)
		prd.radiance += result * prd.throughput;
}
#endif

RT_PROGRAM void pt_miss_env()
{
	float theta = atan2f(ray.direction.x, ray.direction.z);
	float phi = M_PIf * 0.5f - acosf(ray.direction.y);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	float3 result = make_float3(tex2D(envmap, u, v));

	result = sky.getColor(ray.direction);
	//if (prd_radiance.depth == 0)
		//prd_radiance.radiance = make_float3(0.0f);
	//else
	//prd.radiance += make_float3(0.5f,0.7,0.8f) * prd.throughput;
	//prd.radiance += make_float3(0.5f, 0.7, 0.8f) * prd.throughput * 0.0;
//	result = 
	prd.done = true;
	//if (prd.depth == 0 || prd.specularBounce == true)
		//prd.radiance += result * prd.throughput;
}

//RT_PROGRAM void BDPT_env_hit()

RT_PROGRAM void bdpt_miss_env()
{
	float theta = atan2f(ray.direction.x, ray.direction.y);
	float phi = M_PIf * 0.5f - acosf(ray.direction.z);
	float u = (theta + M_PIf) * (0.5f * M_1_PIf);
	float v = 0.5f * (1.0f + sin(phi));
	float3 result = make_float3(tex2D(envmap, u, v));

	//if (prd_radiance.depth == 0)
		//prd_radiance.radiance = make_float3(0.0f);
	//else
	//prd.radiance += make_float3(0.5f,0.7,0.8f) * prd.throughput;
	//prd.radiance += make_float3(0.5f, 0.7, 0.8f) * prd.throughput * 0.0;
	BDPTVertex &MidVertex = prd.stackP->v[(prd.stackP->size) % STACKSIZE];
	BDPTVertex &LastVertex = prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];
	MidVertex.zoneId = SUBSPACE_NUM - 1;//error
	if(MidVertex.depth>=2)
	{

		float inverLastPdf =  4.0 / (M_PIf * sceneMaxLength * sceneMaxLength) * abs(dot(LastVertex.normal,ray.direction));
		float t_rate = LastVertex.singlePdf / inverLastPdf;
		
		float currentWeight = MidVertex.inBrdf?0.0f:1.0f;
				
#ifdef ZGCBPT
		currentWeight *= connectRate(LastVertex.zoneId,MidVertex.zoneId);
#endif 

		//MidVertex.d = LastVertex.d / t_rate + currentWeight; 
	}
	else
	{
		MidVertex.RMIS_pointer = 0;
	}
	float lightPdf = 1.0 / (4 * M_PIf *  sysNumberOfLights);
	float t_rate = MidVertex.singlePdf / lightPdf;
	float currentWeight = average_light_length;
	
#ifdef ZGCBPT
currentWeight  = 1;
#endif 
	//MidVertex.d = MidVertex.d / t_rate + currentWeight;


	prd.done = true;
	prd.radiance += result * prd.throughput * currentWeight;// / MidVertex.d;
}
