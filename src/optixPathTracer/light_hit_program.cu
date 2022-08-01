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
#include "BDPT.h"
#include "random.h"
#include "rt_function.h"
#include "light_parameters.h"
#include "state.h"
#include "rmis.h"
using namespace optix;


rtDeclareVariable( float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable( float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable( float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, back_hit_point, attribute back_hit_point, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, hit_dist, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float, scene_epsilon, , );
rtBuffer<LightParameter> sysLightParameters;
rtDeclareVariable(int, lightMaterialId, , ); 
rtDeclareVariable(int,           KD_SET, , ) = { 0 };

rtBuffer<UberZoneLVC,1>          uberLVC; 
rtDeclareVariable(float, scene_area, , ) = {1.0};
rtBuffer<KDPos,1>        Kd_position; 
RT_FUNCTION int find_closest_pmfCache(float3 position)
{
  int closest_index = 0;
  float closest_dis2 = dot(Kd_position[0].position - position, Kd_position[0].position - position);
  unsigned int stack[25];
  float dis_stack[25];
  unsigned int stack_current = 0;
  unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

  float block_min = 0.0;
  dis_stack[stack_current] = 0.0;
  push_node( 0 );

  do {
    if(closest_dis2 < block_min)
    {
      node = pop_node();
      block_min = dis_stack[stack_current];
      continue;
    }
    KDPos& currentVDirector = Kd_position[node];
    uint axis = currentVDirector.axis;
    if( !( axis & PPM_NULL ) ) {

      float3 vd_position = currentVDirector.position;
      float3 diff = position - vd_position;
      float distance2 = dot(diff, diff);

      if (distance2 < closest_dis2) {
        closest_dis2 = distance2;
        closest_index = node;
      
      }

      // Recurse
      if( !( axis & PPM_LEAF ) ) {
        float d;
        if      ( axis & PPM_X ) d = diff.x;
        else if ( axis & PPM_Y ) d = diff.y;
        else                      d = diff.z;

        // Calculate the next child selector. 0 is left, 1 is right.
        int selector = d < 0.0f ? 0 : 1;
        if( d*d < closest_dis2 ) {
          dis_stack[stack_current] = d*d;
          push_node( (node<<1) + 2 - selector );
        }

        node = (node<<1) + 1 + selector;
      } else {
        node = pop_node();
        block_min = dis_stack[stack_current];
      }
    } else {
      node = pop_node();
      block_min = dis_stack[stack_current];
    }
  } while ( node );
  return closest_index;
}
RT_PROGRAM void closest_hit()
{
	const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	const float3 world_geometric_normal = normalize( rtTransformNormal( RT_OBJECT_TO_WORLD, geometric_normal ) );
	const float3 ffnormal = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

	LightParameter light = sysLightParameters[lightMaterialId];
	float cosTheta = dot(-ray.direction, light.normal);

	if ((light.lightType == QUAD && cosTheta > 0.0f) || light.lightType == SPHERE)
	{
		if(prd.depth == 0 || prd.specularBounce)
			prd.radiance += light.emission * prd.throughput;
		else
		{
			float lightPdf = (hit_dist * hit_dist) / (light.area * clamp(cosTheta, 1.e-3f, 1.0f));
			prd.radiance +=  powerHeuristic(prd.pdf, lightPdf) * prd.throughput * light.emission;
			//prd.radiance +=   prd.throughput * light.emission;
		}
	}

	prd.done = true;
}
RT_PROGRAM void BDPT_closest_hit()
{
	prd.done = true;
	LightParameter light = sysLightParameters[lightMaterialId];
	BDPTVertex& MidVertex = prd.stackP->v[(prd.stackP->size) % STACKSIZE];
	BDPTVertex& LastVertex = prd.stackP->v[(prd.stackP->size - 1) % STACKSIZE];
	MidVertex.position = ray.origin + t_hit * ray.direction;
	MidVertex.normal = light.normal;
	if (dot(MidVertex.normal, ray.direction) > 0.0f)
		return;
	MidVertex.type = HIT_LIGHT_SOURCE;

	MidVertex.pg_lightPdf = make_float2(quadLightPdf_area(lightMaterialId, MidVertex.uv, -ray.direction)
		, quadLightPdf_dir(lightMaterialId, MidVertex.uv, -ray.direction));

	float lightPdf = 1.0 / sysNumberOfLights * MidVertex.pg_lightPdf.x;


	float pdf_G = abs(dot(MidVertex.normal, ray.direction) * dot(LastVertex.normal, ray.direction)) / (t_hit * t_hit);
	if (LastVertex.isOrigin)
	{
		MidVertex.flux = LastVertex.flux * pdf_G * light.emission;
	}
	else
	{
		MidVertex.flux = MidVertex.flux * LastVertex.flux * pdf_G * light.emission;
	}



	MidVertex.lastPosition = LastVertex.position;
	MidVertex.lastNormalProjection = abs(dot(LastVertex.normal, ray.direction));

	//MidVertex.zoneId = SUBSPACE_NUM - lightMaterialId - 1;
	MidVertex.zoneId = get_light_zone(light, make_float2(texcoord));
	MidVertex.uv = make_float2(texcoord);
	//MidVertex.zoneId = -1;
	MidVertex.lastZoneId = LastVertex.zoneId;


	MidVertex.singlePdf = MidVertex.singlePdf * pdf_G / abs(dot(LastVertex.normal, ray.direction));
	MidVertex.pdf = LastVertex.pdf * MidVertex.singlePdf;

	//MidVertex.dLast = LastVertex.d;
	MidVertex.materialId = lightMaterialId;

	MidVertex.depth = LastVertex.depth + 1;

	 

	if (prd.stackP->size == 1)
	{
		MidVertex.RMIS_pointer = 1.0;

		prd.stackP->size++;
		return;
	}

	BDPTVertex virtual_light;
	virtual_light.position = MidVertex.position;
	virtual_light.RMIS_pointer = 1;
	virtual_light.normal = MidVertex.normal;
	virtual_light.pdf = lightPdf;
	virtual_light.singlePdf = lightPdf;
	virtual_light.flux = light.emission;
	virtual_light.zoneId = MidVertex.zoneId;
	virtual_light.isBrdf = false;
	//rtPrintf("%f %f\n", 1.0 / MidVertex.d, light_hit(LastVertex, virtual_light));
#ifdef  ZGCBPT 
	MidVertex.RMIS_pointer = 1.0 / light_hit(LastVertex, virtual_light);
#endif //  ZGCBPT


	prd.stackP->size++;


}

RT_PROGRAM void BDPT_L_closest_hit()
{
	prd.done = true;

}