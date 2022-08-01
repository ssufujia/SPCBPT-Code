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

#pragma once

#include <optixu/optixu_vector_types.h>
#include"BDPT.h"
struct PerRayData_radiance
{
  int depth;
  unsigned int seed;

  // shading state
  bool done;
  bool inShadow;
  bool specularBounce;
  float3 radiance;
  float3 origin;
  float3 direction;
  float3 throughput;
  float pdf;
  BDPTVertexStack *stackP;

  RT_FUNCTION PerRayData_radiance() {}
  RT_FUNCTION PerRayData_radiance(unsigned int seed, BDPTVertexStack * stackP):
    seed(seed),stackP(stackP)//eye only
  {

      depth = 0; 
      done = false;
      pdf = 0.0f;
      specularBounce = false;
       
      // These represent the current shading state and will be set by the closest-hit or miss program

      // attenuation (<= 1) from surface interaction.
      throughput = make_float3(1.0f);

      // light from a light source or miss program
      radiance = make_float3(0.0f);

      // next ray to be traced
      origin = make_float3(0.0f);
      direction = make_float3(0.0f);
      origin = stackP->v[0].position;
      direction = stackP->v[0].normal;
  }

};

struct PerRayData_shadow
{
  bool inShadow;
};


