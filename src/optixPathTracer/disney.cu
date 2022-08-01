/*
 Copyright Disney Enterprises, Inc.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License
 and the following modification to it: Section 6 Trademarks.
 deleted and replaced with:

 6. Trademarks. This License does not grant permission to use the
 trade names, trademarks, service marks, or product names of the
 Licensor and its affiliates, except as required for reproducing
 the content of the NOTICE file.

 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
*/

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu_matrix_namespace.h>
#include "helpers.h"
#include "prd.h"
#include "random.h"
#include "rt_function.h"
#include "material_parameters.h"
#include "state.h"
#include "BDPT.h"
#include "PG_device.h"
using namespace optix;

rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );



/*
	http://simon-kallweit.me/rendercompo2015/
*/
RT_CALLABLE_PROGRAM void Pdf(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	float3 n = state.ffnormal;
	float3 V = -ray.direction;
	float3 L = prd.direction;

	float specularAlpha = max(0.001f, mat.roughness);
	float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);
	
	float diffuseRatio = 0.5f * (1.f - mat.metallic);
	float specularRatio = 1.f - diffuseRatio;

	float3 half = normalize(L+V);

	float cosTheta = abs(dot(half, n));
	float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
	float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

	// calculate diffuse and specular pdfs and mix ratio
	float ratio = 1.0f / (1.0f + mat.clearcoat);
	float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
	float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

	// weight pdfs according to ratios
	prd.pdf =  diffuseRatio * pdfDiff + specularRatio * pdfSpec;

	if (pg_enable&& state.eye_side == true)
	{
		float3 position = state.fhp;
		float pdf2 = pg_api.pdf(position, prd.direction);
		prd.pdf = lerp(prd.pdf, pdf2, PG_RATE);
	}
	else if (pg_lightSide_enable && state.eye_side == false)
	{ 
		float3 position = state.fhp;
		float pdf2 = pg_api.pdf_lightside(position, prd.direction);
		prd.pdf = lerp(prd.pdf, pdf2, PG_RATE);
	}
}

/*
	https://learnopengl.com/PBR/IBL/Specular-IBL
*/

RT_CALLABLE_PROGRAM void Sample(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	if (pg_enable|| pg_lightSide_enable)
	{
		if (rnd(prd.seed) < PG_RATE)
		{
			if (pg_enable && state.eye_side == true)
			{
				prd.origin = state.fhp;
				prd.direction = pg_api.sample(prd.seed, state.fhp);
				return;
			}
			else if (pg_lightSide_enable && state.eye_side == false)
			{
				prd.origin = state.fhp;
				prd.direction = pg_api.sample_lightside(prd.seed, state.fhp);
				return;

			}
		}
	}
	float3 N = state.ffnormal;
	float3 V = -ray.direction;
	prd.origin = state.fhp;

	float3 dir;
	
	float probability = rnd(prd.seed);
	float diffuseRatio = 0.5f * (1.0f - mat.metallic);

	float r1 = rnd(prd.seed);
	float r2 = rnd(prd.seed);

	optix::Onb onb( N ); // basis

	if (probability < diffuseRatio) // sample diffuse
	{
		cosine_sample_hemisphere(r1, r2, dir);
		onb.inverse_transform(dir);
	}
	else
	{
		float a = max(0.001f, mat.roughness);

		float phi = r1 * 2.0f * M_PIf;
        
		float cosTheta = sqrtf((1.0f - r2) / (1.0f + (a*a-1.0f) *r2));      
		float sinTheta = sqrtf(1.0f - (cosTheta * cosTheta));
		float sinPhi = sinf(phi);
		float cosPhi = cosf(phi);

		float3 half = make_float3(sinTheta*cosPhi, sinTheta*sinPhi, cosTheta);
		onb.inverse_transform(half);

		dir = 2.0f*dot(V, half)*half - V; //reflection vector

	}
	prd.direction = dir;
}


RT_CALLABLE_PROGRAM float3 Eval(MaterialParameter &mat, State &state, PerRayData_radiance &prd)
{
	float3 N = state.ffnormal;
	float3 V = -ray.direction;
	float3 L = prd.direction;

	float NDotL = dot(N, L);
	float NDotV = dot(N, V);
	if (NDotL <= 0.0f || NDotV <= 0.0f) return make_float3(0.0f);

	float3 H = normalize(L + V);
	float NDotH = dot(N, H);
	float LDotH = dot(L, H);

	float3 Cdlin = mat.color;
	float Cdlum = 0.3f*Cdlin.x + 0.6f*Cdlin.y + 0.1f*Cdlin.z; // luminance approx.

	float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3(1.0f); // normalize lum. to isolate hue+sat
	float3 Cspec0 = lerp(mat.specular*0.08f*lerp(make_float3(1.0f), Ctint, mat.specularTint), Cdlin, mat.metallic);
	float3 Csheen = lerp(make_float3(1.0f), Ctint, mat.sheenTint);

	// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
	// and mix in diffuse retro-reflection based on roughness
	float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
	float Fd90 = 0.5f + 2.0f * LDotH*LDotH * mat.roughness;
	float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);

	// Based on Hanrahan-Krueger brdf approximation of isotrokPic bssrdf
	// 1.25 scale is used to (roughly) preserve albedo
	// Fss90 used to "flatten" retroreflection based on roughness
	float Fss90 = LDotH*LDotH*mat.roughness;
	float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
	float ss = 1.25f * (Fss * (1.0f / (NDotL + NDotV) - 0.5f) + 0.5f);

	// specular
	//float aspect = sqrt(1-mat.anisotrokPic*.9);
	//float ax = Max(.001f, sqr(mat.roughness)/aspect);
	//float ay = Max(.001f, sqr(mat.roughness)*aspect);
	//float Ds = GTR2_aniso(NDotH, Dot(H, X), Dot(H, Y), ax, ay);
	
	float a = max(0.001f, mat.roughness);
	float Ds = GTR2(NDotH, a);
	float FH = SchlickFresnel(LDotH);
	float3 Fs = lerp(Cspec0, make_float3(1.0f), FH);
	float roughg = sqr(mat.roughness*0.5f + 0.5f);
	float Gs = smithG_GGX(NDotL, roughg) * smithG_GGX(NDotV, roughg);

	// sheen
	float3 Fsheen = FH * mat.sheen * Csheen;

	// clearcoat (ior = 1.5 -> F0 = 0.04)
	float Dr = GTR1(NDotH, lerp(0.1f, 0.001f, mat.clearcoatGloss));
	float Fr = lerp(0.04f, 1.0f, FH);
	float Gr = smithG_GGX(NDotL, 0.25f) * smithG_GGX(NDotV, 0.25f);

	float3 out = ((1.0f / M_PIf) * lerp(Fd, ss, mat.subsurface)*Cdlin + Fsheen)
		* (1.0f - mat.metallic)
		+ Gs*Fs*Ds + 0.25f*mat.clearcoat*Gr*Fr*Dr;

	return out * clamp(dot(N, L), 0.0f, 1.0f);
}
