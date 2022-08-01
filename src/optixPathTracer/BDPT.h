#pragma once 

#ifndef BDPT_H
#define BDPT_H

#include"BDPT_STRUCT.h"
#include "prd.h"
#include "BDenv_device.h"
#include "subspace_device.h"
#include "PG_device.h"
#include "SVM_device.h"
#include"gamma_device.h"
using namespace optix;
RT_FUNCTION optix::float3 DisneyEval(MaterialParameter &mat, optix::float3 normal, optix::float3 V, optix::float3 L);
RT_FUNCTION optix::float3 conVertex(BDPTVertex &lv, BDPTVertex &ev);

rtDeclareVariable(int, light_vertex_count, , ) = { 0 };
rtBuffer<MaterialParameter> sysMaterialParameters;
rtDeclareVariable(int, light_path_count, , ) = { 0 };
rtDeclareVariable(float, sceneMaxLength, , ) = { 1.0 };
rtDeclareVariable(float, average_light_length, , ) = { 1.0f };
rtDeclareVariable(float, DirProjectionArea, , ) = { 1.0f };
rtDeclareVariable(int, light_path_sum, , ) = { 0 };
rtDeclareVariable(int, Q_light_path_sum, , ) = { 0 };
rtDeclareVariable(int, sysNumberOfLights, , );
rtBuffer<ZoneSampler, 1>          zoneLVC;
rtBuffer<PMFCache, 1>        KDPMFCaches;
rtBuffer<ZoneMatrix, 1>           M2_buffer;

rtDeclareVariable(lightSelectionFunction_device*, gamma_p, , ) = { NULL };

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, project_box_max, , );
rtDeclareVariable(float3, project_box_min, , );
rtDeclareVariable(float3, max_box, , );
rtDeclareVariable(float3, min_box, , );
rtDeclareVariable(int, virtual_pmf_id, , ) = {0};
rtDeclareVariable(float, LTC_weight, , ) = { 0.1 };

#ifdef INDIRECT_ONLY
rtDeclareVariable(float, direct_eval_rate, , ) = { 0 };
#endif // INDIRECT_ONLY
#define connectRate_SOL(a, b, c) (M2_buffer[a].r[b] / M2_buffer[a].sum * c / zoneLVC[b].Q * Q_light_path_sum * iterNum)

#define uber_zone_sample_pdf(zoneId) ( UBER_VERTEX_NUM / scene_area)
#define uber_pdf_2(a,b) (M2_buffer[a].r[b] / M2_buffer[a].sum  / uberLVC[b].realSize * uber_zone_sample_pdf(b))
#define uber_pdf_1(a,b) (connectRate(a,b) * UberWidth)
#define uber_pdf(a,b,c) (uber_pdf_1(b,c) * uber_pdf_2(a,b))

#define connectRate(a,b) (M2_buffer[a].r[b] / M2_buffer[a].sum / zoneLVC[b].size * light_path_sum * iterNum) 
#define connectRate_2(a,b) (M3_buffer[a].r[b] / M3_buffer[a].sum / zoneLVC[b].realSize * light_path_count * iterNum) 
//#define connectRate(a,b) (1.0 / average_light_length)

RT_FUNCTION float sqr(float x) { return x * x; }


//__device__ float gamma(BDPTVertex& y, BDPTVertex& z)
//{
//    return Gamma[z.zoneId][y.zoneId] * luminance(y.flux) / Q[y.zoneId];
//}


__device__ float3 hsv2rgb(int h, float s, float v)
{
    float C = v * s;
    float X = C * (1 - abs((float(h % 120) / 60) - 1));
    float m = v - C;
    float3 rgb_;
    if (h < 60)
    {
        rgb_ = make_float3(C, X, 0);
    }
    else if (h < 120)
    {
        rgb_ = make_float3(X, C, 0);
    }
    else if (h < 180)
    {
        rgb_ = make_float3(0, C, X);
    }
    else if (h < 240)
    {
        rgb_ = make_float3(0, X, C);
    }
    else if (h < 300)
    {
        rgb_ = make_float3(X, 0, C);
    }
    else
    {
        rgb_ = make_float3(C, 0, X);
    }
    return make_float3(m) + rgb_;
}

__device__ float3 rgb2hsv(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;
    float cmax = fmaxf(rgb);
    float cmin = fminf(rgb);
    float delta = cmax - cmin;
    float h, s, v;
    if (delta == 0.0f)
    {
        h = 0;
    }
    else if (r >= g && r >= b)
    {
        h = 60 * (int((g - b) / delta) % 6);
    }
    else if (g >= r && g >= b)
    {
        h = 60 * (2 + (b - r) / delta);
    }
    else
    {
        h = 60 * (4 + (r - g) / delta);
    }
    s = cmax == 0 ? 0 : delta / cmax;
    v = cmax;
    return make_float3(h, s, v);
}
RT_FUNCTION float SchlickFresnel(float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2*m; // pow(m,5)
}

RT_FUNCTION float GTR1(float NDotH, float a)
{
    if (a >= 1.0f) return (1.0f / M_PIf);
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
    return (a2 - 1.0f) / (M_PIf*logf(a2)*t);
}

RT_FUNCTION float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
    return a2 / (M_PIf * t*t);
}

RT_FUNCTION float smithG_GGX(float NDotv, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1.0f / (NDotv + sqrtf(a + b - a * b));
}

RT_FUNCTION float fresnel(float cos_theta_i, float cos_theta_t, float eta)
{
    const float rs = (cos_theta_i - cos_theta_t * eta) /
        (cos_theta_i + eta * cos_theta_t);
    const float rp = (cos_theta_i*eta - cos_theta_t) /
        (cos_theta_i*eta + cos_theta_t);

    return 0.5f * (rs*rs + rp * rp);
}

RT_FUNCTION float3 logf(float3 v)
{
    return make_float3(logf(v.x), logf(v.y), logf(v.z));
}

__device__ inline float get_LTC_weight(float3 dir)
{
    float cosA = abs(dot(normalize(dir), normalize(W)));
    float area = length(cross(U, V)) / dot(W,W);
    //float ltc_n =  float(LTC_SAVE_SUM) / LIGHT_VERTEX_NUM * light_path_count; 
    float ltc_n = float(LTC_SAVE_SUM) / average_light_length * 0.1; 
    //ltc_n = 0;
    //rtPrintf("%f\n", ltc_n);
    return 0.0;
    return 1.0 / (cosA * cosA * area) * LTC_weight * ltc_n / 1920/1000;
}

RT_FUNCTION optix::float3 DisneyEval(MaterialParameter &mat, optix::float3 normal, optix::float3 V, optix::float3 L)
{
#ifdef BRDF
    if (mat.brdf)
    {
        return mat.color;
    }
#endif
    float3 N = normal;

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
    float Fss90 = LDotH * LDotH*mat.roughness;
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
        + Gs * Fs*Ds + 0.25f*mat.clearcoat*Gr*Fr*Dr;

    return out;
}
RT_FUNCTION float DisneyPdf(MaterialParameter &mat, optix::float3 normal, optix::float3 V, optix::float3 L,float3 position=make_float3(0.0),bool eye_side = false)
{
#ifdef BRDF
    if (mat.brdf)
        return 1.0f;// return abs(dot(L, normal));
#endif


    float3 n = normal;

    float specularAlpha = max(0.001f, mat.roughness);
    float clearcoatAlpha = lerp(0.1f, 0.001f, mat.clearcoatGloss);

    float diffuseRatio = 0.5f * (1.f - mat.metallic);
    float specularRatio = 1.f - diffuseRatio;

    float3 half = normalize(L + V);

    float cosTheta = abs(dot(half, n));
    float pdfGTR2 = GTR2(cosTheta, specularAlpha) * cosTheta;
    float pdfGTR1 = GTR1(cosTheta, clearcoatAlpha) * cosTheta;

    // calculate diffuse and specular pdfs and mix ratio
    float ratio = 1.0f / (1.0f + mat.clearcoat);
    float pdfSpec = lerp(pdfGTR1, pdfGTR2, ratio) / (4.0 * abs(dot(L, half)));
    float pdfDiff = abs(dot(L, n))* (1.0f / M_PIf);

    // weight pdfs according to ratios
    float pdf =  diffuseRatio * pdfDiff + specularRatio * pdfSpec;

    if (pg_enable && eye_side)
    {
        float pdf2 = pg_api.pdf(position, L);
        pdf = lerp(pdf, pdf2, PG_RATE);
    }
    if (pg_lightSide_enable && eye_side == false)
    {
        float pdf2 = pg_api.pdf_lightside(position, L);
        pdf = lerp(pdf, pdf2, PG_RATE); 
    }
    return pdf;
}
RT_FUNCTION int get_light_zone(LightParameter& L, float2 UV)
{
    int id1 = optix::clamp(static_cast<int>(floorf(UV.x * L.divLevel)), 0, L.divLevel - 1);
    int id2 = optix::clamp(static_cast<int>(floorf(UV.y * L.divLevel)), 0, L.divLevel - 1);
    int res2 = L.divBase - (id1 * L.divLevel + id2);

    int res = SUBSPACE_NUM - res2 - 1;
    if (SUBSPACE_NUM < 1000)
    {
        int target = SUBSPACE_NUM / float(1000) * 200;
        res = res * target / 200.0;
    }
    return SUBSPACE_NUM - res - 1;
    
}

RT_FUNCTION float3 connect_without_mis(BDPTVertex& a, BDPTVertex& b)
{
    float3 connectVec = a.position - b.position;
    float3 connectDir = normalize(connectVec);
    float G = abs(dot(a.normal, connectDir)) * abs(dot(b.normal, connectDir)) / dot(connectVec, connectVec);
    float3 LA = a.lastPosition - a.position;
    float3 LA_DIR = normalize(LA);
    float3 LB = b.lastPosition - b.position;
    float3 LB_DIR = normalize(LB);

    float3 fa, fb;
    float3 ADcolor;
    MaterialParameter mat_a = sysMaterialParameters[a.materialId];
    mat_a.color = a.color;
    fa = DisneyEval(mat_a, a.normal, -connectDir, LA_DIR);

    MaterialParameter mat_b;
    if (!b.isOrigin)
    {
        mat_b = sysMaterialParameters[b.materialId];
        mat_b.color = b.color;
        fb = DisneyEval(mat_b, b.normal, connectDir, LB_DIR);
    }
    else
    {
        if (dot(b.normal, -connectDir) > 0.0f)
        {
            fb = make_float3(0.0f);
        }
        else
        {
            fb = make_float3(1.0f);
        }
    }

    float3 contri = a.flux * b.flux * fa * fb * G;
    float pdf = a.pdf * b.pdf;
    return contri / pdf;
}

//compute the RIS awaness weight for pcbpt
RT_FUNCTION float PCBPT_UNIFORM_RIS_WEIGHT()
{
    float a = 1.0 / average_light_length;
    a /= light_path_count;
    a *= light_path_count / (a + light_path_count - 1);

    a = light_path_count / ((light_path_count - 1) * average_light_length + 1);
    return a;
}

RT_FUNCTION float PCBPT_RIS_WEIGHT(float lum, PMFCache& cache, int b_depth = 1)
{
    if (cache.is_virtual)return PCBPT_UNIFORM_RIS_WEIGHT();
    //lum /= light_path_count;
    float epsilon = 0.001;// *light_path_count; 
    float weight_sum = 0;
    weight_sum += min(lum / cache.Q, 1.0 / epsilon);
    //weight_sum = lerp(weight_sum, 1.0 / average_light_length, UNIFORM_GUIDE_RATE);

    //weight_sum = light_path_count * weight_sum / (weight_sum + light_path_count - 1);

    weight_sum = light_path_count / ((light_path_count - 1) * (1.0 / weight_sum) + 1);
    return abs(weight_sum);
}
RT_FUNCTION float PCBPT_RIS_WEIGHT(float lum, uint pmf_id, int b_depth = 1)
{
    return PCBPT_RIS_WEIGHT(lum, KDPMFCaches[pmf_id], b_depth);
    //lum /= light_path_count;
    float epsilon = 0.001;// *light_path_count;
    int id = pmf_id;
    float weight_sum = 0;
    PMFCache& cache = KDPMFCaches[id];
    weight_sum += min(lum / cache.Q,1.0 / epsilon);
    //weight_sum = lerp(weight_sum, 1.0 / average_light_length, UNIFORM_GUIDE_RATE);

    //weight_sum = light_path_count * weight_sum / (weight_sum + light_path_count - 1);

    weight_sum = light_path_count / ((light_path_count - 1) * (1.0 / weight_sum) + 1);
    return abs(weight_sum );
}
RT_FUNCTION float PCBPT_RIS_WEIGHT(float lum, uint3 pmf_ids,int b_depth = 1)
{
     


    if (b_depth == 0)
    { 
        lum /= PCBPT_DIRECT_FACTOR;
    }
    float epsilon = 0.001;// *light_path_count;
    int ids[3] = { pmf_ids.x,pmf_ids.y,pmf_ids.z };
    //float T_MAX = fmaxf(make_float3(KDPMFCaches[ids[0]].Q_variance, KDPMFCaches[ids[1]].Q_variance, KDPMFCaches[ids[2]].Q_variance));
    float weight_sum = 0;


    for (int i = 0; i < 3; i++)
    {
        weight_sum += PCBPT_RIS_WEIGHT(lum, ids[i]);
    }

    weight_sum += PCBPT_UNIFORM_RIS_WEIGHT();
    weight_sum /= 4;
    return weight_sum;


    //code below  
    //dead code 
    for (int i = 0; i < 3; i++)
    {
        PMFCache& cache = KDPMFCaches[ids[i]];
        weight_sum += min((lum + 0.000001) /(cache.Q),1.0 / epsilon); 
    }
    weight_sum /= 3;
    //weight_sum = lerp(weight_sum, 1.0 / average_light_length, UNIFORM_GUIDE_RATE);
    
    float U = UNIFORM_GUIDE_RATE;
    //U += 4 * sqrt(T_MAX) / (light_path_count * average_light_length);
    U = min(1.0, U); 
    weight_sum = lerp(weight_sum, 1.0 / average_light_length, U); 
    float ris_M = light_path_count; 
    weight_sum = ris_M * weight_sum / (weight_sum + ris_M - 1);

    return weight_sum;


}
RT_FUNCTION float RIS_light_subpath_lum(BDPTVertex& light_vertex, float3 position, float3 normal)
{
    if (light_vertex.depth == 0)
    {
        if (light_vertex.type == DIRECTION || light_vertex.type == ENV)
        {
            float3 lum = light_vertex.flux / light_vertex.pdf / sky.projectPdf() * abs(dot(light_vertex.normal, normal));
            return ENERGY_WEIGHT(lum) * light_path_count;

        }
        else  
        {
            float3 diff = position - light_vertex.position;
            float3 dir = normalize(diff);
            float d = dot(diff, diff);

            float3 lum = light_vertex.flux / light_vertex.pdf * abs(dot(dir,normal) * dot(dir,light_vertex.normal)) / d ;
            return ENERGY_WEIGHT(lum) * light_path_count;            
        }

    }
    MaterialParameter mat = sysMaterialParameters[light_vertex.materialId];
    mat.color = light_vertex.color;
    float3 C_VEC = position - light_vertex.position;
    float3 C_DIR = normalize(C_VEC);
    float3 L_DIR = normalize(light_vertex.lastPosition - light_vertex.position);
    float3 f = DisneyEval(mat, light_vertex.normal, C_DIR, L_DIR);
    float G = abs(dot(light_vertex.normal, C_DIR)) * abs(dot(normal, C_DIR)) / (length(C_VEC) * length(C_VEC));
    float3 flux = f * G * light_vertex.flux / light_vertex.pdf;
    return light_path_count * ENERGY_WEIGHT(flux);

}
RT_FUNCTION int randomSampleZoneMatrix(ZoneMatrix &a, float random)
{
    float index = random * a.sum;
    int mid = SUBSPACE_NUM / 2 - 1, l = 0, r = SUBSPACE_NUM;
    while (r - l > 1)
    {
        if (index < a.m[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }
    return l;
}

RT_FUNCTION int binary_index_by_array(float* a, int size, float random, float& pdf)//normalize
{
    float index = random;
    int mid = size / 2 - 1, l = 0, r = size;
    while (r - l > 1)
    {
        if (index < a[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }

    pdf = l == size - 1 ? 1.0 - a[l]: a[l + 1] - a[l];
    return l;
}
RT_FUNCTION BDPTVertex& randomSampleZoneLVC(ZoneSampler &a, float random)
{
    int size = min(a.realSize, MAX_LIGHT_VERTEX_PER_TRIANGLE);
    return a.v[optix::clamp(static_cast<int>(floorf(random * size)), 0, size - 1)];
    //return optix::clamp(static_cast<int>(floorf(random * a.size)), 0, a.size - 1);
    /*float index = random * a.sum;
    int mid = a.size / 2 - 1, l = 0, r = a.size;
    while (r - l > 1)
    {
        if (index < a.m[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }
    return l;*/
}
RT_FUNCTION BDPTVertex& randomSOL_ZoneLVC(ZoneSampler &a, float random)
{
    int size = min(a.realSize, MAX_LIGHT_VERTEX_PER_TRIANGLE);
    //return a.v[optix::clamp(static_cast<int>(floorf(random * size)), 0, size - 1)]; 
    float index = random * a.m[size-1];
    int mid = size / 2 - 1, l = 0, r = size;
    while (r - l > 1)
    {
        if (index < a.m[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }
    return a.v[l];
     
}
RT_FUNCTION int randomSelectVertexFromPMFCache(PMFCache &a, float random)
{
    float index = random * a.sum;
    int mid = light_vertex_count / 2 - 1, l = 0, r = light_vertex_count;
    while (r - l > 1)
    {
        if (index < a.m[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    } 
    return l;
}

RT_FUNCTION float test_MIS_weight(BDPTVertex& a, BDPTVertex& b, float3 energy)
{
    float current_weight = 1.0f;
    if (b.depth == 0)
    {
        return 1.0;
    }
#ifdef ZGCBPT

    //current_weight = connectRate(a.zoneId, b.zoneId);
#ifdef ZGC_SAMPLE_ON_LUM

    //float3 current_weight_SOL = connectRate_SOL(a.zoneId, b.zoneId, energy);
    //current_weight = current_weight_SOL.x + current_weight_SOL.y + current_weight_SOL.z;
#endif
#endif // ZGCBPT

#ifdef PCBPT 
    float3 v1 = normalize(b.lastPosition - b.position);
    float3 v2 = normalize(a.position - b.position);
    MaterialParameter mat = sysMaterialParameters[b.materialId];
    mat.color = b.color;
    float3 fb = DisneyEval(mat, b.normal, v1, v2) / (mat.brdf ? abs(dot(b.normal, v2)) : 1.0f);

    float3 connectVec = a.position - b.position;
    float3 connectDir = normalize(connectVec);
    float G = abs(dot(a.normal, connectDir)) * abs(dot(b.normal, connectDir)) / dot(connectVec, connectVec);



    float3 Le = fb;
#ifdef PCBPT_STANDARD_MIS
    Le *= energy;
#endif
    current_weight *= (Le.x + Le.y + Le.z) * G * light_path_count / KDPMFCaches[a.pmf_id].Q;
#ifdef KD_3
    current_weight = 1.0f;
    float r = (Le.x + Le.y + Le.z) * G * light_path_count;
    current_weight *= connectRate_kd3(r, a.pmf_kd3);
#endif 
    current_weight = lerp(current_weight, 1.0 / average_light_length, UNIFORM_GUIDE_RATE);
    current_weight = light_path_count * current_weight / (current_weight + light_path_count - 1);

#endif // PCBPT

    return current_weight;
}
RT_FUNCTION float3 test_energy_transform(BDPTVertex& a, BDPTVertex& b, float3 energy)
{
    if (b.depth == 0)
    {
        if (b.type == DIRECTION)
        {
            return energy * DirProjectionArea;
        }
        return M_PIf * energy;
    }
    else
    {
        float3 v1 = normalize(b.lastPosition - b.position);
        float3 v2 = normalize(a.position - b.position);
        MaterialParameter mat = sysMaterialParameters[b.materialId];
        mat.color = b.color;
        float3 f = DisneyEval(mat, b.normal, v1, v2) / (mat.brdf ? abs(dot(b.normal, v2)) : 1.0f);
        float pdf = DisneyPdf(mat, b.normal, v1, v2,b.position);

        float rr_rate = fmaxf(mat.color);
#ifdef RR_MIN_LIMIT
        rr_rate = max(rr_rate, MIN_RR_RATE);
#endif
#ifdef RR_DISABLE
        rr_rate = 1.0f;
#endif
        return energy * f / pdf / rr_rate * abs(dot(b.normal, v2));
    }
}
RT_FUNCTION float3 color_int2float(int R, int G, int B)
{
    R = clamp(R, 0, 255);
    G = clamp(G, 0, 255);
    B = clamp(B, 0, 255);
    return make_float3(R / 256.0, G / 256.0, B / 256.0);
}
RT_FUNCTION float3 jet_color(int grey)
{
    grey *= 2.55/2;
    grey = clamp(grey, 0, 255);
    int i = grey; 
    if (grey <= 32)
    {
        return color_int2float(0, 0, 128 + 4 * i);
    }
    i -= 32;
    if (grey <= 96)
    {
        return color_int2float(0, 4 + 4 * i, 255);
    }
    i -= 64;
    if (grey <= 160)
    {
        return color_int2float(6 + 4 * i, 255, 250 - 4 * i);
    }
    i -= 64;
    if (grey <= 224)
    {
        return color_int2float(255, 252 - 4 * i, 0);
    }
    i -= 64;
    return color_int2float(252 - i * 4, 0, 0);
}

RT_FUNCTION OptimizationPathInfo OPT_info_from_path(BDPTVertexStack& path)
{
 //本代码目前仅适用于面光源无玻璃无pathGuiding场合，其他时候需要另行修改实现   
    OptimizationPathInfo res;
    if (path.size < 3)
    {
        return res;
    }
    BDPTVertex& end = path.v[path.size - 1];
    res.contri = luminance(end.flux);
    res.actual_pdf = end.pdf ; 
    res.path_length = path.size;
    for (int i = 0; i <= path.size - 1; i++)
    {
        res.ss_ids[i] = path.v[i].zoneId;
    }

    float inver_pdfs[OPT_PATH_LENGTH];
    //inver_pdfs[i] 为生成下标为i的光顶点的singlePdf
    inver_pdfs[path.size - 1] = 1.0 / sysNumberOfLights * end.pg_lightPdf.x;
    for (int i = path.size - 2; i > 0; i--)
    {
        BDPTVertex& MidVertex = path.v[i];
        BDPTVertex& LastVertex = path.v[i+1];
        BDPTVertex& NextVertex = path.v[i-1];
        float3 wi_vec = LastVertex.position - MidVertex.position; 
        float3 wi = normalize(wi_vec); 
        float pdf_G = abs(dot(MidVertex.normal, wi) * dot(LastVertex.normal, wi)) / dot(wi_vec, wi_vec);
        float bsdf;
        //此处bsdf包括了bsdf项和俄罗斯轮盘赌项同时去除了与G项中重合的出射角度影响
        if (LastVertex.isOrigin)
        {
            bsdf = 1.0 / M_PI;
        }
        else
        {
            BDPTVertex& LastLastVertex = path.v[i + 2];
            float3 wii = normalize(LastLastVertex.position - LastVertex.position);
            MaterialParameter mat = sysMaterialParameters[LastVertex.materialId];
            mat.color = LastVertex.color;
            float RR_rate = max(fmaxf(mat.color), MIN_RR_RATE);
            bsdf = DisneyPdf(mat, LastVertex.normal, wii, -wi, LastVertex.position, true) / abs(dot(LastVertex.normal,wi)) * RR_RATE;
        }
        inver_pdfs[i] = bsdf * pdf_G;
    }

    res.light_pdfs[path.size - 1] = inver_pdfs[path.size - 1];
    for (int i = path.size - 2; i > 0; i--)
    {
        res.light_pdfs[i] = res.light_pdfs[i + 1] * inver_pdfs[i];
    }

    //res.pdfs[i]表示以顶点i为末端视路径时的策略的pdf
    //float c_pdf = end.pdf;
    for (int i = path.size - 2; i !=0 ; i--)
    {
        res.pdfs[i] = path.v[i].pdf * res.light_pdfs[i + 1];
        //res.pdfs[i] = c_pdf;
        //c_pdf *= inver_pdfs[i - 1] / path.v[i].singlePdf;
    }
    res.pdfs[path.size - 1] = end.pdf;

    float3 fconn[OPT_PATH_LENGTH];
    //fconn[i]代表顶点i与顶点i+1连接的局部contri
    for (int i = 1; i < path.size - 1; i++)
    {
        BDPTVertex& L1 = path.v[i - 1];
        BDPTVertex& L0 = path.v[i];
        BDPTVertex& R0 = path.v[i + 1];
        BDPTVertex& R1 = path.v[i + 2];

        MaterialParameter mat_a = sysMaterialParameters[L0.materialId];
        mat_a.color = L0.color;
        float3 bsdf_a = DisneyEval(mat_a, L0.normal, normalize(L1.position - L0.position), normalize(R0.position - L0.position));

        float3 bsdf_b;
        if (i == path.size - 2)
        {
            bsdf_b = make_float3(1);
        }
        else
        {
            MaterialParameter mat_b = sysMaterialParameters[R0.materialId];
            mat_b.color = R0.color;
            bsdf_b = DisneyEval(mat_b, R0.normal, normalize(L0.position - R0.position), normalize(R1.position - R0.position));
        }

        float3 connectDir = normalize(L0.position - R0.position);
        float3 connectVec = L0.position - R0.position;
        float G = abs(dot(connectDir, L0.normal) * dot(connectDir, R0.normal)) / (dot(connectVec, connectVec));

        fconn[i] = bsdf_a * bsdf_b * G;
    } 

    for (int i = 1; i < path.size; i++)
    {
        float3 light_contri;
        if (i == path.size - 1)
        {
            light_contri = end.flux / path.v[path.size - 2].flux / fconn[path.size - 2];
        }
        else
        {
            light_contri = end.flux / path.v[i].flux;
        }
        float weight = light_contri.x + light_contri.y + light_contri.z;
        res.light_contris[i] = weight;
    }
    res.light_contris[0] = luminance(end.flux);


    return res;
}


RT_FUNCTION OptimizationPathInfo OPT_info_from_path_NEE(BDPTVertexStack& path,BDPTVertex &b)
{
    //本代码目前仅适用于面光源无玻璃无pathGuiding场合，其他时候需要另行修改实现   
    OptimizationPathInfo res;
    if (path.size < 2)
    {
        return res;
    }
    BDPTVertex& a = path.v[path.size - 1];

    path.v[path.size] = b;
    path.size++;

    float3 connectVec = a.position - b.position;
    float3 connectDir = normalize(connectVec);
    float G = abs(dot(a.normal, connectDir)) * abs(dot(b.normal, connectDir)) / dot(connectVec, connectVec);
    float3 LA = a.lastPosition - a.position;
    float3 LA_DIR = normalize(LA);
    float3 LB = b.lastPosition - b.position;
    float3 LB_DIR = normalize(LB);

    float3 fa, fb;
    MaterialParameter mat_a = sysMaterialParameters[a.materialId];
    mat_a.color = a.color;
    fa = DisneyEval(mat_a, a.normal, -connectDir, LA_DIR) / (mat_a.brdf ? abs(dot(a.normal, connectDir)) : 1.0f);


    if (dot(b.normal, -connectDir) > 0.0f)
    {
        fb = make_float3(0.0f);
    }
    else
    {
        fb = make_float3(1.0f);
    }


    res.contri = luminance(a.flux * b.flux * fa * fb * G);
    res.actual_pdf = a.pdf * b.pdf;


    res.path_length = path.size;
    for (int i = 0; i <= path.size - 1; i++)
    {
        res.ss_ids[i] = path.v[i].zoneId;
        res.positions[i] = path.v[i].position;
    }

    float inver_pdfs[OPT_PATH_LENGTH];
    //inver_pdfs[i] 为生成下标为i的光顶点的singlePdf
    inver_pdfs[path.size - 1] = b.pdf;
    for (int i = path.size - 2; i > 0; i--)
    {
        BDPTVertex& MidVertex = path.v[i];
        BDPTVertex& LastVertex = path.v[i + 1];
        BDPTVertex& NextVertex = path.v[i - 1];
        float3 wi_vec = LastVertex.position - MidVertex.position;
        float3 wi = normalize(wi_vec);
        float pdf_G = abs(dot(MidVertex.normal, wi) * dot(LastVertex.normal, wi)) / dot(wi_vec, wi_vec);
        float bsdf;
        //此处bsdf包括了bsdf项和俄罗斯轮盘赌项同时去除了与G项中重合的出射角度影响
        if (LastVertex.isOrigin)
        {
            bsdf = 1.0 / M_PI;
        }
        else
        {
            BDPTVertex& LastLastVertex = path.v[i + 2];
            float3 wii = normalize(LastLastVertex.position - LastVertex.position);
            MaterialParameter mat = sysMaterialParameters[LastVertex.materialId];
            mat.color = LastVertex.color;
            float RR_rate = max(fmaxf(mat.color), MIN_RR_RATE);
            bsdf = DisneyPdf(mat, LastVertex.normal, wii, -wi, LastVertex.position, true) / abs(dot(LastVertex.normal, wi)) * RR_rate;
        }
        inver_pdfs[i] = bsdf * pdf_G;
    }

    res.light_pdfs[path.size - 1] = b.pdf;
    for (int i = path.size - 2; i > 0; i--)
    {
        res.light_pdfs[i] = res.light_pdfs[i + 1] * inver_pdfs[i];
    }

    //res.pdfs[i]表示以顶点i为末端视路径时的策略的pdf
    //float c_pdf = end.pdf;
    for (int i = path.size - 2; i != 0; i--)
    {
        res.pdfs[i] = path.v[i].pdf * res.light_pdfs[i + 1];
        //res.pdfs[i] = c_pdf;
        //c_pdf *= inver_pdfs[i - 1] / path.v[i].singlePdf;
    }

    float single_pdf;
    {
        BDPTVertex& MidVertex = a;
        BDPTVertex& LastVertex = path.v[path.size - 3];
        BDPTVertex& NextVertex = b;
        float3 wi_vec = NextVertex.position - MidVertex.position;
        float3 wi = normalize(wi_vec);
        float pdf_G = abs( dot(NextVertex.normal, wi)) / dot(wi_vec, wi_vec);
        float bsdf;
        //此处bsdf包括了bsdf项和俄罗斯轮盘赌项同时去除了与G项中重合的出射角度影响
         
        float3 wo = normalize(LastVertex.position - MidVertex.position);
        MaterialParameter mat = sysMaterialParameters[MidVertex.materialId];
        mat.color = MidVertex.color;
        float RR_rate = max(fmaxf(mat.color), MIN_RR_RATE);
        bsdf = DisneyPdf(mat, MidVertex.normal, wo, wi, MidVertex.position, true) *RR_rate;

        single_pdf = bsdf * pdf_G;
    }
    res.pdfs[path.size - 1] = a.pdf * single_pdf;

    float3 fconn[OPT_PATH_LENGTH];
    //fconn[i]代表顶点i与顶点i+1连接的局部contri
    for (int i = 1; i < path.size - 1; i++)
    {
        BDPTVertex& L1 = path.v[i - 1];
        BDPTVertex& L0 = path.v[i];
        BDPTVertex& R0 = path.v[i + 1];
        BDPTVertex& R1 = path.v[i + 2];

        MaterialParameter mat_a = sysMaterialParameters[L0.materialId];
        mat_a.color = L0.color;
        float3 bsdf_a = DisneyEval(mat_a, L0.normal, normalize(L1.position - L0.position), normalize(R0.position - L0.position));

        float3 bsdf_b;
        if (i == path.size - 2)
        {
            bsdf_b = make_float3(1);
        }
        else
        {
            MaterialParameter mat_b = sysMaterialParameters[R0.materialId];
            mat_b.color = R0.color;
            bsdf_b = DisneyEval(mat_b, R0.normal, normalize(L0.position - R0.position), normalize(R1.position - R0.position));
        }

        float3 connectDir = normalize(L0.position - R0.position);
        float3 connectVec = L0.position - R0.position;
        float G = abs(dot(connectDir, L0.normal) * dot(connectDir, R0.normal)) / (dot(connectVec, connectVec));

        fconn[i] = bsdf_a * bsdf_b * G;
    }

    float3 last_light_contri;
    for (int i = path.size - 1; i > 0; i--)
    {
        float3 light_contri;
        if (i == path.size - 1)
        {
            light_contri = b.flux;
        }
        else
        {
            light_contri = last_light_contri * fconn[i];
        }
        
        float weight = light_contri.x + light_contri.y + light_contri.z;
        res.light_contris[i] = weight;
        last_light_contri = light_contri;
    }
    res.light_contris[0] = res.contri;

    path.size--;
    return res;
}
#ifndef TOBEREWRITE

RT_FUNCTION optix::float3 contriCompute(BDPTVertexStack &path)
{
    //要求：第0个顶点为eye，第size-1个顶点为light
    optix::float3 throughput = make_float3(pow(M, path.size));
    BDPTVertex & light = path.v[path.size - 1];
    BDPTVertex & lastMidPoint = path.v[path.size - 2];
    optix::float3 lightLine = lastMidPoint.position - light.position;
    optix::float3 lightDirection = normalize(lightLine);
    float lAng = dot(light.normal, lightDirection);
    if (lAng < 0.0f)
    {
        return make_float3(0.0f);
    }
    optix::float3 Le = light.emission * lAng;
    throughput *= Le;
    for (int i = 1; i < path.size; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        optix::float3 line = midPoint.position - lastPoint.position;
        throughput /= dot(line, line);
    }
    for (int i = 1; i < path.size - 1; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        BDPTVertex &nextPoint = path.v[i + 1];
        optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
        throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
            * DisneyEval(midPoint.material, midPoint.normal, lastDirection, nextDirection);
    }
    return throughput;
}
RT_FUNCTION optix::float3 contriCompute_test(BDPTVertexStack &path, int eyepathLenth)
{
    //要求：第0个顶点为eye，第size-1个顶点为light
    optix::float3 throughput = make_float3(pow(M, path.size));
    BDPTVertex & light = path.v[path.size - 1];
    BDPTVertex & lastMidPoint = path.v[path.size - 2];
    optix::float3 lightLine = lastMidPoint.position - light.position;
    optix::float3 lightDirection = normalize(lightLine);
    float lAng = dot(light.normal, lightDirection);
    if (lAng < 0.0f)
    {
        return make_float3(0.0f);
    }
    optix::float3 Le = light.emission * lAng;
    throughput *= Le;
    for (int i = 1; i < path.size; i++)
    {
        if (i <= eyepathLenth)
            continue;
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        optix::float3 line = midPoint.position - lastPoint.position;
        throughput /= dot(line, line);
    }
    for (int i = 1; i < path.size - 1; i++)
    {
        if (i < eyepathLenth)
            continue;
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        BDPTVertex &nextPoint = path.v[i + 1];
        optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
        if (i == eyepathLenth)
        {
            throughput *= abs(dot(midPoint.normal, nextDirection));
            continue;
        }
        throughput *= abs(dot(midPoint.normal, lastDirection)) * abs(dot(midPoint.normal, nextDirection))
            * DisneyEval(midPoint.material, midPoint.normal, lastDirection, nextDirection);
    }
    return throughput;
}RT_FUNCTION double pdfCompute_test(BDPTVertexStack &path, int lightPathLength)
{
    int eyePathLength = path.size - lightPathLength;
    /*这里用double是考虑了精度问题，否则在大场景时很可能把float的精度给爆了，但是据说double的运算速率要比float慢很多，或许在以后可以考虑这个优化*/
    double pdf = pow(M, path.size);
    /*光源默认为面光源上一点，因此可以用cos来近似模拟其光照效果，如果是点光源需要修改以下代码*/

    /*俄罗斯轮盘赌的pdf影响*/
    if (lightPathLength > RR_BEGIN_DEPTH)
    {
        pdf *= pow(RR_RATE, lightPathLength - RR_BEGIN_DEPTH);
    }
    if (lightPathLength > 0)
    {
        BDPTVertex &light = path.v[path.size - 1];
        pdf *= light.pdf;
    }
    if (lightPathLength > 1)
    {
        BDPTVertex &light = path.v[path.size - 1];
        BDPTVertex & lastMidPoint = path.v[path.size - 2];
        optix::float3 lightLine = lastMidPoint.position - light.position;
        optix::float3 lightDirection = normalize(lightLine);
        pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

        /*因距离和倾角导致的pdf*/
        for (int i = 1; i < lightPathLength; i++)
        {
            BDPTVertex &midPoint = path.v[path.size - i - 1];
            BDPTVertex &lastPoint = path.v[path.size - i];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }

        for (int i = 1; i < lightPathLength - 1; i++)
        {
            BDPTVertex &midPoint = path.v[path.size - i - 1];
            BDPTVertex &lastPoint = path.v[path.size - i];
            BDPTVertex &nextPoint = path.v[path.size - i - 2];
            optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
            pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection,midPoint.position);
        }

    }
    return pdf;
}

/*参数lightPathLength为0时，代表由eye path直接打到光源上*/
RT_FUNCTION double pdfCompute(BDPTVertexStack &path, int lightPathLength)
{
    int eyePathLength = path.size - lightPathLength;
    /*这里用double是考虑了精度问题，否则在大场景时很可能把float的精度给爆了，但是据说double的运算速率要比float慢很多，或许在以后可以考虑这个优化*/
    double pdf = pow(M, path.size);
    /*光源默认为面光源上一点，因此可以用cos来近似模拟其光照效果，如果是点光源需要修改以下代码*/

    /*俄罗斯轮盘赌的pdf影响*/
    if (lightPathLength > RR_BEGIN_DEPTH)
    {
        pdf *= pow(RR_RATE, lightPathLength - RR_BEGIN_DEPTH);
    }
    if (eyePathLength > RR_BEGIN_DEPTH)
    {
        pdf *= pow(RR_RATE, eyePathLength - RR_BEGIN_DEPTH);
    }
    if (lightPathLength > 0)
    {
        BDPTVertex &light = path.v[path.size - 1];
        pdf *= light.pdf;
    }
    if (lightPathLength > 1)
    {
        BDPTVertex &light = path.v[path.size - 1];
        BDPTVertex & lastMidPoint = path.v[path.size - 2];
        optix::float3 lightLine = lastMidPoint.position - light.position;
        optix::float3 lightDirection = normalize(lightLine);
        pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

        /*因距离和倾角导致的pdf*/
        for (int i = 1; i < lightPathLength; i++)
        {
            BDPTVertex &midPoint = path.v[path.size - i - 1];
            BDPTVertex &lastPoint = path.v[path.size - i];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }

        for (int i = 1; i < lightPathLength - 1; i++)
        {
            BDPTVertex &midPoint = path.v[path.size - i - 1];
            BDPTVertex &lastPoint = path.v[path.size - i];
            BDPTVertex &nextPoint = path.v[path.size - i - 2];
            optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
            pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection,midPoint.position);
        }

    }
    /*由于投影角导致的pdf变化*/
    for (int i = 1; i < eyePathLength; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        optix::float3 line = midPoint.position - lastPoint.position;
        optix::float3 lineDirection = normalize(line);
        pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
    }
    /*采样方向的概率*/
    for (int i = 1; i < eyePathLength - 1; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        BDPTVertex &nextPoint = path.v[i + 1];
        optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
        pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection,midPoint.position);
    }
    return pdf;
}
RT_FUNCTION optix::float3 evalPath(BDPTVertexStack &path)
{
    double pdf = 0.0f;
    optix::float3 contri;
    contri = contriCompute(path);
    /*这里应该随着策略的不同而有所取舍，目前的版本不会直接从眼睛连到light path的末端,也不会从eye path直接打到光源上*/
    for (int i = 1; i < path.size - 1; i++)
    {
        pdf += pdfCompute(path, i);
    }
    optix::float3 ans = contri / pdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
    {
        return make_float3(0.0f);
    }
    return contri / pdf;
}
RT_FUNCTION int randomSelectFromDirector(MeshLightDirector &a, float random)
{
    return a.v[optix::clamp(static_cast<int>(floorf(random * a.size)), 0, a.size - 1)];
    //return optix::clamp(static_cast<int>(floorf(random * a.size)), 0, a.size - 1);
    /*float index = random * a.sum;
    int mid = a.size / 2 - 1, l = 0, r = a.size;
    while (r - l > 1)
    {
        if (index < a.m[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }
    return l;*/
}
RT_FUNCTION bool vertexInDirector(MeshLightDirector &a, int k)
{
    for (int i = 0; i < a.size; i++)
    {
        if (a.v[i] == k)
        {
            return true;
        }
    }
    return false;
}
RT_FUNCTION int randomSelectMeshFromDirector(Mesh2MeshDirector &a, float random)
{
    float index = random * a.sum;
    int mid = SUBSPACE_NUM / 2 - 1, l = 0, r = SUBSPACE_NUM;
    while (r - l > 1)
    {
        if (index < a.m[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }
    return l;
}

RT_FUNCTION double mg_pdfCompute(BDPTVertexStack &path, int lightPathLength, Mesh2MeshDirector *M2, MeshLightDirector *p, int maxLength)
{
    int eyePathLength = path.size - lightPathLength;
    /*这里用double是考虑了精度问题，否则在大场景时很可能把float的精度给爆了，但是据说double的运算速率要比float慢很多，或许在以后可以考虑这个优化*/
    double pdf = pow(M, path.size);
    /*光源默认为面光源上一点，因此可以用cos来近似模拟其光照效果，如果是点光源需要修改以下代码*/
    if (lightPathLength > maxLength + 1 || eyePathLength > maxLength + 1)
    {
        return 0.0f;
    }
    /*俄罗斯轮盘赌的pdf影响*/
    if (lightPathLength > RR_BEGIN_DEPTH)
    {
        pdf *= pow(RR_RATE, lightPathLength - RR_BEGIN_DEPTH);
    }
    if (eyePathLength > RR_BEGIN_DEPTH)
    {
        pdf *= pow(RR_RATE, eyePathLength - RR_BEGIN_DEPTH);
    }
    if (lightPathLength > 0)
    {
        BDPTVertex &light = path.v[path.size - 1];
        pdf *= light.pdf;
    }
    if (lightPathLength > 1)
    {
        BDPTVertex &light = path.v[path.size - 1];
        BDPTVertex & lastMidPoint = path.v[path.size - 2];
        optix::float3 lightLine = lastMidPoint.position - light.position;
        optix::float3 lightDirection = normalize(lightLine);
        pdf *= abs(dot(lightDirection, light.normal)) / M_PI;

        /*因距离和倾角导致的pdf*/
        for (int i = 1; i < lightPathLength; i++)
        {
            BDPTVertex &midPoint = path.v[path.size - i - 1];
            BDPTVertex &lastPoint = path.v[path.size - i];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            pdf *= 1.0 / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }

        for (int i = 1; i < lightPathLength - 1; i++)
        {
            BDPTVertex &midPoint = path.v[path.size - i - 1];
            BDPTVertex &lastPoint = path.v[path.size - i];
            BDPTVertex &nextPoint = path.v[path.size - i - 2];
            optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
            optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
            pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection,midPoint.position);
        }

    }
    /*由于投影角导致的pdf变化*/
    for (int i = 1; i < eyePathLength; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        optix::float3 line = midPoint.position - lastPoint.position;
        optix::float3 lineDirection = normalize(line);
        pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
    }
    /*采样方向的概率*/
    for (int i = 1; i < eyePathLength - 1; i++)
    {
        BDPTVertex &midPoint = path.v[i];
        BDPTVertex &lastPoint = path.v[i - 1];
        BDPTVertex &nextPoint = path.v[i + 1];
        optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
        optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
        pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
    }
    BDPTVertex &eyeEndVertex = path.v[eyePathLength - 1];
    BDPTVertex &lightEndVertex = path.v[eyePathLength];
    int pIdx = eyeEndVertex.primIdx;
    int LPidx = lightEndVertex.primIdx;
    float mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * p[LPidx].pdf;
    if (p[LPidx].size == 0 || M2[pIdx].sum == 0)
        return 0.0f;
    //if (M2[pIdx].r[LPidx] != 0)
    //    rtPrintf("%f %d %d %d-%d %f\n", 1.0f / mg_rate, M2[pIdx].r[LPidx], M2[pIdx].sum,pIdx,LPidx, p[LPidx].pdf);
    //rtPrintf("%f %d %d %d-%d %f\n", mg_rate, M2[pIdx].r[LPidx], M2[pIdx].sum, pIdx, LPidx, p[LPidx].pdf);
    return pdf * mg_rate;
}

RT_FUNCTION optix::float3 mg_evalPath(BDPTVertexStack &path, Mesh2MeshDirector *M2, MeshLightDirector *p, int maxLength)
{
    double pdf = 0.0f;
    optix::float3 contri;
    contri = contriCompute(path);
    /*这里应该随着策略的不同而有所取舍，目前的版本不会直接从眼睛连到light path的末端,也不会从eye path直接打到光源上*/
    for (int i = 1; i < path.size - 1; i++)
    {
        pdf += mg_pdfCompute(path, i, M2, p, maxLength);
    }
    optix::float3 ans = contri / pdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
    {
        return make_float3(0.0f);
    }
    return contri / pdf;
}
RT_FUNCTION optix::float3 nmg_evalPath(BDPTVertexStack &path, Mesh2MeshDirector *M2, MeshLightDirector *p, int maxLength)
{
    optix::float3 contri;
    contri = contriCompute(path);

    path.v[0].epdf = 1.0f;
    for (int i = 1; i < path.size; i++)
    {
        int li = path.size - i - 1;
        if (i > maxLength)
        {
            path.v[i].epdf = 0.0f;
            path.v[li].pdf = 0.0f;
            continue;
        }
        path.v[i].epdf = M;
        path.v[li].pdf = M;

        if (i >= RR_BEGIN_DEPTH)
        {
            path.v[i].epdf *= RR_RATE;
            path.v[li].pdf *= RR_RATE;
        }
        {//距离与投影导致的几何项
            BDPTVertex &midPoint = path.v[i];
            BDPTVertex &lastPoint = path.v[i - 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[i].epdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        {//距离与投影导致的几何项(光)
            BDPTVertex &midPoint = path.v[li];
            BDPTVertex &lastPoint = path.v[li + 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[li].pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        if (i == 1)
        {
            BDPTVertex &light = path.v[li + 1];
            BDPTVertex & lastMidPoint = path.v[li];
            optix::float3 lightLine = lastMidPoint.position - light.position;
            optix::float3 lightDirection = normalize(lightLine);
            path.v[li].pdf *= abs(dot(lightDirection, light.normal)) / M_PI;
        }
        else
        {
            {//brdf cos项，眼

                BDPTVertex &midPoint = path.v[i - 1];
                BDPTVertex &lastPoint = path.v[i - 2];
                BDPTVertex &nextPoint = path.v[i];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[i].epdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
            {//brdf cos项，光

                BDPTVertex &midPoint = path.v[li + 1];
                BDPTVertex &lastPoint = path.v[li + 2];
                BDPTVertex &nextPoint = path.v[li];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[li].pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
        }
        path.v[i].epdf *= path.v[i - 1].epdf;
        path.v[li].pdf *= path.v[li + 1].pdf;
    }

    float pdf = 0.0f;
    for (int i = 1; i < path.size - 1; i++)
    {
        BDPTVertex &eyeEndVertex = path.v[i];
        BDPTVertex &lightEndVertex = path.v[i + 1];
        int pIdx = eyeEndVertex.primIdx;
        int LPidx = lightEndVertex.primIdx;
        //float mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * p[LPidx].pdf;
        float Lweight = lightEndVertex.emission.x + lightEndVertex.emission.y + lightEndVertex.emission.z;
        float mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * p[LPidx].pdf;
        if (p[LPidx].size == 0 || M2[pIdx].sum == 0)
            mg_rate = 0.0f;
        pdf += path.v[i].epdf * path.v[i + 1].pdf * mg_rate * M * M;
    }
    //pdf += path.v[path.size - 1].epdf * M * M;
    optix::float3 ans = contri / pdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
    {
        return make_float3(0.0f);
    }
    return contri / pdf;
}


RT_FUNCTION optix::float3 lbpt_evalPath(BDPTVertexStack &path, int maxLength, float bufferWeight)
{
    optix::float3 contri;
    contri = contriCompute(path);

    path.v[0].epdf = 1.0f;
    for (int i = 1; i < path.size; i++)
    {
        int li = path.size - i - 1;
        if (i > maxLength)
        {
            path.v[i].epdf = 0.0f;
            path.v[li].pdf = 0.0f;
            continue;
        }
        path.v[i].epdf = M;
        path.v[li].pdf = M;

        if (i >= RR_BEGIN_DEPTH)
        {
            path.v[i].epdf *= RR_RATE;
            path.v[li].pdf *= RR_RATE;
        }
        {//距离与投影导致的几何项
            BDPTVertex &midPoint = path.v[i];
            BDPTVertex &lastPoint = path.v[i - 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[i].epdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        {//距离与投影导致的几何项(光)
            BDPTVertex &midPoint = path.v[li];
            BDPTVertex &lastPoint = path.v[li + 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[li].pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        if (i == 1)
        {
            BDPTVertex &light = path.v[li + 1];
            BDPTVertex & lastMidPoint = path.v[li];
            optix::float3 lightLine = lastMidPoint.position - light.position;
            optix::float3 lightDirection = normalize(lightLine);
            path.v[li].pdf *= abs(dot(lightDirection, light.normal)) / M_PI;
        }
        else
        {
            {//brdf cos项，眼

                BDPTVertex &midPoint = path.v[i - 1];
                BDPTVertex &lastPoint = path.v[i - 2];
                BDPTVertex &nextPoint = path.v[i];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[i].epdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
            {//brdf cos项，光

                BDPTVertex &midPoint = path.v[li + 1];
                BDPTVertex &lastPoint = path.v[li + 2];
                BDPTVertex &nextPoint = path.v[li];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[li].pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
        }
        path.v[i].epdf *= path.v[i - 1].epdf;
        path.v[li].pdf *= path.v[li + 1].pdf;
    }

    float pdf = 0.0f;
    for (int i = 1; i < path.size - 1; i++)
    {
        pdf += path.v[i].epdf * path.v[i + 1].pdf * M * M * bufferWeight;
    }
    pdf += path.v[path.size - 1].epdf * M;
    optix::float3 ans = contri / pdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
    {
        return make_float3(0.0f);
    }
    return ans;
}

RT_FUNCTION optix::float3 n_pow_evalPath(BDPTVertexStack &path, int maxLength, int eyePathLen, float bufferWeight)
{
    optix::float3 contri;
    contri = contriCompute(path);
    path.v[0].epdf = 1.0f;
    for (int i = 1; i < path.size; i++)
    {
        int li = path.size - i - 1;
        if (i > maxLength)
        {
            path.v[i].epdf = 0.0f;
            path.v[li].pdf = 0.0f;
            continue;
        }
        path.v[i].epdf = M;
        path.v[li].pdf = M;

        if (i >= RR_BEGIN_DEPTH)
        {
            path.v[i].epdf *= RR_RATE;
            path.v[li].pdf *= RR_RATE;
        }
        {//距离与投影导致的几何项
            BDPTVertex &midPoint = path.v[i];
            BDPTVertex &lastPoint = path.v[i - 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[i].epdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        {//距离与投影导致的几何项(光)
            BDPTVertex &midPoint = path.v[li];
            BDPTVertex &lastPoint = path.v[li + 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[li].pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        if (i == 1)
        {
            BDPTVertex &light = path.v[li + 1];
            BDPTVertex & lastMidPoint = path.v[li];
            optix::float3 lightLine = lastMidPoint.position - light.position;
            optix::float3 lightDirection = normalize(lightLine);
            path.v[li].pdf *= abs(dot(lightDirection, light.normal)) / M_PI;
        }
        else
        {
            {//brdf cos项，眼

                BDPTVertex &midPoint = path.v[i - 1];
                BDPTVertex &lastPoint = path.v[i - 2];
                BDPTVertex &nextPoint = path.v[i];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[i].epdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
            {//brdf cos项，光

                BDPTVertex &midPoint = path.v[li + 1];
                BDPTVertex &lastPoint = path.v[li + 2];
                BDPTVertex &nextPoint = path.v[li];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[li].pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
        }
        path.v[i].epdf *= path.v[i - 1].epdf;
        path.v[li].pdf *= path.v[li + 1].pdf;
    }

    float pdf = 0.0f;
    float currentPdf;
    float currentWeight = 0.0f;
    float weight = 0.0f;
    float powRate = POWER_RATE;
    if (eyePathLen == path.size)
    {
        currentPdf = path.v[path.size - 1].epdf * M;
        currentWeight = currentPdf;
    }
#ifdef INDIRECT_ONLY
    else if (eyePathLen == path.size - 1)
    {
        currentPdf = path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * M * M * direct_eval_rate;
        currentWeight = currentPdf;
    }
#endif // INDIRECT_ONLY
    else
    {
        currentPdf = path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * M * M * bufferWeight;
        currentWeight = currentPdf;
    }


    for (int i = 1; i < path.size - 1; i++)
    {
#ifdef INDIRECT_ONLY
        if (i == path.size - 2)
        {
            weight += pow(path.v[i].epdf * path.v[i + 1].pdf * M * M * direct_eval_rate, powRate);
            continue;
        }
#endif
        weight += pow(path.v[i].epdf * path.v[i + 1].pdf * M * M * bufferWeight, powRate);
    }
    weight += pow(path.v[path.size - 1].epdf * M, powRate);
    optix::float3 ans = contri * (currentWeight / weight) / currentPdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z) || ans.x > DISCARD_VALUE || ans.y > DISCARD_VALUE || ans.z > DISCARD_VALUE)
    {
        return make_float3(0.0f);
    }
    return ans;
}


RT_FUNCTION optix::float3 n_pcpt_evalPath(BDPTVertexStack &path, int maxLength, int eyePathLen, float selectRate)
{
    optix::float3 contri;
    contri = contriCompute(path);
    path.v[0].epdf = 1.0f;
    for (int i = 1; i < path.size - 1; i++)
    {
        int li = path.size - i - 1;
        if (i > maxLength)
        {
            path.v[i].epdf = 0.0f;
            path.v[li].pdf = 0.0f;
            continue;
        }
        path.v[i].epdf = M;
        path.v[li].pdf = M;

        if (i >= RR_BEGIN_DEPTH)
        {
            path.v[i].epdf *= RR_RATE;
            path.v[li].pdf *= RR_RATE;
        }
        {//距离与投影导致的几何项
            BDPTVertex &midPoint = path.v[i];
            BDPTVertex &lastPoint = path.v[i - 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[i].epdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        {//距离与投影导致的几何项(光)
            BDPTVertex &midPoint = path.v[li];
            BDPTVertex &lastPoint = path.v[li + 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[li].pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        if (i == 1)
        {
            BDPTVertex &light = path.v[li + 1];
            BDPTVertex & lastMidPoint = path.v[li];
            optix::float3 lightLine = lastMidPoint.position - light.position;
            optix::float3 lightDirection = normalize(lightLine);
            path.v[li].pdf *= abs(dot(lightDirection, light.normal)) / M_PI;
        }
        else
        {
            {//brdf cos项，眼

                BDPTVertex &midPoint = path.v[i - 1];
                BDPTVertex &lastPoint = path.v[i - 2];
                BDPTVertex &nextPoint = path.v[i];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[i].epdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
            {//brdf cos项，光

                BDPTVertex &midPoint = path.v[li + 1];
                BDPTVertex &lastPoint = path.v[li + 2];
                BDPTVertex &nextPoint = path.v[li];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[li].pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
        }
        path.v[i].epdf *= path.v[i - 1].epdf;
        path.v[li].pdf *= path.v[li + 1].pdf;
    }

    float pdf = 0.0f;
    float currentPdf = path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * M * M;
    for (int i = 1; i < path.size - 1; i++)
    {
        pdf += pow(path.v[i].epdf * path.v[i + 1].pdf * M * M / currentPdf, 2);
    }
    optix::float3 ans = contri / pdf / currentPdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z))
    {
        return make_float3(0.0f);
    }
    return ans;
}


#ifdef ZONE_SAMPLER_OPTIMAL

rtBuffer<ZoneVertexSampler, 1>    zone_vertex_sampler_buffer;
RT_FUNCTION int get_vertex_from_zone(MeshLightDirector &a, int target_zone_id, float random, float &pdf)
{
    //return a.v[optix::clamp(static_cast<int>(floorf(random * a.size)), 0, a.size - 1)];
    switch (a.sample_direct[target_zone_id])
    {
    case -1:
        pdf = 1.0f;
        return a.v[optix::clamp(static_cast<int>(floorf(random * a.size)), 0, a.size - 1)];
    default:
        ZoneVertexSampler &b = zone_vertex_sampler_buffer[a.sample_direct[target_zone_id]];
        int size = a.size;
        float max_weight = b.m[size - 1];
        float index = random * max_weight;
        int mid = a.size / 2 - 1, l = 0, r = a.size;
        while (r - l > 1)
        {
            if (index < b.m[mid])
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }
        pdf = b.r[l] / b.m[a.size - 1] * a.size;
        return a.v[l];
        break;
    }
}
//a:light zone ；c：eye zone；target zone id：eye zone id ； v:light vertex; last_dir：light vertex的上个方向
RT_FUNCTION float get_vertex_from_zone_pdf(MeshLightDirector &a, MeshLightDirector &c, int target_zone_id, BDPTVertex& v, float3 last_dir)
{
    float defalut_weight = 100.f;
    //return a.pdf;
    switch (a.sample_direct[target_zone_id])
    {
    case -1:
        return a.pdf;
    default:
        ZoneVertexSampler &b = zone_vertex_sampler_buffer[a.sample_direct[target_zone_id]];
        float3 o_dir = normalize(c.weight_position - v.position);
        if (dot(o_dir, v.normal) < 0.0f)
        {
            return defalut_weight / (b.m[a.size - 1] / a.size) * a.pdf;
        }
        float3 w3 = v.emission * DisneyEval(v.material, v.normal, last_dir, o_dir) * dot(v.normal, o_dir);
        return (w3.x + w3.y + w3.z) / (b.m[a.size - 1] / a.size) * a.pdf;
        break;
    }
}
#endif // ZONE_SAMPLER_OPTIMAL

RT_FUNCTION optix::float3 mgpt_pow_evalPath(BDPTVertexStack &path, Mesh2MeshDirector *M2, MeshLightDirector *p, int maxLength, int eyePathLen, float bufferWeight)
{
    float tmps;
    if (path.size != eyePathLen)
        tmps = path.v[eyePathLen].emission.x;
    float powRate = POWER_RATE;
    optix::float3 contri;
    contri = contriCompute(path);

    path.v[0].epdf = 1.0f;
    for (int i = 1; i < path.size; i++)
    {
        int li = path.size - i - 1;
        if (i > maxLength)
        {
            path.v[i].epdf = 0.0f;
            path.v[li].pdf = 0.0f;
#ifdef ZONE_SAMPLER_OPTIMAL
            path.v[li].emission = make_float3(0.0f);
#endif // ZONE_SAMPLER_OPTIMAL
            continue;
        }
        path.v[i].epdf = M;
        path.v[li].pdf = M;

        if (i >= RR_BEGIN_DEPTH)
        {
            path.v[i].epdf *= RR_RATE;
            path.v[li].pdf *= RR_RATE;

#ifdef ZONE_SAMPLER_OPTIMAL
            path.v[li].emission = make_float3(1.0f / RR_RATE);
#endif // ZONE_SAMPLER_OPTIMAL
        }
        {//距离与投影导致的几何项
            BDPTVertex &midPoint = path.v[i];
            BDPTVertex &lastPoint = path.v[i - 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[i].epdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        {//距离与投影导致的几何项(光)
            BDPTVertex &midPoint = path.v[li];
            BDPTVertex &lastPoint = path.v[li + 1];
            optix::float3 line = midPoint.position - lastPoint.position;
            optix::float3 lineDirection = normalize(line);
            path.v[li].pdf *= 1.0f / dot(line, line) * abs(dot(midPoint.normal, lineDirection));
        }
        if (i == 1)
        {
            BDPTVertex &light = path.v[li + 1];
            BDPTVertex & lastMidPoint = path.v[li];
            optix::float3 lightLine = lastMidPoint.position - light.position;
            optix::float3 lightDirection = normalize(lightLine);
            path.v[li].pdf *= abs(dot(lightDirection, light.normal)) / M_PI;
        }
        else
        {
            {//brdf cos项，眼

                BDPTVertex &midPoint = path.v[i - 1];
                BDPTVertex &lastPoint = path.v[i - 2];
                BDPTVertex &nextPoint = path.v[i];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
                path.v[i].epdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
            }
            {//brdf cos项，光

                BDPTVertex &midPoint = path.v[li + 1];
                BDPTVertex &lastPoint = path.v[li + 2];
                BDPTVertex &nextPoint = path.v[li];
                optix::float3 lastDirection = normalize(lastPoint.position - midPoint.position);
                optix::float3 nextDirection = normalize(nextPoint.position - midPoint.position);
#ifdef ZONE_SAMPLER_OPTIMAL
                float d_pdf = DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
                path.v[li].pdf *= d_pdf;
                path.v[li].emission *= DisneyEval(midPoint.material, midPoint.normal, lastDirection, nextDirection) * dot(midPoint.normal, nextDirection)
                    / d_pdf * M * path.v[li + 1].emission;
#else
                path.v[li].pdf *= DisneyPdf(midPoint.material, midPoint.normal, lastDirection, nextDirection, midPoint.position);
#endif // ZONE_SAMPLER_OPTIMAL

            }
        }
        path.v[i].epdf *= path.v[i - 1].epdf;
        path.v[li].pdf *= path.v[li + 1].pdf;
    }

    float weight = 0.0f;
    float currentWeight = 0.0f;
    float currentPdf = 0.0f;
    float virtualWeight = 0.0f;

    int M_path = light_path_count;
    for (int i = 1; i < path.size - 1; i++)
    {
#ifdef INDIRECT_ONLY
        if (i == path.size - 2)
        {
            weight += pow(path.v[i].epdf * path.v[i + 1].pdf * M * M * direct_eval_rate / BUFFER_WEIGHT, powRate);
            continue;
        }
#endif
        BDPTVertex &eyeEndVertex = path.v[i];
        BDPTVertex &lightEndVertex = path.v[i + 1];
        int pIdx = eyeEndVertex.primIdx;
        int LPidx = lightEndVertex.primIdx;
        //float mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * p[LPidx].pdf;
        float Lweight = lightEndVertex.emission.x + lightEndVertex.emission.y + lightEndVertex.emission.z;
        float mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * p[LPidx].pdf;
#ifdef ZONE_SAMPLER_OPTIMAL
        float3 light_last_dir;
        if (i + 2 == path.size)
        {
            light_last_dir = make_float3(1.0f);
        }
        else
        {
            light_last_dir = normalize(path.v[i + 2].position - lightEndVertex.position);
        }

        mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * get_vertex_from_zone_pdf(p[LPidx], p[pIdx], pIdx, lightEndVertex, light_last_dir);
#endif

        if (p[LPidx].size == 0 || M2[pIdx].sum == 0)
            mg_rate = 0.0f;

        weight += pow(path.v[i].epdf * path.v[i + 1].pdf * M * M * mg_rate * bufferWeight, powRate);
        {
            float tmpWeight = 1.0f / (path.v[i].epdf * path.v[i + 1].pdf * M_path) + (1 - 1.0f / M_path) * 1.0f / mg_rate;
            tmpWeight = 1.0f / tmpWeight;
            tmpWeight = pow(tmpWeight, powRate);
            tmpWeight = mg_rate;
            //weight += tmpWeight;
        }
        virtualWeight += pow(path.v[i].epdf * path.v[i + 1].pdf * M * M, powRate);
    }
    float uniWeight = pow(path.v[path.size - 1].epdf * M / BUFFER_WEIGHT, powRate);
    float virtualUniWeight = pow(path.v[path.size - 1].epdf * M / BUFFER_WEIGHT, powRate);
    weight += uniWeight;

    if (eyePathLen == path.size)
    {
        currentPdf = path.v[path.size - 1].epdf * M;
        currentWeight = uniWeight;
        //currentWeight = pow(currentPdf / BUFFER_WEIGHT, powRate) / virtualWeight;
        //currentWeight =  virtualUniWeight;
        //weight = virtualWeight + virtualUniWeight;
    }
#ifdef INDIRECT_ONLY
    else if (eyePathLen == path.size - 1)
    {
        currentPdf = path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * M * M * direct_eval_rate;
        currentWeight = pow(currentPdf / BUFFER_WEIGHT, powRate);
    }
#endif // INDIRECT_ONLY
    else
    {
        BDPTVertex &eyeEndVertex = path.v[eyePathLen - 1];
        BDPTVertex &lightEndVertex = path.v[eyePathLen];
        int pIdx = eyeEndVertex.primIdx;
        int LPidx = lightEndVertex.primIdx;
        float mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * p[LPidx].pdf;
#ifdef ZONE_SAMPLER_OPTIMAL
        float3 light_last_dir;
        if (eyePathLen + 1 == path.size)
        {
            light_last_dir = make_float3(1.0f);
        }
        else
        {
            light_last_dir = normalize(path.v[eyePathLen + 1].position - lightEndVertex.position);
        }
        mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum * get_vertex_from_zone_pdf(p[LPidx], p[pIdx], pIdx, lightEndVertex, light_last_dir);
        if (p[LPidx].size == 0 || M2[pIdx].sum == 0)
            mg_rate = 0.0f;
        //mg_rate = float(M2[pIdx].r[LPidx]) / M2[pIdx].sum* p[LPidx].pdf
        //    * (lightEndVertex.emission.x / zone_vertex_sampler_buffer[p[LPidx].sample_direct[pIdx]].m[p[LPidx].size - 1] 
        //        / (1.0f / p[LPidx].size));
#endif
        currentWeight = pow(path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * M * M * mg_rate * bufferWeight, powRate);
        currentPdf = path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * mg_rate * M * M * bufferWeight;
        {
            float tmpWeight = 1.0f / (path.v[eyePathLen - 1].epdf * path.v[eyePathLen].pdf * M_path) + (1 - 1.0f / M_path) * 1.0f / mg_rate;
            tmpWeight = 1.0f / tmpWeight;
            tmpWeight = pow(tmpWeight, powRate);
            tmpWeight = mg_rate;
            //currentWeight = tmpWeight;
            //weight *= (virtualWeight + virtualUniWeight) / virtualWeight;
        }
    }
    optix::float3 ans = contri * (currentWeight / weight) / currentPdf;
    if (isnan(ans.x) || isnan(ans.y) || isnan(ans.z) || ans.x > DISCARD_VALUE || ans.y > DISCARD_VALUE || ans.z > DISCARD_VALUE)
    {
        return make_float3(0.0f);
    }
    return ans;
}

#endif // TOBEREWRITE


#endif