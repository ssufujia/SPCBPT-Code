#ifndef PG_DEVICE
#define PG_DEVICE
#include"PG_common.h"
#include "random.h"

rtBuffer<Spatio_tree_node, 1> spatio_tree_nodes;
rtBuffer<Spatio_tree_node, 1> spatio_tree_nodes_light;
rtBuffer<quad_tree_node, 1> quad_tree_nodes;
rtBuffer<quad_tree_node, 1> quad_tree_nodes_light;
rtBuffer<quad_tree_node, 1> L_trees;
rtBuffer<quad_tree_node, 1> L_tree_dir;

rtDeclareVariable(int, pg_enable, , ) = { 0 };
rtDeclareVariable(float, epsilon_lum, , ) = { 0.001 };
rtDeclareVariable(int, pg_lightSide_enable, , ) = { 0 };
rtDeclareVariable(int, pg_lightSource_enable, , ) = { 0 };
rtBuffer<LightParameter> Lights;
struct PG_API
{
    RT_FUNCTION PG_API() {}
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
    RT_FUNCTION int quad_random_walk(int id, float lum)
    {
        if (quad_tree_nodes[id].leaf == false)
        { 
            float lum_header = quad_tree_nodes[id].lum;
            float lum_sum =
                quad_tree_nodes[quad_tree_nodes[id].child[0]].lum +
                quad_tree_nodes[quad_tree_nodes[id].child[1]].lum +
                quad_tree_nodes[quad_tree_nodes[id].child[2]].lum +
                quad_tree_nodes[quad_tree_nodes[id].child[3]].lum;
            for (int i = 0; i < 4; i++)
            {
                int c_id = quad_tree_nodes[id].child[i];
                if (lum < quad_tree_nodes[c_id].lum)
                {
                    //rtPrintf("%f %f %f\n",lum_header, lum_sum, lum);
                    
                    return c_id;
                }
                else
                {
                    lum -= quad_tree_nodes[c_id].lum;
                }    
            }
        }
        //rtPrintf("%f %f %f %f %f\n", quad_tree_nodes[id].lum, quad_tree_nodes[quad_tree_nodes[id].child[0]].lum,
        //    quad_tree_nodes[quad_tree_nodes[id].child[1]].lum, quad_tree_nodes[quad_tree_nodes[id].child[2]].lum,
        //    quad_tree_nodes[quad_tree_nodes[id].child[3]].lum);
        return id;
    }
    RT_FUNCTION int getStreeId(float3 position)
    {
        int c_id = 0;
        while (spatio_tree_nodes[c_id].leaf == false)
        {
            c_id = spatio_tree_nodes[c_id].traverse(position);
        }
        return c_id;
    }

    RT_FUNCTION float pdf(float3 position, float3 dir)
    {
        int c_id = getStreeId(position);
        int q_id = spatio_tree_nodes[c_id].quad_tree_id;
        if (quad_tree_nodes[q_id].lum < epsilon_lum )
        {
            return 1.0 / (4 * M_PI);
        }
        float lum_sum = quad_tree_nodes[q_id].lum;
        float2 uv = dir2uv(dir);
        while (quad_tree_nodes[q_id].leaf == false)
        {
            q_id = quad_tree_nodes[q_id].traverse(uv);
        }
        float pdf1 = quad_tree_nodes[q_id].lum / lum_sum;
        float area = quad_tree_nodes[q_id].area();
        //if(q_id == 2)
          //  rtPrintf("%d %f %f %f\n",q_id, pdf1, quad_tree_nodes[2].lum,lum_sum);
        return pdf1 / (area * 4 * M_PI);
    }
    RT_FUNCTION float3 sample(unsigned int& seed, float3 position)
    {
        int c_id = getStreeId(position);
        int q_id = spatio_tree_nodes[c_id].quad_tree_id;
        float lum_sum = quad_tree_nodes[q_id].lum;
        if (quad_tree_nodes[q_id].lum < epsilon_lum )
        {
            float t1 = rnd(seed);
            float t2 = rnd(seed);
            float2 uv = make_float2(t1, t2);
            return uv2dir(uv);
        }
        while (quad_tree_nodes[q_id].leaf == false)
        {
            float t_lum = rnd(seed) * quad_tree_nodes[q_id].lum;
            q_id = quad_random_walk(q_id, t_lum);
//            q_id = quad_tree_nodes[q_id].child[optix::clamp(static_cast<int>(floorf(rnd(seed) * 4)), 0, 3)];
        }
        float t1 = rnd(seed);
        float t2 = rnd(seed);
        float2 uv = make_float2(
            lerp(quad_tree_nodes[q_id].m_min.x, quad_tree_nodes[q_id].m_max.x, t1),
            lerp(quad_tree_nodes[q_id].m_min.y, quad_tree_nodes[q_id].m_max.y, t2));
        //float pdf2 = pdf(position, uv2dir(uv));
        //rtPrintf("%f %f %f \n", quad_tree_nodes[q_id].lum / lum_sum /(quad_tree_nodes[q_id].area() * 4 * M_PI),pdf2, quad_tree_nodes[q_id].area());
        return uv2dir(uv);
    }



    RT_FUNCTION int quad_random_walk_lightside(int id, float lum)
    {
        if (quad_tree_nodes_light[id].leaf == false)
        {
            float lum_header = quad_tree_nodes_light[id].lum;
            float lum_sum =
                quad_tree_nodes_light[quad_tree_nodes_light[id].child[0]].lum +
                quad_tree_nodes_light[quad_tree_nodes_light[id].child[1]].lum +
                quad_tree_nodes_light[quad_tree_nodes_light[id].child[2]].lum +
                quad_tree_nodes_light[quad_tree_nodes_light[id].child[3]].lum;
            for (int i = 0; i < 4; i++)
            {
                int c_id = quad_tree_nodes_light[id].child[i];
                if (lum < quad_tree_nodes_light[c_id].lum)
                {
                    //rtPrintf("%f %f %f\n",lum_header, lum_sum, lum);

                    return c_id;
                }
                else
                {
                    lum -= quad_tree_nodes_light[c_id].lum;
                }
            }
        }
        //rtPrintf("%f %f %f %f %f\n", quad_tree_nodes_light[id].lum, quad_tree_nodes_light[quad_tree_nodes_light[id].child[0]].lum,
        //    quad_tree_nodes_light[quad_tree_nodes_light[id].child[1]].lum, quad_tree_nodes_light[quad_tree_nodes_light[id].child[2]].lum,
        //    quad_tree_nodes_light[quad_tree_nodes_light[id].child[3]].lum);
        return id;
    }
    RT_FUNCTION int getStreeId_lightside(float3 position)
    {
        int c_id = 0;
        while (spatio_tree_nodes_light[c_id].leaf == false)
        {
            c_id = spatio_tree_nodes_light[c_id].traverse(position);
        }
        return c_id;
    }

    RT_FUNCTION float pdf_lightside(float3 position, float3 dir)
    {
        int c_id = getStreeId_lightside(position);
        int q_id = spatio_tree_nodes_light[c_id].quad_tree_id;
        if (quad_tree_nodes_light[q_id].lum < epsilon_lum)
        {
            return 1.0 / (4 * M_PI);
        }
        float lum_sum = quad_tree_nodes_light[q_id].lum;
        float2 uv = dir2uv(dir);
        while (quad_tree_nodes_light[q_id].leaf == false)
        {
            q_id = quad_tree_nodes_light[q_id].traverse(uv);
        }
        float pdf1 = quad_tree_nodes_light[q_id].lum / lum_sum;
        float area = quad_tree_nodes_light[q_id].area();
        //if(q_id == 2)
          //  rtPrintf("%d %f %f %f\n",q_id, pdf1, quad_tree_nodes_light[2].lum,lum_sum);
        return pdf1 / (area * 4 * M_PI);
    }
    RT_FUNCTION float3 sample_lightside(unsigned int& seed, float3 position)
    {
        int c_id = getStreeId_lightside(position);
        int q_id = spatio_tree_nodes_light[c_id].quad_tree_id;
        float lum_sum = quad_tree_nodes_light[q_id].lum;
        if (quad_tree_nodes_light[q_id].lum < epsilon_lum)
        {
            float t1 = rnd(seed);
            float t2 = rnd(seed);
            float2 uv = make_float2(t1, t2);
            return uv2dir(uv);
        }
        while (quad_tree_nodes_light[q_id].leaf == false)
        {
            float t_lum = rnd(seed) * quad_tree_nodes_light[q_id].lum;
            q_id = quad_random_walk_lightside(q_id, t_lum);
            //            q_id = quad_tree_nodes_light[q_id].child[optix::clamp(static_cast<int>(floorf(rnd(seed) * 4)), 0, 3)];
        }
        float t1 = rnd(seed);
        float t2 = rnd(seed);
        float2 uv = make_float2(
            lerp(quad_tree_nodes_light[q_id].m_min.x, quad_tree_nodes_light[q_id].m_max.x, t1),
            lerp(quad_tree_nodes_light[q_id].m_min.y, quad_tree_nodes_light[q_id].m_max.y, t2));
        //float pdf2 = pdf(position, uv2dir(uv));
        //rtPrintf("%f %f %f \n", quad_tree_nodes_light[q_id].lum / lum_sum /(quad_tree_nodes_light[q_id].area() * 4 * M_PI),pdf2, quad_tree_nodes_light[q_id].area());
        return uv2dir(uv);
    }

    RT_FUNCTION int quad_tree_traverse(quad_tree_node* p, float2 uv, int root_id)
    {
        int c_id = root_id;
        while (p[c_id].leaf == false)
        {
            c_id = p[c_id].traverse(uv);
        }
        return c_id;
    }
    RT_FUNCTION int quad_tree_random_walk(quad_tree_node* p, int id, float lum)
    {
        if (p[id].leaf == false)
        {
            float lum_header = quad_tree_nodes[id].lum;
            float lum_sum =
                p[p[id].child[0]].lum +
                p[p[id].child[1]].lum +
                p[p[id].child[2]].lum +
                p[p[id].child[3]].lum;


            for (int i = 0; i < 4; i++)
            {
                int c_id = p[id].child[i];
                if (lum < p[c_id].lum)
                { 
                    return c_id;
                }
                else
                {
                    lum -= p[c_id].lum;
                }
            }
        }
        rtPrintf("random walk error %f %f %f %f %f \n", p[p[id].child[0]].lum, p[p[id].child[1]].lum, p[p[id].child[2]].lum, p[p[id].child[3]].lum, quad_tree_nodes[id].lum);
        return p[id].child[0];
    }
    RT_FUNCTION float quad_tree_pdf(quad_tree_node* p, float2 uv, int id)
    {
        if (p[id].lum < epsilon_lum )
        { 
            return 1.0;
        }
        int leaf_id = quad_tree_traverse(p, uv, id);
        return p[leaf_id].lum / p[id].lum / p[leaf_id].area();
    }
    RT_FUNCTION float2 quad_sample(quad_tree_node* p, unsigned int& seed, int id)
    { 
        float lum_sum = p[id].lum;
        if (p[id].lum < epsilon_lum)
        {
            float t1 = rnd(seed);
            float t2 = rnd(seed);
            float2 uv = make_float2(t1, t2);
            return uv;
        }
        while (p[id].leaf == false)
        {
            float t_lum = rnd(seed) * p[id].lum;
            id = quad_tree_random_walk(p, id, t_lum);
            //            q_id = quad_tree_nodes_light[q_id].child[optix::clamp(static_cast<int>(floorf(rnd(seed) * 4)), 0, 3)];
        }
        float t1 = rnd(seed);
        float t2 = rnd(seed);
        float2 uv = make_float2(
            lerp(p[id].m_min.x, p[id].m_max.x, t1),
            lerp(p[id].m_min.y, p[id].m_max.y, t2));
        //float pdf2 = pdf(position, uv2dir(uv));
        //rtPrintf("%f %f %f \n", quad_tree_nodes_light[q_id].lum / lum_sum /(quad_tree_nodes_light[q_id].area() * 4 * M_PI),pdf2, quad_tree_nodes_light[q_id].area());
        return uv;
    }
    RT_FUNCTION float2 L_area_sample(unsigned int& seed, int id)
    {
        float2 uv = quad_sample(&L_trees[0], seed, id);
        return uv;
    }
    RT_FUNCTION float L_area_pdf(int id, float2 uv)
    {
        return quad_tree_pdf(&L_trees[0], uv, id);
    }

    RT_FUNCTION float3 L_dir_sample_0(unsigned int& seed, int id)
    {
        float2 uv = quad_sample(&L_tree_dir[0], seed, id);
        return uv2dir(uv);
    }

    RT_FUNCTION float3 L_dir_sample(unsigned int& seed, int light_id, float2 uv)
    {
        int L_id = quad_tree_traverse(&L_trees[0], uv, light_id);
        int LD_id = L_trees[L_id].quad_tree_id;
        return L_dir_sample_0(seed, LD_id); 
    }
    RT_FUNCTION float L_dir_pdf(int light_id, float2 uv, float3 dir)
    {
        int L_id = quad_tree_traverse(&L_trees[0], uv, light_id);
        int LD_id = L_trees[L_id].quad_tree_id;
        float2 dir_uv = dir2uv(dir);
        return quad_tree_pdf(&L_tree_dir[0], dir_uv, LD_id);
    }
}; 
rtDeclareVariable(PG_API, pg_api, , ) = { };


RT_FUNCTION float quadLightPdf_area(int light_id, float2 uv, float3 dir)
{
    float factor = 1.0;
    if ( pg_lightSource_enable)
    {
        factor = lerp(factor, pg_api.L_area_pdf(light_id, uv), PG_LIGHTSOURCE_RATE);
    }
    return 1.0 / Lights[light_id].area * factor;
}
RT_FUNCTION float quadLightPdf_dir(int light_id, float2 uv, float3 dir)
{
    float factor = abs(dot(dir, Lights[light_id].normal)) / M_PI;

    if (pg_lightSource_enable)
    {
        factor = lerp(factor, pg_api.L_dir_pdf(light_id, uv, dir) / (4 * M_PI), PG_LIGHTSOURCE_RATE);
    } 
    return factor;
}
RT_FUNCTION float quadLightPdf(int light_id, float2 uv, float3 dir)
{
    float pdf1 = quadLightPdf_area(light_id, uv, dir);
    float pdf2 = quadLightPdf_dir(light_id, uv, dir);
    return pdf1 * pdf2;
}

RT_FUNCTION float pg_quadPdfRate(BDPTVertex& v, float3 dir)
{
    int light_id = v.materialId;
    float pdf_origin = abs(dot(dir, v.normal)) / M_PI;
    float pdf_new = quadLightPdf_dir(light_id, v.uv, dir);
   // rtPrintf("%f %f %f %f\n", pdf_origin / pdf_new, dir.x, dir.y, dir.z);
    return pdf_new / pdf_origin ;
}

RT_FUNCTION float pg_quadAreaRate(int light_id, float2 uv, float3 dir)
{ 
    float pdf_origin = 1.0 / Lights[light_id].area;
    float pdf_new = quadLightPdf_area(light_id, uv, dir);
    //rtPrintf("%f\n", pdf_origin / pdf_new);
    return pdf_new / pdf_origin;
}
 
#endif // !PG_DEVICE
