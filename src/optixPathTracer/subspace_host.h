#ifndef SUBSPACE_HOST
#define SUBSPACE_HOST
#include "subspace_common.h"
#include"BDPT_STRUCT.h" 
#include<random>
#include<vector>

using namespace optix;
struct subspaceSampler_host:subspaceSampler
{
    inline void add_vertex(BDPTVertex& a)
    {
        size++;
        vertex_sum++;
        float3 flux = a.flux / a.pdf;
        float weight = flux.x + flux.y + flux.z;
        Q = lerp(Q, weight, 1.0 / vertex_sum);
        energy_sum += weight;
    }
};
struct SubspapceHost
{ 
    Buffer LVC_buffer;
    Buffer oldLVC_buffer;
    Buffer raw_buffer;
    Buffer jump_buffer;
    Buffer sampler_buffer;
    RTsize LVCsize;
    void subspace_sampler_init(Context& context)
    {
#ifdef LIGHTVERTEX_REUSE 
        subspace_sampler_init_LVC_reuse(context);
        return;
#endif
        LVC_buffer = context["LVC"]->getBuffer();
        raw_buffer = context["raw_LVC"]->getBuffer();
        LVC_buffer->getSize(LVCsize);
        jump_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, LVCsize);
        jump_buffer->setElementSize(sizeof(jumpUnit));
        sampler_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, SUBSPACE_NUM);
        sampler_buffer->setElementSize(sizeof(subspaceSampler));

        context["vertex_sampler"]->setBuffer(sampler_buffer);
        context["jump_buffer"]->setBuffer(jump_buffer);
        context["subspace_LVC"]->setBuffer(LVC_buffer);
        return;
    }

    void subspace_sampler_init_LVC_reuse(Context& context)
    {
        LVC_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT,RT_FORMAT_USER,SUBSPACE_NUM * MAX_LIGHT_VERTEX_PER_TRIANGLE);
        LVC_buffer->setElementSize(sizeof(BDPTVertex));
        raw_buffer = context["raw_LVC"]->getBuffer();
        oldLVC_buffer = context["LVC"]->getBuffer();
        LVC_buffer->getSize(LVCsize);
        jump_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, LVCsize);
        jump_buffer->setElementSize(sizeof(jumpUnit));
        sampler_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, SUBSPACE_NUM);
        sampler_buffer->setElementSize(sizeof(subspaceSampler));

        context["vertex_sampler"]->setBuffer(sampler_buffer);
        context["jump_buffer"]->setBuffer(jump_buffer);
        context["subspace_LVC"]->setBuffer(LVC_buffer);
        return;
    }

    void buildSubspace_LVC_reuse(Context& context)
    {
        static int vertex_count_accm = 0;
        static int LVC_vertex_count[SUBSPACE_NUM] = { 0 };
        auto lvc_p = reinterpret_cast<BDPTVertex*>(LVC_buffer->map());
        auto old_lvc_p = reinterpret_cast<BDPTVertex*>(oldLVC_buffer->map());
        auto jump_p = reinterpret_cast<jumpUnit*>(jump_buffer->map());
        auto sampler_p = reinterpret_cast<subspaceSampler_host*>(sampler_buffer->map());

        {//build init
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                sampler_p[i].size = 0;
                sampler_p[i].base = 0;
                sampler_p[i].energy_sum = 0;
            }
        }

        int vertex_count = context["light_vertex_count"]->getInt();
        vertex_count_accm += vertex_count;
        for (int i = 0; i < vertex_count; i++)
        {
            int zid = old_lvc_p[i].zoneId;
            int lvc_id = zid * MAX_LIGHT_VERTEX_PER_TRIANGLE + (LVC_vertex_count[zid] % MAX_LIGHT_VERTEX_PER_TRIANGLE);
            lvc_p[lvc_id] = old_lvc_p[i];
            LVC_vertex_count[zid]++;
        }
        for (int i = 0; i < LVCsize; i++)
        {
            auto& v = lvc_p[i];
            int zid = i / MAX_LIGHT_VERTEX_PER_TRIANGLE;
            if ((i % MAX_LIGHT_VERTEX_PER_TRIANGLE) >= LVC_vertex_count[zid])continue;
            sampler_p[v.zoneId].add_vertex(v);
        }

        for (int i = 1; i < SUBSPACE_NUM; i++)
        {
            sampler_p[i].base = sampler_p[i - 1].base + sampler_p[i - 1].size;
            sampler_p[i - 1].size = 0;
        }
        sampler_p[SUBSPACE_NUM - 1].size = 0;
        for (int i = 0; i < LVCsize; i++)
        {
            auto& v = lvc_p[i];
            int zid = i / MAX_LIGHT_VERTEX_PER_TRIANGLE;
            if ((i % MAX_LIGHT_VERTEX_PER_TRIANGLE) >= LVC_vertex_count[zid])continue;

            auto& s = sampler_p[v.zoneId];
            auto& ju = jump_p[s.base + s.size];
            ju.p = i;
            float3 flux = v.flux / v.pdf;
            float weight = flux.x + flux.y + flux.z;
            ju.pmf = weight / s.energy_sum;

            s.size++;
        }

        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            int base = sampler_p[i].base;
            for (int j = 0; j < sampler_p[i].size; j++)
            {
                int id = base + j;
                auto& ju = jump_p[id];
                ju.cmf = ju.pmf;
                if (j >= 1)
                {
                    auto& ju_last = jump_p[id - 1];
                    ju.cmf += ju_last.cmf;
                }
                ju.pmf *= float(min(LVC_vertex_count[i], MAX_LIGHT_VERTEX_PER_TRIANGLE)) / float(vertex_count)
                    * (float(vertex_count_accm) / float(LVC_vertex_count[i]));
                 
            }
        } 
        LVC_buffer->unmap();
        jump_buffer->unmap();
        sampler_buffer->unmap();
        oldLVC_buffer->unmap();
    }


    void buildSubspace_M3reset(Context& context)
    {
        auto lvc_p = reinterpret_cast<BDPTVertex*>(LVC_buffer->map());
        auto jump_p = reinterpret_cast<jumpUnit*>(jump_buffer->map());
        auto sampler_p = reinterpret_cast<subspaceSampler_host*>(sampler_buffer->map());

        {//build init
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                sampler_p[i].size = 0;
                sampler_p[i].base = 0;
                sampler_p[i].energy_sum = 0;
            }
        }

        int vertex_count = context["light_vertex_count"]->getInt();
        for (int i = 0; i < vertex_count; i++)
        {
            auto& v = lvc_p[i];
            sampler_p[v.zoneId].add_vertex(v);
        }

        for (int i = 1; i < SUBSPACE_NUM; i++)
        {
            sampler_p[i].base = sampler_p[i - 1].base + sampler_p[i - 1].size;
            sampler_p[i - 1].size = 0;
        }
        sampler_p[SUBSPACE_NUM - 1].size = 0;
        for (int i = 0; i < vertex_count; i++)
        {
            auto& v = lvc_p[i];
            auto& s = sampler_p[v.zoneId];
            auto& ju = jump_p[s.base + s.size];
            ju.p = i;
            float3 flux = v.flux / v.pdf;
            float weight = flux.x + flux.y + flux.z;
            ju.pmf = weight / s.energy_sum;

            s.size++;
        }

        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            int base = sampler_p[i].base;
            for (int j = 0; j < sampler_p[i].size; j++)
            {
                int id = base + j;
                auto& ju = jump_p[id];
                ju.cmf = ju.pmf;
                if (j >= 1)
                {
                    auto& ju_last = jump_p[id - 1];
                    ju.cmf += ju_last.cmf;
                }
            }
        }
        if (true)
        {
            ZoneMatrix* M3_buffer = reinterpret_cast<ZoneMatrix*>(context["M3_buffer"]->getBuffer()->map());
            ZoneMatrix* M2_buffer = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                for (int j = 0; j < SUBSPACE_NUM; j++)
                {
                    if (sampler_p[j].size == 0)
                    {
                        M3_buffer[i].r[j] = 0;
                    }
                    else
                    {
                        M3_buffer[i].r[j] = M2_buffer[i].r[j];
                    }
                }
                M3_buffer[i].validation();
            }
            context["M3_buffer"]->getBuffer()->unmap();
            context["M2_buffer"]->getBuffer()->unmap();

        }
        LVC_buffer->unmap();
        jump_buffer->unmap();
        sampler_buffer->unmap();
    }
    void buildSubspace(Context& context)
    {
#ifdef LIGHTVERTEX_REUSE  
        buildSubspace_LVC_reuse(context);
        return;
#endif
        auto lvc_p = reinterpret_cast<BDPTVertex*>(LVC_buffer->map());
        auto jump_p = reinterpret_cast<jumpUnit*>(jump_buffer->map());
        auto sampler_p = reinterpret_cast<subspaceSampler_host*>(sampler_buffer->map());
        
        {//build init
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                sampler_p[i].size = 0;
                sampler_p[i].base = 0;
                sampler_p[i].energy_sum = 0;
            }
        }
        
        int vertex_count = context["light_vertex_count"]->getInt();
        for (int i = 0; i < vertex_count; i++)
        {
            auto& v = lvc_p[i];
            sampler_p[v.zoneId].add_vertex(v);
        }
         
        for (int i = 1; i < SUBSPACE_NUM; i++)
        {
            sampler_p[i].base = sampler_p[i - 1].base + sampler_p[i - 1].size;
            sampler_p[i-1].size = 0;
        } 
        sampler_p[SUBSPACE_NUM - 1].size = 0;
        for (int i = 0; i < vertex_count; i++)
        {
            auto& v = lvc_p[i];
            auto& s = sampler_p[v.zoneId];
            auto &ju = jump_p[s.base + s.size];
            ju.p = i;
            float3 flux = v.flux / v.pdf;
            float weight = flux.x + flux.y + flux.z;
            ju.pmf = weight / s.energy_sum;

            s.size++;
        }

        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            int base = sampler_p[i].base;
            for (int j = 0; j < sampler_p[i].size; j++)
            {
                int id = base + j;
                auto& ju = jump_p[id];
                ju.cmf = ju.pmf;
                if (j >= 1)
                {
                    auto& ju_last = jump_p[id - 1];
                    ju.cmf += ju_last.cmf;
                }
            }
        } 
        LVC_buffer->unmap();
        jump_buffer->unmap();
        sampler_buffer->unmap();
    }
};
SubspapceHost subspaces_api;
#endif
