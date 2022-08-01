#ifndef BDENV_HOST
#define BDENV_HOST
#include "BDenv_common.h"
#include"BDPT_STRUCT.h" 
#include<random>
#include<vector>
 
struct envInfo_host :envInfoBase
{
    vector<int> surroundsIndex(int index)
    {
        int2 coord = index2coord(index);
        vector<int2> s_coord;
        for (int dx = -2; dx <= 2; dx++)
        {
            for (int dy = -2; dy <= 2; dy++)
            {
                if(abs(dx) + abs(dy)<=2)
                    s_coord.push_back(coord + make_int2(dx, dy));
            }
        } 
        vector<int> ans;
        for (int i = 0; i < s_coord.size(); i++)
        {
            if (s_coord[i].x >= 0 && s_coord[i].y >= 0 && s_coord[i].x < width && s_coord[i].y < height)
            {
                ans.push_back(coord2index(s_coord[i]));
            }
        }
        return ans;
    }
    void setSamplingBuffer(Context& context)
    {
        float uniform_rate = 0.25;
        float uniform_pdf = 1.0 / size;
        auto buffer = context["envmap_buffer"]->getBuffer();
        auto p = reinterpret_cast<float4*>(buffer->map());
        auto sampling_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, size + 1);
        auto p2 = reinterpret_cast<float*>(sampling_buffer->map());
        for (int i = 0; i < size; i++)
        { 
            auto s_index = surroundsIndex(i);
            p2[i] = luminance(make_float3(p[i]));
            for (auto ii = s_index.begin(); ii != s_index.end(); ii++)
            {
                p2[i] += luminance(make_float3(p[*ii])) / s_index.size();
            }
            if (i >= 1)
            {
                p2[i] += p2[i - 1];
            }
        }
        float sum = p2[size - 1];
        for (int i = 0; i < size; i++)
        {
            p2[i] /= sum;
            p2[i] = lerp(p2[i], uniform_pdf * (i+1), uniform_rate);
        }
        p2[size] = 1;
        buffer->unmap();
        sampling_buffer->unmap();
        context["env_sampling_buffer"]->setBuffer(sampling_buffer);
    };
    void setlabelBuffer(Context& context) 
    {
        int label_num = 100;
        int resol = 9;
        float limit = 0.06;
        auto label_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, size + 1);
        auto sampling_buffer = context["env_sampling_buffer"]->getBuffer();
        auto p2 = reinterpret_cast<float*>(sampling_buffer->map());
        auto p3 = reinterpret_cast<int*>(label_buffer->map());
        
        float sum = 0;
        for (int i = 0; i < size; i++)
        {
            p3[i] = p2[i] * label_num;
            int2 coord = index2coord(i);
            float2 uv = coord2uv(coord);
            int index2 = int(uv.x * resol) * resol + int(uv.y * resol);
            p3[i] = index2;
            if (p3[i] >= label_num)
            {
                p3[i] = label_num - 1;
            }
        } 
        int base = resol * resol;
        for (int i = 0; i < size; i++)
        {
            float w = p2[i];
            if (i > 0)
            {
                w -= p2[i - 1];
            }
            if (w > limit)
            {
                p3[i] = base;
                base++;
            } 
            if (base >= label_num)
            {
                break;
            }
        }
        sampling_buffer->unmap();
        label_buffer->unmap();
        context["env_label_buffer"]->setBuffer(label_buffer);
    };

    void setDeviceBuffer(Context& context)
    {
        auto buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        buffer->setElementSize(sizeof(envInfoBase));
        auto p = reinterpret_cast<envInfoBase*>(buffer->map());
        memcpy_s(p, sizeof(envInfoBase), this, sizeof(envInfoBase)); 
        buffer->unmap();
        context["env_info"]->setBuffer(buffer);
    }
    void hostSetting(Context& context)
    {
        auto sampler = context["envmap"]->getTextureSampler();
        auto env_buffer = sampler->getBuffer();
        context["envmap_buffer"]->setBuffer(env_buffer);
        RTsize w, h;
        env_buffer->getSize(w, h);
        width = w;
        height = h;
        size = width * height;
         
        setSamplingBuffer(context);
        setlabelBuffer(context);
        setDeviceBuffer(context);
    }

    void add_direction_light(Context& context, float3 dir, float3 color)
    {
        auto buffer = context["envmap_buffer"]->getBuffer();
        auto p = reinterpret_cast<float4*>(buffer->map());
        int index = coord2index(uv2coord(dir2uv(dir)));
        int2 coord = uv2coord(dir2uv(dir));
        printf("add direction %d %d\n", coord.x, coord.y);
        p[index] += make_float4(color * size / (4 * M_PI),1.0);
        buffer->unmap();
        setSamplingBuffer(context);
        setlabelBuffer(context);
        setDeviceBuffer(context);
    }
    void set_env_lum(Context& context, float factor)
    { 
        auto buffer = context["envmap_buffer"]->getBuffer();
        auto p = reinterpret_cast<float4*>(buffer->map());
        RTsize buffer_size;
        buffer->getSize(buffer_size);
        //factor = 10;
        for (int i = 0; i < buffer_size; i++)
        {
            p[i] *= factor;
        }

        printf("env_lum %f\n", factor);
        buffer->unmap();
    }
    void setAABB(Context& context, Aabb& aabb)
    {
        center = (aabb.m_max + aabb.m_min) / 2;
        r = length(aabb.m_max - aabb.m_min) / 2;
        setDeviceBuffer(context);
    }
};
envInfo_host env_info;

#endif