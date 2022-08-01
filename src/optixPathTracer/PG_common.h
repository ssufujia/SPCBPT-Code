#ifndef PG_COMMON
#define PG_COMMON 
#include"rt_function.h"
#include <optixu/optixu_math_namespace.h>
using namespace optix;

struct PG_training_mat
{
    float3 position;
    float2 uv;
    float2 uv_light;
    float lum;
    bool valid;
    bool light_source;
    int light_id;
    RT_FUNCTION PG_training_mat() :valid(false) {}
};

struct quad_tree_node
{
    float2 m_min;
    float2 m_max;
    float2 m_mid;
    int child[4];
    int quad_tree_id;
    int count;
    bool leaf;
    float lum;
    RT_FUNCTION quad_tree_node(float2 m_min, float2 m_max) :m_min(m_min), m_max(m_max), m_mid((m_min + m_max) / 2), leaf(true), lum(0) {}
    RT_FUNCTION int whichChild(float2 uv)
    {

        int base = 1;
        int index = 0;
        if (uv.x > m_mid.x)
        {
            index += base;
        }
        base *= 2;
        if (uv.y > m_mid.y)
        {
            index += base;
        }
        base *= 2;
        return index;
    }

    RT_FUNCTION int getChild(int index)
    {
        return child[index];
    }
    RT_FUNCTION int traverse(float2 uv)
    {
        return getChild(whichChild(uv));
    }
    RT_FUNCTION float area()
    {
        float2 d = m_max - m_min;
        return d.x * d.y;
    }
};

struct Spatio_tree_node
{
    float3 m_min;
    float3 m_max;
    float3 m_mid;
    bool leaf;
    int count;
    int child[8];
    int quad_tree_id;
    RT_FUNCTION Spatio_tree_node(float3 m_min, float3 m_max) :m_min(m_min), m_max(m_max), m_mid((m_min + m_max) / 2), leaf(true), count(0), quad_tree_id(0) {}
    RT_FUNCTION int whichChild(float3 pos)
    {
        int base = 1;
        int index = 0;
        if (pos.x > m_mid.x)
        {
            index += base;
        }
        base *= 2;
        if (pos.y > m_mid.y)
        {
            index += base;
        }
        base *= 2;
        if (pos.z > m_mid.z)
        {
            index += base;
        }
        base *= 2;
        return index;
    }
    RT_FUNCTION int getChild(int index)
    {
        return child[index];
    }
    RT_FUNCTION int traverse(float3 pos)
    {
        return getChild(whichChild(pos));
    }
};
#define GMM_CORE 3

#endif