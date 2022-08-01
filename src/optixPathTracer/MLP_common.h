#ifndef MLP_COMMON_H
#define MLP_COMMON_H

#include"rt_function.h"
#include <optixu/optixu_math_namespace.h>
#include"BDPTVertex.h"
using namespace optix;

struct feed_token
{
    float3 position;
    float pdf0;
    int grid_label;
    int dual_label;
    float f_square;
    float ratio;
};
struct BP_token
{
    float feature_0[16];
    float feature_1[16];
    float res[32];
    float pdf;
    float f_square;
    int2  index_id;

    RT_FUNCTION void clear()
    {
        for (int i = 0; i < 16; i++)
        { 
            feature_0[i] = 0; 
        }

        for (int i = 0; i < 16; i++)
        {
            feature_1[i] = 0;
        }

        for (int i = 0; i < 32; i++)
        {
            res[i] = 0;
        }
    }
};
namespace MLP
{

    struct nVertex
    {
        float3 position;
        float3 dir;
        float3 normal;
        float3 weight;//pdf for eye vertex, light contri for light vertex
        float3 color;
        float pdf;//light vertex only, for fast compute the overall sampled path pdf 
        float save_t;
        int last_id;//for cached light vertex
        int materialId;
        int label_id; 
        int depth;

        bool isBrdf;
        //bool isDirection;
        bool valid;
        __host__ RT_FUNCTION nVertex():valid(false){}
        __host__ __device__
            void setLightSourceFlag(bool dir_light = false)
        {
            materialId = dir_light ? -2 : -1;
        }
        __host__ __device__
            bool isLightSource()const
        {
            return materialId < 0;
        }
        __host__ __device__
            bool isDirLight()const
        {
            return materialId == -2;
        }

        __host__ __device__
            bool brdf_weight()const
        {
            return !isBrdf;
        }

        __host__ __device__
            bool isAreaLight()const
        {
            return materialId == -1;
        }
        __host__ RT_FUNCTION nVertex(const BDPTVertex& a, bool eye_side) :
            position(a.position), normal(a.normal), color(a.color), materialId(a.materialId), pdf(a.pdf), valid(true),label_id(a.zoneId),isBrdf(a.isBrdf),save_t(0),depth(a.depth)
            //isDirection(a.is_DIRECTION())
        {
            dir = a.depth == 0 ? make_float3(0.0) : normalize(a.lastPosition - a.position);
            weight = eye_side ? make_float3(pdf) : a.flux;
            if (eye_side == false && a.depth == 0)
            {
                if (a.type == QUAD) setLightSourceFlag(false);
                if (a.type == DIRECTION || a.type == ENV) setLightSourceFlag(true);
            }
        }
    }; 

    struct pathInfo_node
    {
        float3 A_position;
        float3 B_position;
        float3 A_dir_d;
        float3 A_normal_d;
        float3 B_normal_d;
        float3 B_dir_d;
        float peak_pdf;
        //about peak_pdf:
        // = a.pdf * b.contri when generated from tracing
        // = a.pdf * b.contri / Q_b after transform
        int path_id;
        int label_A;//empty until the inital tree is constructed
        int label_B;//empty until the inital tree is constructed
        bool valid;
        bool light_source;
        RT_FUNCTION pathInfo_node():valid(false){}
        RT_FUNCTION pathInfo_node(nVertex& a, nVertex& b):
            A_position(a.position),B_position(b.position),A_dir_d(a.dir),B_dir_d(b.dir),valid(true),light_source(b.isLightSource()),label_B(b.label_id)
        {
            A_normal_d = a.normal;
            B_normal_d = b.normal;
            A_dir_d = a.dir;
            B_dir_d = b.dir;
            peak_pdf = a.weight.x * ENERGY_WEIGHT(b.weight) * b.brdf_weight() * a.brdf_weight(); 
            set_eye_depth(a.depth);
        }
        RT_FUNCTION int get_eye_depth() { return label_A; }
        RT_FUNCTION void set_eye_depth(int depth) { label_A = depth; }
        __host__ RT_FUNCTION float3 B_normal()const
        {
            return B_normal_d;
        }
        __host__ RT_FUNCTION float3 A_normal()const
        {
            return A_normal_d;
        }

        __host__ RT_FUNCTION float3 B_dir()const
        {
            return B_dir_d;
        }
        __host__ RT_FUNCTION float3 A_dir()const
        {
            return A_dir_d;
        }
    };
    struct pathInfo_sample
    {
        float3 contri;
        float sample_pdf;
        float fix_pdf; 
        int begin_ind;
        int end_ind;
        int choice_id;
        int2 pixiv_id;
        bool valid;
    };

    struct data_buffer
    {
        template<typename T>
        struct slim_vector
        {
            T* v;
            int size; 
            __host__ __device__ __forceinline__
                T& operator[](int i)
            {
                return v[i];
            }
        };
        slim_vector<nVertex> LVC;
        slim_vector<nVertex> samples;
        slim_vector<float> cmfs;
        slim_vector<pathInfo_node> p_nodes;
        slim_vector<pathInfo_sample> p_samples;
        slim_vector<float> Q_weight;
        bool use_resample; 
        int sample_M; 
        int2 launch_size;
        int2 res_padding;
        int launch_seed;
        int construct_frame;
        int construct_size;
        __host__ __device__ __forceinline__ int get_sample_depth(int light_id)
        {
            if (samples[light_id].isLightSource())
                return 0;
            int d = 1;
            light_id = samples[light_id].last_id;

            while (LVC[light_id].isLightSource() == false)
            {
                light_id = LVC[light_id].last_id;
                d++;
            }
            return d;
        }
        __host__ __device__ __forceinline__ int light_sample(float random)
        { 
            float index = random;
            int mid = samples.size / 2 - 1, l = 0, r = samples.size;
            while (r - l > 1)
            {
                if (index < cmfs[mid])
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
        __host__ __device__ __forceinline__
            float light_select_pdf(int i)
        {
            //to be rewrite to a adaptive version
            if (!use_resample)
            { 
                return float(sample_M) / samples.size;
            }
            else
            { 
                if (Q_weight[i] == 0.0)return 0;
                return float(sample_M) * Q_weight[i];
            }
        }

        
    };
}
struct MLP_network
{
    float layer_0[60][16];
    float layer_1[16][16];
    float layer_2[16][32];
    float bias_0[16];
    float bias_1[16];
    float bias_2[32];
    RT_FUNCTION void update(MLP_network& a,float lr = 0.0001)
    {
        for (int i = 0; i < 60; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                layer_0[i][j] -= lr * a.layer_0[i][j];
            }
        }
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                layer_1[i][j] -= lr * a.layer_1[i][j];
            }
        }
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 32; j++)
            {
                layer_1[i][j] -= lr * a.layer_2[i][j];
            }
        }
    }
    RT_FUNCTION void clear()
    {
        for (int i = 0; i < 60; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                layer_0[i][j] = 0;
            } 
        }
        for (int i = 0; i < 16; i++)
        { 
            for (int j = 0; j < 16; j++)
            {
                layer_1[i][j] = 0;
            }
        }
        for (int i = 0; i < 16; i++)
        { 
            for (int j = 0; j < 32; j++)
            {
                layer_1[i][j] = 0;
            }
        }
        for (int i = 0; i < 32; i++)
        {
            bias_2[i] = 0; 
        }
        for (int i = 0; i < 16; i++)
        {
            bias_1[i] = 0;
            bias_0[i] = 0;
        }
    }
};
namespace HS_algorithm
{
    struct point
    {
        //virtual float d(point){}
    };

    struct gamma_point_2 :point
    {
        float x;
        __host__ RT_FUNCTION
            float d(const gamma_point_2& a)const
        {
            return (x - a.x) * (x - a.x);
        }

    };
    struct gamma_point :point
    {
        float3 position;
        float3 direction;
        float3 normal;
        float diag2;
        __host__ RT_FUNCTION
            float d(const gamma_point& a)const
        {
            float k = DIR_JUDGE;
            float3 diff = a.position - position;
            float d_a = dot(diff, diff);
            float diff_direction = dot(direction, a.direction);
            float diff_normal = dot(normal, a.normal);
            return d_a + diag2 * ((1 - diff_normal) + (1 - diff_direction) * k);
        }
        __host__ RT_FUNCTION
        float d(const gamma_point& a,float diag2)const
        {
            float k = DIR_JUDGE;
            float3 diff = a.position - position;
            float d_a = dot(diff, diff);
            float diff_direction = dot(direction, a.direction);
            float diff_normal = dot(normal, a.normal);
            return d_a + diag2 * ((1 - diff_normal) + (1 - diff_direction) * k);
        }
    };

}
#endif
