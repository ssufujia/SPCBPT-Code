#ifndef ZGC_DEVICE
#define ZGC_DEVICE 
#include "random.h"
#include "ZGC_common.h"
#include"classTree_device.h"
rtBuffer<int, 1>  slic_label_buffer;
rtBuffer<int, 1>  slic_unlabel_index_buffer;
rtBuffer<BDPTVertex, 1>  slic_cluster_center_buffer;
rtBuffer<triangleStruct, 1>  slic_tris;


rtBuffer<int, 1>					 triBase;
rtBuffer<int, 1>					 divLevel;
rtBuffer<int3, 1>					 zgc_gridInfo;
rtBuffer<triangleStruct, 1>          div_tris;
rtDeclareVariable(float, space_normal_item, , );
rtDeclareVariable(int, slic_cluster_num, , );
RT_FUNCTION float slic_diff(BDPTVertex& v,triangleStruct &t)
{
    float mat_diff,normal_diff,space_diff;

    float3 center = t.center();
    float3 dVector = center - v.position;
    space_diff = dot(dVector, dVector) / space_normal_item;
    mat_diff = v.materialId == t.objectId ? 0 : 0.1;
    normal_diff = -dot(v.normal, t.normal());
    return mat_diff + normal_diff + space_diff;
}

struct labelUnit
{
    float3 position;
    float3 normal;
    float3 dir;

    float2 uv;
    float2 tri_uv;
    int objectId;
    int tri_id;
    int type;
    bool light_side;
    RT_FUNCTION labelUnit(float3 position,float3 normal,float3 dir, float2 uv, float2 tri_uv, int objectId, int tri_id,bool light_side, int type = 0) :
        position(position), normal(normal), dir(dir), uv(uv), tri_uv(tri_uv), objectId(objectId), tri_id(tri_id),light_side(light_side), type(type) {}
    RT_FUNCTION labelUnit(float3 position, float3 normal, float3 dir, bool light_side) : position(position), normal(normal), dir(dir), light_side(light_side) {}
    RT_FUNCTION int FindDivTriNum(int divLevel, float2 beta_gamma)
    {
        int div_block = 0;
        int div_class = divLevel;
        float beta = beta_gamma.x;
        float gamma = beta_gamma.y;
        while (div_class != 0)
        {
            div_block = div_block << 1;
            div_class -= 1;
            float n_gamma = abs((beta + gamma - 1));
            float n_beta = abs((beta - gamma));
            if (beta < gamma)
            {
                div_block += 1;
                gamma = n_beta;
                beta = n_gamma;
            }
            else
            {
                beta = n_beta;
                gamma = n_gamma;

            }
        }
        return div_block;
    }
    RT_FUNCTION int FindDivGridNum(int divLevel, float2 uv)
    {
        int div_block = 0;
        int div_class = divLevel;
        float2 box_min = make_float2(0.0);
        float2 box_max = make_float2(1.0); 
        int axis = 0;
        while (div_class != 0)
        {
            div_block = div_block << 1;
            div_class -= 1;
            axis = 1 - axis;
            float2 box = box_max - box_min;
            float2 coord = uv - box_min;
            if (axis)
            {
                box.x /= 2;
                if (coord.x > box.x)
                {
                    div_block += 1;
                    box_min.x += box.x;
                }
                else
                {
                    box_max.x -= box.x;
                }
            }
            else
            {
                box.y /= 2;
                if (coord.y > box.y)
                {
                    div_block += 1;
                    box_min.y += box.y;
                }
                else
                {
                    box_max.y -= box.y;
                }
            } 
        }
        return div_block;
    }
    RT_FUNCTION int getLabel()
    {
#ifndef ZGCBPT
        return 0;
#endif 
        if (light_side)
        {
            return classTree::getLightLabel(position,normal,dir);
        }
        else
        {
            return classTree::getEyeLabel(position,normal,dir);
        }

        int grid_base = zgc_gridInfo[objectId].x;
        int grid_slot = zgc_gridInfo[objectId].y;
        int grid_level = zgc_gridInfo[objectId].z; 
        //return grid_base + FindDivGridNum(grid_level, uv);

        int div_tri_id = triBase[tri_id] + FindDivTriNum(divLevel[tri_id], tri_uv);
        return div_tris[div_tri_id].zoneNum;

    }

};
#endif