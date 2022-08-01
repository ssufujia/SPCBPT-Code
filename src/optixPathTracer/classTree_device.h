#ifndef CLASSTREE_DEVICE
#define CLASSTREE_DEVICE 
#include "classTree_common.h"   
#include"random.h"
namespace classTree
{
    rtBuffer<tree_node, 1>   light_trees;
    rtBuffer<tree_node, 1>   eye_trees;

    rtDeclareVariable(tree_node*, temple_tree, , ) = {nullptr};
    rtDeclareVariable(tree_node*, temple_light, , ) = {nullptr};
    rtDeclareVariable(tree_node*, temple_eye, , ) = {nullptr}; 
    RT_FUNCTION int getLabel(tree_node* root, float3 position, float3 normal = make_float3(0.0), float3 direction = make_float3(0.0))
    {
        return tree_index(root, position, normal, direction);
        int node_id = 0;
        int loop_time = 0;
        while (root[node_id].leaf == false)
        {
            loop_time += 1;
            node_id = root[node_id](position,normal,direction);
        }
        //rtPrintf("%d %f %f %f %d\n", loop_time,position.x,position.y,position.z,node_id);
        return root[node_id].label;
    }

    RT_FUNCTION int getLightLabel(float3 position, float3 normal = make_float3(0.0), float3 direction = make_float3(0.0))
    {
        if (temple_light != nullptr)
        { 
            //printf("get A\n");
            return getLabel(temple_light, position, normal, direction);
        }
        return getLabel(&(light_trees[0]), position);
    }
    RT_FUNCTION int getEyeLabel(float3 position,float3 normal = make_float3(0.0),float3 direction = make_float3(0.0))
    {
        if (temple_eye != nullptr)
        {
            //printf("get B\n");
            //rtPrintf("root type%d %f %f %f\n", temple_eye[0].type, temple_eye[0].mid.x, temple_eye[0].mid.y, temple_eye[0].mid.z);
            //if(getLabel(temple_eye, position, normal, direction) != getLabel(temple_eye, position, -normal, direction))
            //  rtPrintf("normal_change %d %d %f %f %f\n", getLabel(temple_eye, position, normal, direction), getLabel(temple_eye, position, -normal, direction),normal.x,normal.y,normal.z);
            return getLabel(temple_eye, position, normal, direction);
        }
        return getLabel(&(eye_trees[0]), position);
    }


    RT_FUNCTION float eval_contri(lightTreeNode& a, float3 position, float3 normal)
    { 
        //float3 diff; 
        float cos_theta;
        float3 diff = a.pos_box.mid_pos() - position ;
        float d = sqrt(dot(diff, diff));

        float3 a_pointer = a.pos_box.mid_pos() - a.pos_box.aa;
        float radians = sqrt(dot(a_pointer, a_pointer));
        if (d < radians)
        {
            d = 0.1;
            cos_theta = 1;
        } 
        else
        {
            float cos_theta_0 = dot(normal, normalize(diff));
            float theta_0 = acos(cos_theta_0);
            float cos_theta_1 = sqrt(d * d - radians * radians) / d;
            float theta_1 = acos(cos_theta_1);
            float theta = theta_1 > theta_0 ? 0 : theta_0 - theta_1;
            cos_theta = cos(theta);
            if (cos_theta < 0.01)cos_theta = 0.01;
            d = d - radians;


            float3 dirr = normalize(a.position - position);
            float coss = dot(dirr, normal); 
        }
         
        float g = cos_theta / (d * d);
        if (ENERGY_WEIGHT((a.color * g))<0.0)
        {
            rtPrintf("sorry but i return a invalid value");
        }
        return ENERGY_WEIGHT((a.color * g));
    }
    struct simple_heap
    {
        int* data;
        float* value;
        int size;
        RT_FUNCTION
        simple_heap(int* data,float* value,int target_size) :data(data),value(value), size(0)
        {
           // value = new float[target_size];
        }
        RT_FUNCTION
        ~simple_heap()
        {
            //delete[] value;
        }
        RT_FUNCTION int father(int i) { return (i - 1) / 2; }
        RT_FUNCTION int left_child(int i) { return 2 * i + 1;}
        RT_FUNCTION int right_child(int i) { return 2 * i + 2;}
        RT_FUNCTION bool leaf(int i) { return left_child(i) >= size; }
        RT_FUNCTION bool full_mid(int i) { return right_child(i) < size; }

        RT_FUNCTION void swap(int i, int j) 
        {
            float v = value[i];
            value[i] = value[j];
            value[j] = v; 
            int t = data[i];
            data[i] = data[j];
            data[j] = t; 
        }

        RT_FUNCTION void move_down_head()
        {
            int try_id = 0;
            int t_id;
            while (true)
            {
                if (leaf(try_id))return;
                if (full_mid(try_id) == false)
                {
                    t_id = left_child(try_id);
                }
                else
                {
                    t_id = value[left_child(try_id)] > value[right_child(try_id)] ? left_child(try_id) : right_child(try_id);
                }

                if (value[try_id] < value[t_id])
                {
                    swap(try_id, t_id);
                    try_id = t_id;
                }
                else
                {
                    break;
                }
            } 
        }
        RT_FUNCTION void move_up_back()
        {
            int try_id = size - 1;
            int t_id;
            while (try_id != 0)
            {
                t_id = father(try_id);
                if (value[try_id] > value[t_id])
                {
                    swap(try_id, t_id);
                    try_id = t_id;
                }
                else
                {
                    break;
                }
            }
        }
        RT_FUNCTION void insert_value(int id, float v)
        {
            value[size] = v;
            data[size] = id;
            size++;
            move_up_back();
        }
        RT_FUNCTION void down_head()
        {
            value[0] = 0;
            move_down_head();
        }
        RT_FUNCTION void remove_head()
        {
            //value[0] = -1;
            swap(0, size - 1);
            size--;
            move_down_head();
            //printf("remove last vertex%f\n", value[size - 1]);
        }
        RT_FUNCTION int head()
        {
            return data[0];
        }
        RT_FUNCTION bool check_valid()
        {
            float stand = value[0];
            for (int i = 1; i < size; i++)
            {
                //if (value[i]<0) { printf("compare %f %f %d\n",stand,value[i],i); return true; }
            }
            return false;
        }
    };
    rtDeclareVariable(light_tree_api, light_tree_dev, , ) = { };
    RT_FUNCTION void access_light_cut(float3 position, float3 normal, light_tree_api t, int* index, float* values, int cut_size)
    { 
        //return;
//        lightTreeNode node = t.get_root();
        simple_heap sh(index, values, cut_size); 
        sh.insert_value(t.root_id(), eval_contri(t.get_root(), position, normal));
        //rtPrintf("%d \n%f %f %f \n%d \n%f %f %f\n %f\n\n", t.size, position.x, position.y, position.z, cut_size, normal.x, normal.y, normal.z,values[0]);
        int loop_count = 0;
        bool p = false;
        while (sh.size < cut_size)
        {
            loop_count++; 
            //rtPrintf("head id%d %f\n", sh.head(), sh.value[0]);
            lightTreeNode& node = t[sh.head()];
            if (node.leaf == true)
            {
                //rtPrintf("loop_count %d %d\n", loop_count, sh.size);
                sh.down_head();
                 
                continue;
            }
            else
            { 
                sh.remove_head(); 
                sh.insert_value(node.left_id, eval_contri(t[node.left_id], position, normal));
                 
                sh.insert_value(node.right_id, eval_contri(t[node.right_id], position, normal));
                 

            }
        }

    }
    RT_FUNCTION void interior_node_decent(light_tree_api t, int* index, float* values, int cut_size, unsigned int & seed)
    {  
        for (int i = 0; i < cut_size; i++)
        {
            values[i] = 1.0;
            int id = index[i];
            while (true)
            {
                lightTreeNode& node = t[id];
                if (node.leaf == true)
                {
                    break;
                }
                float root_weight = ENERGY_WEIGHT(node.color), left_weight = ENERGY_WEIGHT(t[node.left_id].color), right_weight = ENERGY_WEIGHT(t[node.right_id].color);
                float random = rnd(seed) * root_weight;
                if (random < left_weight)
                {
                    id = node.left_id;
                    values[i] *= left_weight / root_weight;
                }
                else
                {
                    id = node.right_id;
                    values[i] *= right_weight / root_weight;

                }
                
            }
            //rtPrintf("leaf id %d\n", id);
            index[i] = id;
        }
    }

}

#endif // !CLASSTREE_HOST
