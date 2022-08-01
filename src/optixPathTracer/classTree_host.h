#ifndef CLASSTREE_HOST
#define CLASSTREE_HOST

#include "classTree_common.h"
#include"BDPT_STRUCT.h" 
#include"ZGC.h"
#include<random>
#include<vector>
#include <sutil.h>
#include<fstream>
#include<algorithm>
#include<map>
using namespace optix;

namespace classTree
{
    struct classTreeHost
    {
        void init(Context& context)
        {
            tree_node empty_node;
            empty_node.leaf = true;
            empty_node.label = 0;


            std::vector<tree_node> Tree_node_eye;
            Tree_node_eye.push_back(empty_node);
            if (false)//load existing tree
            {
                std::ifstream inFile;
                inFile.open("./exp_record/OPTP_data/classTree_eye.txt");
                bool leaf;
                while (inFile >> leaf)
                {
                    tree_node node;
                    inFile >> node.label;
                    node.leaf = leaf;
                    if (!leaf)
                    {
                        inFile >> node.mid.x >> node.mid.y >> node.mid.z;
                        for (int i = 0; i < 8; i++)
                        {
                            inFile >> node.child[i];
                        }
                    }
                    Tree_node_eye.push_back(node);
                }
            }

            std::vector<tree_node> Tree_node_light;
            Tree_node_light.push_back(empty_node);
            if (false)//load existing tree
            {
                std::ifstream inFile;
                inFile.open("./exp_record/OPTP_data/classTree_light.txt");
                bool leaf;
                while (inFile >> leaf)
                {
                    tree_node node;
                    inFile >> node.label;
                    node.leaf = leaf;
                    if (!leaf)
                    {
                        inFile >> node.mid.x >> node.mid.y >> node.mid.z;
                        for (int i = 0; i < 8; i++)
                        {
                            inFile >> node.child[i];
                        }
                    }
                    Tree_node_light.push_back(node);
                }
            }

            //map to cpu
            auto light_tree_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, Tree_node_light.size());
            light_tree_buffer->setElementSize(sizeof(tree_node));
            context["classTree::light_trees"]->setBuffer(light_tree_buffer);
            auto p = reinterpret_cast<tree_node*>(light_tree_buffer->map());
            memcpy(p, Tree_node_light.data(), sizeof(tree_node) * Tree_node_light.size());
            light_tree_buffer->unmap();

            auto eye_tree_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, Tree_node_eye.size());
            eye_tree_buffer->setElementSize(sizeof(tree_node));
            context["classTree::eye_trees"]->setBuffer(eye_tree_buffer);
            p = reinterpret_cast<tree_node*>(eye_tree_buffer->map());
            memcpy(p, Tree_node_eye.data(), sizeof(tree_node) * Tree_node_eye.size());
            eye_tree_buffer->unmap();

            printf("NUMBER:%d %d\n", Tree_node_light.size(), Tree_node_eye.size());
        }
    };
    struct buildLightTree
    {  
        struct aabb
        {
            float3 aa;
            float3 bb;
            __host__ __device__
            aabb make_union(aabb& a)
            {
                aabb c;
                c.aa = fminf(a.aa, aa);
                c.bb = fmaxf(a.bb, bb);
                return c;
            }
            __host__ __device__
            float3 diag()
            {
                return aa - bb;
            }
            float3 dir_mid()
            {
                return normalize(aa + bb);
            }
            float half_angle_cos()
            {
                float3 a = dir_mid();
                return abs(dot(normalize(aa), a));

            }
            float3 mid_pos()
            {
                return (aa + bb) / 2;
            }
        }; 
        struct light_tree_node
        {
            float flux;
            bool leaf;
            int left_id;//when leaf == true, the real vertex id would be saved on the leaf_id;
            int right_id;
        };
        struct light_tree_construct_node: light_tree_node
        { 
            aabb pos_aabb;
            aabb dir_aabb;
            aabb normal_aabb; 
            float weight;
            bool used;
            float3 position()
            {
                return pos_aabb.mid_pos();
            }
            light_tree_construct_node(float3 pos, float3 dir, float3 normal) :used(false)
            {
                pos_aabb.aa = pos_aabb.bb = pos;
                dir_aabb.aa = dir_aabb.bb = dir;
                normal_aabb.aa = normal_aabb.bb = normal;
                flux = 1;

            }
            light_tree_construct_node() :used(false) {}
            __host__ __device__
            float metric(float scene_diag2)
            {
                float3 dis = pos_aabb.diag();
                float d = dis.x * dis.x + dis.y * dis.y + dis.z * dis.z;
                 
                float normal_cos = normal_aabb.half_angle_cos();
                float3 normal_dis = normal_aabb.diag();
                float d2 = dot(normal_dis,normal_dis) / 4;

                float dir_cos = dir_aabb.half_angle_cos();
                float weight = flux; 
                 
                float ans = weight * (d + (1 - d2) * scene_diag2 / 10000.0);
                if (isnan(ans) || isinf(ans))
                {
                    printf("error metric");
                }
                //printf("d2 %f %f %f\n", d2, scene_diag2 / 100.0,ans);
                return ans;
                return weight * (d + (1 - d2) * scene_diag2);
                return weight * (d + (1 - normal_cos * normal_cos) * scene_diag2 + (0.0 * (1 - dir_cos * dir_cos)) * scene_diag2);
            }
            light_tree_construct_node operator+(light_tree_construct_node& b)
            {
                light_tree_construct_node ans;
                ans.flux = flux + b.flux; 
                ans.weight = weight + b.weight;
                ans.pos_aabb = pos_aabb.make_union(b.pos_aabb);
                ans.dir_aabb = dir_aabb.make_union(b.dir_aabb);
                ans.normal_aabb = normal_aabb.make_union(b.normal_aabb);
                return ans;
            }
        };


        struct light_tree
        {
            std::vector<light_tree_node> nodes;
            std::vector<VPL> vpls;
        };

        struct heap_pair
        {
            int src_id;
            int end_id;
            float metric;
            __host__ __device__ 
                bool operator<(const heap_pair& b)
            {
                return metric > b.metric;
            }
        };
        template<typename T>
        heap_pair get_closest_pair(int begin_id, std::vector<light_tree_construct_node>& nodes, T& speedUpTree, float diag2)
        {
            heap_pair p;
            p.src_id = begin_id;
            p.metric = FLT_MAX;
            
            light_tree_construct_node& c_node = nodes[begin_id];
            auto potential_v = speedUpTree(c_node.position()).v; 
            
            float tem = 0;
            //p.end_id = (potential_v.begin()->first);
            for (auto tp = potential_v.begin(); tp!=potential_v.end(); tp++)
            {
                int t_id = tp->first;
                if (t_id == begin_id)continue;
                light_tree_construct_node n_construct_node = c_node + nodes[t_id];
                float _metric = n_construct_node.metric(diag2);
                tem = _metric;
                if (_metric < p.metric)
                {
                    p.metric = _metric;
                    p.end_id = t_id; 
                } 
            }
            if (nodes[p.end_id].used == true)
            {
                printf("invalid found %f %d\n", tem, potential_v.size());
            }
            
            return p;
        }
        light_tree_construct_node light_tree_step(int i, int j, std::vector<light_tree_construct_node>& tree)
        {
            light_tree_construct_node ans = tree[i] + tree[j];
            ans.leaf = false;
            ans.left_id = i;
            ans.right_id = j;
            tree[i].used = true;
            tree[j].used = true;
            ans.used = false;
            tree.push_back(ans);
            //printf("flux add %f %f %f\n", tree[i].flux, tree[j].flux, tree.back().flux);
            return ans;
        }
        struct SUTree
        { 

            struct SU_tree_node :tree_node
            {
                std::map<int, float3> v;
                float3 half_block;
                int father;
                int depth;
                bool valid;
                std::vector<float3> get_potential_node(float3 src, float dis2)
                {
                    std::vector<float3> ans(0);
                    bool need_search[6] = { false };
                    for (int i = 0; i < 8; i++)
                    {
                        int dx = (i % 2) == 0 ? -1 : 1;
                        int dy = ((i >> 1) % 2) == 0 ? -1 : 1;
                        int dz = ((i >> 2) % 2) == 0 ? -1 : 1;
                        float3 d_position = half_block * make_float3(dx, dy, dz);
                        float3 position = mid + d_position;
                        float3 position_change = src - position;
                        float n_dis2 = dot(position_change, position_change);
                        if (n_dis2 < dis2)
                        {
                            need_search[0 + (dx == -1 ? 0 : 1)] = true;
                            need_search[2 + (dy == -1 ? 0 : 1)] = true;
                            need_search[4 + (dz == -1 ? 0 : 1)] = true;
                        }
                    }
                    for (int i = 0; i < 6; i++)
                    {
                        int d = i % 2 == 0 ? -1 : 1;
                        float3 mask;
                        if ((i >> 1) == 0)
                        {
                            mask = make_float3(1, 0, 0);
                        }
                        if ((i >> 1) == 1)
                        {
                            mask = make_float3(0, 1, 0);
                        }
                        if ((i >> 1) == 2)
                        {
                            mask = make_float3(0, 0, 1);
                        }
                        float3 d_position = half_block * mask * d * 1.01 + mid;
                        ans.push_back(d_position);
                    }
                    return ans;
                }
            };

            std::vector<float3> block_size;
            float3 bbox_min;
            float3 bbox_max;
            std::vector<SU_tree_node> v;
            int min_count;
            SU_tree_node& operator()(float3 position,bool insert_op = false)
            {
                int node_id = 0;
                while (v[node_id].leaf == false)
                {
                    int n_id = v[node_id](position);
                    node_id = n_id;
                }
                while (v[node_id].valid == false && node_id != 0)
                {
                    node_id = v[node_id].father;
                }
                //rtPrintf("%d %f %f %f %d\n", loop_time,position.x,position.y,position.z,node_id);
                //printf("su tree return a node with %d contains %d\n", v[node_id].v.size(), v[node_id].valid);
                if (node_id == 0 && v[node_id].v.size() == 0 && insert_op == false)
                {
                    static int alter_id = 0;
                    while (v[alter_id].valid == false||alter_id == 0)
                    {
                        alter_id++;
                        if (alter_id >= v.size())alter_id = 0;

                        //printf("alter id step %d %d %d\n", alter_id, v[alter_id].v.size(),v[alter_id].valid);
                    }
                    //printf("return a empty null block %d %d %d\n",alter_id,v[alter_id].v.size(),v[alter_id].valid);
                    return v[alter_id];
                }
                return v[node_id];
            }
            void check_size(SU_tree_node& t)
            { 
                if (t.v.size() < min_count&& t.depth != 0)
                {
                    for (auto p = t.v.begin(); p != t.v.end(); p++)
                    {
                        v[t.father].v.insert(*p);
                    }
                    t.valid = false;
                    //printf("cut a node\n");
                }
                else
                {
                    t.valid = true;
                }
            }
            void remove(int id, float3 position)
            {
                //printf("ready to cut a node");
                auto &t = (*this)(position);
                //printf("map size before %d\n", t.v.size());
                int count = t.v.erase(id);
                //printf("map size after %d\n", t.v.size());
                //if(t.size)
                check_size(t);
            }
            void insert(int id, float3 position)
            {
                //printf("inser op\n");
                auto &t = (*this)(position,true);
                t.v.insert(std::make_pair(id, position));
            }

            void para_initial(std::vector<float3>& samples, int max_depth)
            { 
                for (auto p = samples.begin(); p != samples.end(); p++)
                {
           
                    bbox_min = fminf(bbox_min, *p);
                    bbox_max = fmaxf(bbox_max, *p);
                } 
                float3 bbox_block = bbox_max - bbox_min;
                for (int i = 0; i < max_depth + 10; i++)
                {
                    block_size.push_back(bbox_block);
                    bbox_block /= 2;
                }
            }
            void split(int id)
            {
                int back = v.size();
                //devide_node& node = v[id];

                v[id].leaf = false;
                float3 inch = block_size[v[id].depth + 2];
                for (int i = 0; i < 8; i++)
                {
                    v[id].child[i] = back + i;
                    v.push_back(SU_tree_node());
                    //printf("sizeeee %d %d\n", v[id].v.size());

                    v.back().father = id;
                    v.back().depth = v[id].depth + 1;

                    float3 delta_mid = make_float3(
                        (i >> 0) % 2 == 0 ? -inch.x : inch.x,
                        (i >> 1) % 2 == 0 ? -inch.y : inch.y,
                        (i >> 2) % 2 == 0 ? -inch.z : inch.z
                    );
                    v.back().mid = v[id].mid + delta_mid;
                    v.back().half_block = block_size[v.back().depth + 1];
                }
                for (auto p = v[id].v.begin(); p != v[id].v.end(); p++)
                {
                    // printf("%d\n", v[v[id](p->position)]);
                    v[v[id](p->second)].v.insert(*p);
                } 
                v[id].v.clear();
            }
            SUTree(std::vector<float3>& v_pos, int max_depth = 10, int min_samples = 50) :min_count(min_samples)
            { 
                para_initial(v_pos, max_depth); 
                v.push_back(SU_tree_node());
                for (int i = 0; i < v_pos.size(); i++)
                {
                    v[0].v.insert(std::make_pair(i, v_pos[i]));
                }
                v[0].mid = (bbox_max + bbox_min) / 2;
                v[0].depth = 0;
                v[0].half_block = block_size[v[0].depth + 1];

                for (int i = 0; i < v.size(); i++)
                { 
                    if (v[i].v.size() > min_samples)
                    {
                        split(i);
                    }
                }
                //printf("bbox_min: %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z);
                //printf("%d", v[0].v.size());
                //exit(0);
                for (int i = v.size() - 1; i >= 0; i--)
                { 
                    check_size(v[i]);
                }
            }
        };
        std::vector<int> build_light_tree(std::vector<VPL>& vpls)
        { 
            //get diag
            //build SpeedUpTree
            std::vector<float3> vpl_pos;
            for (int i = 0; i < vpls.size(); i++) 
            {
                vpl_pos.push_back(vpls[i].position);
            }

            auto bound_tree = SUTree(vpl_pos);
            vpl_pos.clear();
             
            float3 diag = bound_tree.bbox_max - bound_tree.bbox_min;
            float diag2 = dot(diag, diag);
            //for each vpls, build light_tree_construct_node 
            std::vector<light_tree_construct_node> light_tree_construct(0);
            for (int i = 0; i < vpls.size(); i++)
            {
                light_tree_construct.push_back(light_tree_construct_node(vpls[i].position, vpls[i].dir, vpls[i].normal));
                light_tree_construct.back().flux = ENERGY_WEIGHT(vpls[i].color);
                light_tree_construct.back().weight = vpls[i].weight;// ENERGY_WEIGHT(vpls[i].color);
                light_tree_construct.back().leaf = true;


            }
            //build tree from leaf to root
            
            
            std::vector<heap_pair> h;
            //--for each node
            //----compute min distance in the grid it belongs to
            //----check if there is any potential block 
            //----compute min distance, insert to min heap
            for (int i = 0; i < vpls.size(); i++)
            {
                heap_pair p = get_closest_pair(i, light_tree_construct, bound_tree, diag2);
                //to be rewrite
                h.push_back(p); 
            }
            make_heap(h.begin(), h.end());


            //--loop heap
            //----check valid
            //------if invalid, seach the closest node again
            //----if valid, merge node, tree grows, delete ordinal nodes, insert new node
            while (h.size() > 0 && light_tree_construct.size() < 2 * vpls.size() - 1)
            {
                if (h.size() == 1)break;
                int src = h[0].src_id;
                int end = h[0].end_id;   
                //printf("%d %d %d %d %d\n",h.size(), src, end,light_tree_construct.size(),vpls.size());
                if (light_tree_construct[src].used == true)
                {
                    pop_heap(h.begin(), h.end());
                    h.pop_back(); 
                    continue;
                }
                else if (light_tree_construct[end].used == true)
                {
                    heap_pair p = h[0];

                    pop_heap(h.begin(), h.end());
                    h.pop_back();

                    p = get_closest_pair(p.src_id, light_tree_construct, bound_tree, diag2);

                    h.push_back(p);
                    push_heap(h.begin(), h.end()); 
                    continue;
                }

                //待续：如果点对是合法的，那么----
                auto n_node = light_tree_step(src, end, light_tree_construct);
                int n_node_id = light_tree_construct.size() - 1;

                //printf("A %d %d ",light_tree_construct.size(), vpls.size() * 2 + 1);
                if (light_tree_construct.size() == vpls.size() * 2 - 1)break;
                bound_tree.remove(src, light_tree_construct[src].position());   // ***
                bound_tree.remove(end, light_tree_construct[end].position());   // ***
                bound_tree.insert(n_node_id, n_node.position());// ***
                //printf("insert %d valid%d\n", n_node_id, light_tree_construct[n_node_id].valid);

                
                pop_heap(h.begin(), h.end());
                h.pop_back();
                heap_pair p = get_closest_pair(n_node_id, light_tree_construct, bound_tree, diag2);
                h.push_back(p);
                push_heap(h.begin(), h.end()); 
            }

            std::vector<int> ans(light_tree_construct.size(),-1);
            std::multimap<float, int> split_one;
            int root_id = light_tree_construct.size() - 1;
            split_one.insert(make_pair(light_tree_construct.back().weight, root_id));
            while (split_one.size() < 750)
            {
                //printf("A");
                auto p = split_one.end();
                p--;
                int id = p->second;
                //printf("%f \n",p->first);
                split_one.erase(p);
                auto& root_node = light_tree_construct[id];
                if (root_node.leaf == true)
                {
                    //printf("B");
                    continue;
                }
                int left_id = root_node.left_id;
                int right_id = root_node.right_id;
                split_one.insert(make_pair(light_tree_construct[left_id].weight, left_id));
                split_one.insert(make_pair(light_tree_construct[right_id].weight, right_id));
                //printf("zip %f %f\n", light_tree_construct[left_id].flux, light_tree_construct[right_id].flux);
            }
            static float split_weight_count = 0;
            for (auto p = split_one.begin(); p != split_one.end(); p++)
            {
                int id = p->second;
                //printf("split sum %f\n", light_tree_construct[id].flux);

            }
            std::vector<int> labels;
            int label_count = 0;
            while (split_one.size() != 0)
            {
                labels.clear();
                labels.push_back(split_one.begin()->second);
                split_one.erase(split_one.begin());
                while (labels.size() > 0)
                {
                    int id = labels.back();
                    labels.pop_back(); 
                    ans[id] = label_count;
                    if (light_tree_construct[id].leaf == false)
                    {
                        
                        labels.push_back(light_tree_construct[id].left_id);
                        labels.push_back(light_tree_construct[id].right_id);
                    }

                    //printf("loop");
                }
                label_count++;
            }

            return ans;
        }


    };

    std::vector<int> build_light_cut(std::vector<BDPTVertex> v)
    {
        std::vector<VPL> vv;
        for (int i = 0; i < v.size(); i++)
        {
            VPL a;
            a.color = v[i].flux / v[i].pdf;
            a.dir = normalize(v[i].lastPosition - v[i].position);
            a.position = v[i].position;
            a.normal = v[i].normal;
            a.weight = 1.0;
            vv.push_back(a);
        }
        return buildLightTree().build_light_tree(vv);
        printf("BB\n");
    }

    struct buildTreeBaseOnWeightedSample
    {
        struct devide_node :tree_node
        {
            std::vector<divide_weight> v;
            int depth;
            int normal_depth;
            float weight;
            int father;
            bool colored;

            devide_node() :weight(0),v(0),father(0),depth(0), normal_depth(0),colored(false)
            {

            } 

            __inline__ void add_sample(divide_weight w)
            {
                v.push_back(w);
                weight += w.weight;
            }
        };
        std::vector<devide_node> v;
        int label_id;
        std::vector<float3> block_size;

        float3 bbox_min;// = make_float3(FLT_MAX);
        float3 bbox_max;// = make_float3(FLT_MIN);

        buildTreeBaseOnWeightedSample():v(0),label_id(0)
        {
            bbox_min = make_float3(FLT_MAX);
            bbox_max = make_float3(FLT_MIN);
        }
        void split(int id)
        {
            int back = v.size();
            //devide_node& node = v[id];

            v[id].leaf = false;
            int split_type = v[id].depth == 151 ? classTree::tree_node_type::type_normal:classTree::tree_node_type::type_position;
            v[id].type = split_type;

            if (split_type == classTree::tree_node_type::type_normal)
            { 
                //inch = block_size[v[id].depth + 1];
                for (int i = 0; i < 8; i++)
                {
                    v[id].child[i] = back + i;
                    v.push_back(devide_node());
                    //printf("sizeeee %d %d\n", v[id].v.size());

                    v.back().father = id;
                    v.back().depth = v[id].depth + 1;
                    v.back().normal_depth = v[id].normal_depth + 1;
                    //v.back().type = classTree::tree_node_type::type_normal;
                    v.back().mid = v[id].mid;
                }
                v[id].mid = make_float3(0.0);
            }
            else 
            {
                float3 inch = block_size[v[id].depth + 2 - v[id].normal_depth];
                for (int i = 0; i < 8; i++)
                {
                    v[id].child[i] = back + i;
                    v.push_back(devide_node());
                    //printf("sizeeee %d %d\n", v[id].v.size());

                    v.back().father = id;
                    v.back().depth = v[id].depth + 1;
                    v.back().normal_depth = v[id].normal_depth;

                    float3 delta_mid = make_float3(
                        (i >> 0) % 2 == 0 ? -inch.x : inch.x,
                        (i >> 1) % 2 == 0 ? -inch.y : inch.y,
                        (i >> 2) % 2 == 0 ? -inch.z : inch.z
                    );
                    v.back().mid = v[id].mid + delta_mid;

                }

            }
            for (auto p = v[id].v.begin(); p != v[id].v.end(); p++)
            {
               // printf("%d\n", v[v[id](p->position)]);
                v[v[id](p->position,p->dir)].add_sample(*p);
            }
            v[id].weight = 0;
            v[id].v.clear();
        }
        void sample_back(int id)
        {
            devide_node& node = v[id];
            for (auto p = node.v.begin(); p != node.v.end(); p++)
            {
                v[node.father].add_sample(*p);
            }

        }
        void color(int id,int label)
        {
            devide_node& node = v[id];
            if (node.colored == true)
                return;

            node.label = label_id;
            node.colored = true;
            if (node.leaf == false)
            {
                for (int i = 0; i < 8; i++)
                {
                    int c_id = node.child[i]; 
                    color(c_id, label); 
                }
            }
        }
        template<typename T>
        void para_initial(std::vector<T>& samples,int max_depth)
        { 
            float unoramlize_weight = 0.0;
            for (auto p = samples.begin(); p != samples.end(); p++)
            {
                unoramlize_weight += p->weight;
                bbox_min = fminf(bbox_min, p->position);
                bbox_max = fmaxf(bbox_max, p->position);
            }
            for (auto p = samples.begin(); p != samples.end(); p++)
            {
                p->weight /= unoramlize_weight;
            }
            float3 bbox_block = bbox_max - bbox_min;
            for (int i = 0; i < max_depth + 10; i++)
            {
                block_size.push_back(bbox_block);
                bbox_block /= 2;
            } 
        }
        tree operator()(std::vector<divide_weight>& samples, int max_depth = 10, int refer_num_class = 100)
        {
            //weight normalization
            para_initial(samples, max_depth);

            float max_weight = 0.1 / refer_num_class;
            float min_weight = 1.0 / refer_num_class;

            v.push_back(devide_node());
            v[0].v = std::vector<divide_weight>(samples.begin(),samples.end());
            v[0].weight = 1;
            v[0].mid = (bbox_max + bbox_min) / 2;

            for (int i = 0; i < v.size(); i++)
            {
                //printf("weight %f\n", v[i].weight);
                if (v[i].weight > max_weight && v[i].depth < max_depth)
                {
                    split(i);
                     
                }
            }
            //printf("bbox_min: %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z);
            //printf("%d", v[0].v.size());
            //exit(0);
            for (int i = v.size()-1; i > 0; i--)
            {
                if (v[i].weight >= min_weight)
                {
                    color(i, label_id);
                    label_id++;
                }
                else
                {
                    sample_back(i);
                }

            }
            color(0, label_id);

            tree_node* p = new tree_node[v.size()];
            float3* centers = new float3[SUBSPACE_NUM];
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                centers[i] = make_float3(0.0);
            }
            float center_weight[SUBSPACE_NUM] = {0};
            for (int i = 0; i < v.size(); i++)
            {
                p[i] = v[i];
                if (v[i].leaf == true)
                {
                    float tt = center_weight[v[i].label];
                    float3 ttt = centers[v[i].label];

                    center_weight[v[i].label] += v[i].weight;
                    centers[v[i].label] += v[i].weight * v[i].mid;

                    float3 t = centers[v[i].label] / center_weight[v[i].label];
                }
            }
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                centers[i] = center_weight[i] == 0.0 ? make_float3(0.0) : centers[i] / center_weight[i]; 
            }

            tree t;
            t.v = p;
            t.center = centers;
            t.size = v.size();
            t.in_gpu = false;
            t.max_label = label_id;
            t.bbox_max = bbox_max;
            t.bbox_min = bbox_min;
            printf("class tree building complete:size %d and max-label %d\n", t.size, t.max_label);
            return t;
        }

    };


    struct buildTreeBaseOnExistSample
    {
        struct devide_node :tree_node
        {
            std::vector<divide_weight_with_label> v;
            int depth;
            float weight;
            float correct_weight;
            int father;
            int position_depth;
            int normal_depth;
            int dir_depth;

            devide_node() :weight(0), v(0), father(0), depth(0), correct_weight(0), position_depth(0), normal_depth(0), dir_depth(0)
            {

            }

            __inline__ void add_sample(divide_weight_with_label w)
            {
                v.push_back(w);
                weight += w.weight;
            }
            bool need_split()
            { 
                //printf("%f %f\n", correct_weight, weight);
                return v.size() != 0 && correct_weight < weight;
            }
        };
        std::vector<devide_node> v;
        int label_id;
        std::vector<float3> block_size;
        std::vector<float3> direction_block_size;

        float3 bbox_min;// = make_float3(FLT_MAX);
        float3 bbox_max;// = make_float3(FLT_MIN);

        buildTreeBaseOnExistSample() :v(0), label_id(0)
        {
            bbox_min = make_float3(FLT_MAX);
            bbox_max = make_float3(FLT_MIN);
        }
        float split(int id)
        { 
            auto split_type = (v[id].depth % 2 == 0 || v[id].normal_depth > 3) ? tree_node_type::type_position : tree_node_type::type_normal;
            if (v[id].depth == 7 || v[id].depth == 9)
            {
                if (DIR_JUDGE)
                    split_type = tree_node_type::type_direction;
            }
            //split_type = tree_node_type::type_position;
            int back = v.size();
            //devide_node& node = v[id];

            v[id].leaf = false;
            float3 inch;
            if (split_type == tree_node_type::type_position)
            {
                inch = block_size[v[id].position_depth + 1];
            }
            else if (split_type == tree_node_type::type_normal)
            {
                inch = direction_block_size[v[id].normal_depth + 1];
            }
            else if (split_type == tree_node_type::type_direction)
            {
                inch = direction_block_size[v[id].dir_depth + 1];
            }

            float3 mid;
            
            if (v[id].normal_depth == 0 && split_type == tree_node_type::type_normal)
            { 
                mid = make_float3(0.0); 
            }
            else if (v[id].dir_depth == 0 && split_type == tree_node_type::type_direction)
            {
                mid = make_float3(0.0);
            }
            else if (v[id].position_depth == 0)
            {
                mid = v[id].mid;
            }
            else
            {  
                int L_id = id;
                int t_id = v[id].father;

                while (t_id != 0 && v[t_id].type != split_type)
                {
                    L_id = t_id;
                    t_id = v[t_id].father;
                }

                mid = v[t_id].mid;
                int c_id_local = 0;
                for (; c_id_local < 8; c_id_local++)
                {
                    if (v[t_id].child[c_id_local] == L_id)break;
                }

                float3 delta_mid = make_float3(
                    (c_id_local >> 0) % 2 == 0 ? -inch.x : inch.x,
                    (c_id_local >> 1) % 2 == 0 ? -inch.y : inch.y,
                    (c_id_local >> 2) % 2 == 0 ? -inch.z : inch.z
                );
                mid += delta_mid;
            }
            v[id].mid = mid;
            v[id].type = split_type;

            for (int i = 0; i < 8; i++)
            {
                v[id].child[i] = back + i;
                v.push_back(devide_node());
                //printf("sizeeee %d %d\n", v[id].v.size());

                v.back().father = id;
                v.back().depth = v[id].depth + 1;

                //float3 delta_mid = make_float3(
                //    (i >> 0) % 2 == 0 ? -inch.x : inch.x,
                //    (i >> 1) % 2 == 0 ? -inch.y : inch.y,
                //    (i >> 2) % 2 == 0 ? -inch.z : inch.z
                //);
                //v.back().mid = mid + delta_mid;
                v.back().label = v[id].label;
                
                v.back().position_depth = v[id].position_depth + (split_type == tree_node_type::type_position);
                v.back().normal_depth = v[id].normal_depth + (split_type == tree_node_type::type_normal);
                v.back().dir_depth = v[id].dir_depth + (split_type == tree_node_type::type_direction);
                
            }
            for (auto p = v[id].v.begin(); p != v[id].v.end(); p++)
            {
                // printf("%d\n", v[v[id](p->position)]);
                //printf("A %d",id);
                v[v[id](p->position,p->normal,p->dir)].add_sample(*p);
                //printf("B");
            }

            float n_correct_weight = 0.0;
            for (int i = 0; i < 8; i++)
            {
                color(v[id].child[i]); 
                n_correct_weight += v[v[id].child[i]].correct_weight;
            }
            v[id].weight = 0;
            v[id].v.clear();
            return n_correct_weight;
        } 
        template<typename T>
        void para_initial(std::vector<T>& samples, int max_depth)
        {
            float unoramlize_weight = 0.0;
            for (auto p = samples.begin(); p != samples.end(); p++)
            {
                unoramlize_weight += p->weight;
                //printf("%f\n", p->weight);
                bbox_min = fminf(bbox_min, p->position);
                bbox_max = fmaxf(bbox_max, p->position);
            }
            for (auto p = samples.begin(); p != samples.end(); p++)
            {
                p->weight /= unoramlize_weight;
            }
            //printf("unormal weight %f\n", unoramlize_weight);
            float3 bbox_block = (bbox_max - bbox_min) ;
        
            for (int i = 0; i < max_depth + 10; i++)
            {
                block_size.push_back(bbox_block);
                bbox_block /= 2;
            }
            float3 direction_block = make_float3(2.0);
            for (int i = 0; i < 15; i++)
            {
                direction_block_size.push_back(direction_block);
                direction_block /= 2;
            }
        }

        void color(int id)
        {
            auto& t = v[id];
            if (t.v.size() == 0)
            {
                t.correct_weight = 0.0;
                return;
            }
            bool need_split = false;
            t.label = t.v[0].label;
            for (int i = 0; i < t.v.size(); i++)
            {
                if (t.v[i].label != t.label)
                {
                    need_split = true;
                    break;
                }
            }
            if (need_split)
            {
                float weights[SUBSPACE_NUM] = { 0 };
                float max_weight = 0.0;
                int max_weight_id = 0;
                for (int i = 0; i < t.v.size(); i++)
                {
                    weights[t.v[i].label] += t.v[i].weight;
                    if (max_weight < weights[t.v[i].label])
                    {
                        max_weight = weights[t.v[i].label];
                        max_weight_id = t.v[i].label;

                    } 
                }
                t.label = max_weight_id;
                t.correct_weight = max_weight;
            } 
            else  
            {
                t.correct_weight = t.weight;
            }
        }
        tree operator()(std::vector<divide_weight_with_label>& samples,float threshold, int max_depth = 10, int refer_num_class = 100)
        {
            int max_label = 0;
            //weight normalization
            para_initial(samples, max_depth);

            float max_weight = 0.1 / refer_num_class;
            float min_weight = 1.0 / refer_num_class;

            v.push_back(devide_node());
            v[0].v = std::vector<divide_weight_with_label>(samples.begin(), samples.end());
            v[0].weight = 1;
            v[0].mid = (bbox_max + bbox_min) / 2;
            printf("bounding box max: %f %f %f\n", bbox_max.x, bbox_max.y, bbox_max.z);
            printf("bounding box min: %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z);
            color(0);
            //printf("root correct_weight %f\n", v[0].correct_weight);
            float c_w = v[0].correct_weight;

            for (int i = 0; i < v.size(); i++)
            {
                max_label = max(v[i].label, max_label); 
                if (v[i].need_split() && v[i].depth < max_depth && threshold>c_w)
                {
                    c_w -= v[i].correct_weight;
                    c_w += split(i);

                } 
            }
            //printf("bbox_min: %f %f %f\n", bbox_min.x, bbox_min.y, bbox_min.z);
            //printf("%d", v[0].v.size());
            //exit(0);

            int valid_id = 0;
            int sum_id = 0;

            for (int t = 0; t < v.size(); t++)
            {
                if (!v[t].leaf)continue;
                for (int j = 0; j < v[t].v.size(); j++)
                {
                    if (v[t].label == v[t].v[j].label)
                    {
                        valid_id++;
                    }
                    sum_id++;
                }
            }
            printf("acc:%d/%d %e %e\n", valid_id, sum_id, c_w, float(valid_id) / sum_id);

            tree_node* p = new tree_node[v.size()];
            float3* centers = new float3[SUBSPACE_NUM];
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                centers[i] = make_float3(0.0);
            }
            float center_weight[SUBSPACE_NUM] = { 0 };
            for (int i = 0; i < v.size(); i++)
            {
                p[i] = v[i];
                if (v[i].leaf == true)
                {
                    float tt = center_weight[v[i].label];
                    float3 ttt = centers[v[i].label];

                    center_weight[v[i].label] += v[i].weight;
                    centers[v[i].label] += v[i].weight * v[i].mid;

                    float3 t = centers[v[i].label] / center_weight[v[i].label];
                }
            }
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {
                centers[i] = center_weight[i] == 0.0 ? make_float3(0.0) : centers[i] / center_weight[i];
            }

            tree t;
            t.v = p;
            t.center = centers;
            t.size = v.size();
            t.in_gpu = false;
            t.max_label = max_label;
            t.bbox_max = bbox_max;
            t.bbox_min = bbox_min;
            printf("class tree building complete:size %d and max-label %d\n", t.size, t.max_label);
            return t;
        }

    };


}
classTree::classTreeHost class_tree_api;
#include"kd_tree.h"
#include<random>

namespace classTree
{

    struct KD_tree_api
    {
        std::vector<KD_node> nodes;
        float3 bbox_max;
        float3 bbox_min;
        template<typename T>
        void build_from_labeled_position(std::vector<T> p_with_label)
        {
            static splitChoice split_choice = RoundRobin;
            std::vector<KD_node*> kd_nodes_p;
            std::vector<KD_node> t_nodes;
            nodes.clear(); 
            for (auto p = p_with_label.begin(); p != p_with_label.end(); p++)
            {
                t_nodes.push_back(KD_node());
                t_nodes.back().position = p->position;
                t_nodes.back().index = p->label;
                t_nodes.back().valid = true;
            }
            nodes.resize(p_with_label.size() * 20 + 1);
            for (int i = 0; i < p_with_label.size(); i++)
            {
                kd_nodes_p.push_back(&t_nodes[i]);
            }
            for (int i = 0; i < nodes.size(); i++)nodes[i].valid = false;


            bbox_min = make_float3(std::numeric_limits<float>::max());
            bbox_max = make_float3(-std::numeric_limits<float>::max());
            // Compute the bounds of the photons
            for (unsigned int i = 0; i < kd_nodes_p.size(); ++i) {
                float3 position = (*kd_nodes_p[i]).position;
                bbox_min = fminf(bbox_min, position);
                bbox_max = fmaxf(bbox_max, position);
            }

            buildKDTree<KD_node>(kd_nodes_p.data(), 0, p_with_label.size(), 0, nodes.data(), 0, split_choice, make_float3(0.0), make_float3(0));
        }
        void build_from_position(std::vector<float3> positions)
        {
            static splitChoice split_choice = RoundRobin; 
            std::vector<KD_node*> kd_nodes_p;
            std::vector<KD_node> t_nodes;
            nodes.clear();
            for (auto p = positions.begin(); p != positions.end(); p++)
            {
                t_nodes.push_back(KD_node());
                t_nodes.back().position = *p;
                t_nodes.back().index = t_nodes.size() - 1;
                t_nodes.back().valid = true;
            }
            nodes.resize(positions.size() * 20 + 1);
            for (int i = 0; i < positions.size(); i++)
            {
                kd_nodes_p.push_back(&t_nodes[i]);
            }
            for (int i = 0; i < nodes.size(); i++)nodes[i].valid = false;


            bbox_min = make_float3(std::numeric_limits<float>::max());
            bbox_max = make_float3(-std::numeric_limits<float>::max());
            // Compute the bounds of the photons
            for (unsigned int i = 0; i < kd_nodes_p.size(); ++i) {
                float3 position = (*kd_nodes_p[i]).position;
                bbox_min = fminf(bbox_min, position);
                bbox_max = fmaxf(bbox_max, position);
            }
            buildKDTree<KD_node>(kd_nodes_p.data(), 0, positions.size(), 0, nodes.data(), 0, split_choice, make_float3(0.0), make_float3(0));
        }

        struct KD_metric
        {
            float3 position;
            virtual float eval(int id) = 0;
        };
        struct closest_metric:KD_metric
        {
            float3* position_set;
            float min_distance;
            int closest_id;


            closest_metric() :min_distance(FLT_MAX) {}
            float eval(int id)
            {
                float3 diff = position_set[id] - position;
                float dis2 = dot(diff, diff);
                if (min_distance > dis2)closest_id = id;
                min_distance = min(min_distance, dis2);
                printf("eval with id %d %f\n", id, dis2);
                return min_distance;
            }
        };
        void iterate_closest_by_metric(KD_metric * metric)
        {
            int closest_index = 0;
            float closest_dis2 = metric->eval(nodes[0].index);
            std::vector<pair<int,float> >stack; 

            unsigned int stack_current = 0;
            unsigned int node = 0; // 0 is the start

//#define push_node(N) stack[stack_current++] = (N)
//#define pop_node()   stack[--stack_current]

            float block_min = 0.0; 
            stack.push_back(make_pair(0,0.0)); 

            do {
                //printf("loop1");
                if (closest_dis2 < block_min)
                {
                    node = stack.back().first;
                    stack.pop_back();
                    block_min = stack.back().second;
                    continue;
                }
                auto& currentVDirector = nodes[node];
                uint axis = currentVDirector.axis;
                if (!(axis & PPM_NULL)) {

                    float3 vd_position = currentVDirector.position;
                    float3 diff = metric->position - vd_position;
                    float distance2 = metric->eval(currentVDirector.index);

                    if (distance2 < closest_dis2) 
                    {
                        closest_dis2 = distance2; 
                    }

                    // Recurse
                    if (!(axis & PPM_LEAF)) {
                        float d;
                        if (axis & PPM_X) d = diff.x;
                        else if (axis & PPM_Y) d = diff.y;
                        else                      d = diff.z;

                        // Calculate the next child selector. 0 is left, 1 is right.
                        int selector = d < 0.0f ? 0 : 1;
                        if (d * d < closest_dis2) { 
                            stack.push_back(make_pair((node << 1) + 2 - selector, d * d));
                        }

                        node = (node << 1) + 1 + selector;
                    }
                    else {
                        //node = pop_node();
                        //block_min = dis_stack[stack_current];

                        node = stack.back().first;
                        stack.pop_back();
                        block_min = stack.back().second;
                    }
                }
                else {
                    node = stack.back().first;
                    stack.pop_back();
                    block_min = stack.back().second;
                }
            } while (node); 
        }
    };
};  

namespace classTree
{
    using std::default_random_engine;
    default_random_engine random_generator_tree;
    float rnd_float()
    {
        return float(random_generator_tree()) / random_generator_tree.max();
    }
    struct lightTree
    {
        std::vector<lightTreeNode> v;
        struct dir_aabb
        {
            float4 dir_cos;
            __host__ __device__
                dir_aabb make_union(dir_aabb& a)
            {
                float theta0 = acos(dot(make_float3(a.dir_cos), make_float3(dir_cos)));
                float theta1 = dir_cos.w;
                float theta2 = a.dir_cos.w;
                //float3 axis = cross(make_float3(a.dir_cos), make_float3(dir_cos));

                if (theta0 + theta1 <= theta2)
                {
                    return a;
                }
                if (theta0 + theta2 <= theta1)
                {
                    return *this;
                }
                float3 n_center;
                float n_bound;
                n_center = make_float3((a.dir_cos + dir_cos) / 2);

                n_bound = max(theta1, theta2) + theta0 / 2;
                n_bound = min(n_bound, M_PI);

                dir_aabb c;
                c.dir_cos = make_float4(n_center, n_bound);
                return c;
            }
            __host__ __device__
                float half_cos2()
            {
                return cos(dir_cos.w) * cos(dir_cos.w);
            }
            __host__ __device__
                float3 center()
            {
                return make_float3(dir_cos);
            }
        };
        struct aabb
        {
            float3 aa;
            float3 bb;
            __host__ __device__
                aabb make_union(aabb& a)
            {
                aabb c;
                c.aa = fminf(a.aa, aa);
                c.bb = fmaxf(a.bb, bb);
                return c;
            }
            __host__ __device__
                float3 diag()
            {
                return aa - bb;
            }
            float3 dir_mid()
            {
                return normalize(aa + bb);
            }
            float half_angle_cos()
            {
                float3 a = dir_mid();
                return abs(dot(normalize(aa), a));

            }
            float3 mid_pos()
            {
                return (aa + bb) / 2;
            }
        };
        struct lightTreeNode_construct :lightTreeNode
        {
            dir_aabb dir_box;
            dir_aabb nor_box;
            bool used;
            void init()
            {
                leaf = true;
                pos_box.aa = pos_box.bb = position;
                dir_box.dir_cos = make_float4(dir, 0);
                nor_box.dir_cos = make_float4(normal, 0);
                used = false;
            }
            lightTreeNode_construct merge(lightTreeNode_construct& b, float random)
            {
                lightTreeNode_construct c;
                float a_weight = ENERGY_WEIGHT(color);
                float b_weight = ENERGY_WEIGHT(b.color);
                float weight_index = (a_weight + b_weight) * random;
                const VPL* vpl_p = weight_index < a_weight ? reinterpret_cast<VPL*>(this) : reinterpret_cast<VPL*>(&b);
                *(VPL*)(&c) = *vpl_p;
                if (c.color.x == vpl_p->color.x)
                {
                   // printf("cast success\n");
                }
                else
                {
                    //printf("cast failed\n");
                }

                c.color = color + b.color;
                c.leaf = false;
                c.weight = weight + b.weight;
                c.pos_box = pos_box.make_union(b.pos_box);
                c.dir_box = dir_box.make_union(b.dir_box);
                c.nor_box = nor_box.make_union(b.nor_box);
                return c;

            }
            __host__ __device__
                float metric(float scene_diag2)
            {
                float3 dis = pos_box.diag();
                float d = dis.x * dis.x + dis.y * dis.y + dis.z * dis.z;
                float cos2 = nor_box.half_cos2();
                //printf("cos2 %f d %f color %f\n", cos2, d, ENERGY_WEIGHT(color));
                return ENERGY_WEIGHT(color) * d;
                //return ENERGY_WEIGHT(color) * (d + (1 - cos2) * scene_diag2);
            }
        };

        struct heap_pair
        {
            int src_id;
            int end_id;
            float metric;
            __host__ __device__
                bool operator<(const heap_pair& b)
            {
                return metric > b.metric;
            }
        };
        struct lightcut_metric: KD_tree_api::KD_metric
        {
            lightTreeNode_construct* nodes;
            int begin_id;
            int end_id;
            float current_metric;
            float diag2;
            int try_count;
            int try_limit;
            lightcut_metric(int begin_id,lightTreeNode_construct* nodes,float diag2):nodes(nodes),begin_id(begin_id),current_metric(FLT_MAX),diag2(diag2)
            {
                try_count = 0;
                try_limit = 200; 
            }
            float eval(int id)
            {
                if (nodes[id].used||id == begin_id)
                {
                    return FLT_MAX;
                }
                lightTreeNode_construct c = nodes[begin_id].merge(nodes[id],0);
                float metric = c.metric(diag2);
                if (metric < current_metric)
                {
                    current_metric = metric;
                    end_id = id;
                }
                try_count++;
                if (try_count == try_limit)
                {
                    if (nodes[end_id].used == true)
                    {
                        printf("issue catch");
                    }
                    return 0;
                    for (int tt = 0;; tt++)
                    {
                        if (nodes[tt].used == false && id != tt)
                        {
                            current_metric = FLT_MAX / 2;
                            end_id = tt;
                            return 0;
                        }
                    }
                }
                //return dot(c.pos_box.diag() , c.pos_box.diag());
                metric = dot(nodes[id].position - nodes[begin_id].position, nodes[id].position - nodes[begin_id].position);
                return metric;
                return current_metric / ENERGY_WEIGHT((nodes[begin_id].color));
            }
        };
        template<typename T>
        heap_pair get_closest_pair(int begin_id, std::vector<lightTreeNode_construct>& nodes, T& speedUpTree, float diag2)
        {
            heap_pair p;
            p.src_id = begin_id;
            p.metric = FLT_MAX;
             
            lightcut_metric mc(begin_id, nodes.data(), diag2);
            speedUpTree.iterate_closest_by_metric(&mc);

            p.metric = mc.current_metric;
            p.end_id = mc.end_id;

            return p;
        }
        struct KD_unit
        {
            float3 position;
            int label;
        };
        void kd_rebuild(KD_tree_api& kd_tree, std::vector<lightTreeNode_construct>& nodes)
        {
            std::vector<KD_unit> vpl_pos;

            for (int i = 0; i < nodes.size(); i++)
            {
                if (nodes[i].used == false)
                {
                    KD_unit u;
                    u.position = nodes[i].position;
                    u.label = i;
                    vpl_pos.push_back(u);
                }
            }
            kd_tree.build_from_labeled_position(vpl_pos);
        }
        void heap_rebuild(std::vector<heap_pair>& h, KD_tree_api& kd_tree, std::vector<lightTreeNode_construct>& nodes, float diag2)
        {
            h.clear();
            for (int i = 0; i < nodes.size(); i++)
            {
                if (nodes[i].used == false)
                {
                    heap_pair p = get_closest_pair(i, nodes, kd_tree, diag2); 
                    h.push_back(p);
                }
            }
            make_heap(h.begin(), h.end());

        }
        void light_tree_step(int i, int j, std::vector<lightTreeNode_construct>& tree)
        {
            lightTreeNode_construct ans = tree[i].merge(tree[j],rnd_float());
            printf("connect node %d and %d %d %f\n ", i, j, tree.size(), ENERGY_WEIGHT(ans.color));
            ans.leaf = false;
            ans.left_id = i;
            ans.right_id = j;
            tree[i].used = true;
            tree[j].used = true;
            ans.used = false;
            tree.push_back(ans);
            //printf("flux add %f %f %f\n", tree[i].flux, tree[j].flux, tree.back().flux);
        }
        lightTree(std::vector<VPL> vpls)
        {
            //get diag
            //build SpeedUpTree 
            std::vector<lightTreeNode_construct> temple_nodes;
            KD_tree_api kd_tree;
            for (int i = 0; i < vpls.size(); i++)
            {
                lightTreeNode_construct a;
                *(VPL*)(&a) = vpls[i];
                a.leaf = true;
                a.used = false;
                a.init();
                temple_nodes.push_back(a);
            }  
            kd_rebuild(kd_tree, temple_nodes);
            float3 diag = kd_tree.bbox_max - kd_tree.bbox_min;
            float diag2 = dot(diag, diag); 
            //build tree from leaf to root


            std::vector<heap_pair> h;
            //--for each node
            //----compute min distance in the grid it belongs to
            //----check if there is any potential block 
            //----compute min distance, insert to min heap

            heap_rebuild(h, kd_tree, temple_nodes, diag2);

            //--loop heap
            //----check valid
            //------if invalid, seach the closest node again
            //----if valid, merge node, tree grows, delete ordinal nodes, insert new node
            int success_count = 0;
            int next_rebuild_count = vpls.size() / 2;
            int seg_rebuild_count = vpls.size() / 4;
            while (true)
            {
                if (success_count == vpls.size() - 1)break;
                if (success_count == next_rebuild_count || success_count == vpls.size() - 2)
                {
                    kd_rebuild(kd_tree, temple_nodes);
                    heap_rebuild(h, kd_tree, temple_nodes, diag2);
                    next_rebuild_count += seg_rebuild_count;
                    seg_rebuild_count /= 2;
                    seg_rebuild_count = max(seg_rebuild_count, 1);
                }
                int src = h[0].src_id;
                int end = h[0].end_id;
                //printf("%d %d %d %d %d\n",h.size(), src, end,light_tree_construct.size(),vpls.size());
                if (temple_nodes[src].used == true)
                {
                    pop_heap(h.begin(), h.end());
                    h.pop_back();
                    continue;
                }
                else if (temple_nodes[end].used == true)
                {
                    printf("error2 %d-%d ",src,end);
                    heap_pair p = h[0];

                    pop_heap(h.begin(), h.end());
                    h.pop_back();

                    p = get_closest_pair(p.src_id, temple_nodes, kd_tree, diag2);

                    printf("------ %d", p.end_id);
                    h.push_back(p);
                    push_heap(h.begin(), h.end());
                    continue;
                }
                 
                //printf("loss %f\n", h[0].metric);
                light_tree_step(src, end, temple_nodes);
                 


                pop_heap(h.begin(), h.end());
                h.pop_back(); 
                success_count++;
            }
            for (int i = 0; i < temple_nodes.size(); i++)
            {
                v.push_back(temple_nodes[i]);
            }
        }
    };
}
#endif // !CLASSTREE_HOST
