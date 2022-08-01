#ifndef PG_HOST
#define PG_HOST
#include"PG_common.h"
#include"BDPT_STRUCT.h" 
#include<random>
#include<vector>
using namespace std;

float pg_quad_min_area = 0.000001;
float pg_rho = 0.01;
struct quad_tree_group
{
    vector<quad_tree_node> nodes;
    quad_tree_node& getNode(int id)
    {
        if (id >= nodes.size())
        {
            printf("invalid quad tree query: %d/%d", id, nodes.size());
            return nodes[0];
        }
        return nodes[id];
    }
    void training(PG_training_mat& mat, int id)
    { 
        nodes[id].lum += mat.lum;

        while (nodes[id].leaf == false)
        {
            id = nodes[id].traverse(mat.uv);
            nodes[id].lum += mat.lum;
        }
    }
    void clear()
    {
        for (auto p = nodes.begin(); p != nodes.end(); p++)
        {
            p->lum = 0;
        }
    }
    int tree_copy(int id)
    {
        int n_id = nodes.size();
        nodes.push_back(nodes[id]);
        if (!nodes[n_id].leaf)
        {
            for (int i = 0; i < 4; i++)
            {
                int n_branch = tree_copy(nodes[id].child[i]);
                nodes[n_id].child[i] = n_branch;

            }
        }
        return n_id;
    }
    bool valid_check()
    {
        for (int id = 0; id<nodes.size(); id++)
        {
            float lum_sum = 0.0f;
            if (nodes[id].leaf == false)
            {
                for (int i = 0; i < 4; i++)
                {
                    lum_sum += nodes[nodes[id].child[i]].lum;
                }
                if (lum_sum > 1.1 * nodes[id].lum || lum_sum < 0.9 * nodes[id].lum)
                {
                    printf("lum unbalance: %f %f %d\n", lum_sum, nodes[id].lum,id);
                    printf("%d %d %d %d\n", nodes[id].child[0], nodes[id].child[1], nodes[id].child[2], nodes[id].child[3]);
                    //return false;
                }
            }
        }
        return true;

    }
    void subdivide(int id, float lum_limit)
    { 
        if (nodes[id].area() < pg_quad_min_area)
            return;
        nodes[id].leaf = false;
        for (int dx = 0; dx <= 1; dx++)
            for (int dy = 0; dy <= 1; dy++)
            {
                int child_index = 2 * dy + dx;
                float2 box = (nodes[id].m_max - nodes[id].m_min) / 2;
                float2 min_base = nodes[id].m_min;
                if (dx == 1)
                {
                    min_base.x += box.x;
                }
                if (dy == 1)
                {
                    min_base.y += box.y;
                }
                quad_tree_node child(min_base, min_base + box);
                child.lum = nodes[id].lum / 4;
                //if (child_index == 0)   printf("%f limit %f ->%f %d\n", lum_limit, nodes[id].lum, child.lum, nodes.size());
                
                nodes[id].child[child_index] = nodes.size();
                nodes.push_back(child);
            }
        for (int i = 0; i < 4; i++)
        {
            int c_id = nodes[id].child[i]; 
            if (nodes[c_id].lum > lum_limit)
            {
                subdivide(c_id, lum_limit);
            }
        }
    }

    void subdivide_check(int id, float lum_limit)
    {   
        if (!nodes[id].leaf)
        {
            for (int i = 0; i < 4; i++)
            {
                int c_id = nodes[id].child[i];
                if (nodes[c_id].lum > lum_limit)
                {
                    subdivide_check(nodes[id].child[i], lum_limit);
                }
            }
        }
        else if (nodes[id].lum > lum_limit )
        {
            subdivide(id, lum_limit);
        }
    }
    void struct_transform(int header_id)
    {
        auto header = getNode(header_id);
        float lum_limit = header.lum * pg_rho;
        subdivide_check(header_id, lum_limit);
    }
    void lum_check_1(int id, float limit)
    {
        if (nodes[id].leaf == false && nodes[id].lum < limit)
        {
            nodes[id].leaf = true;
        }
        else if (nodes[id].leaf == false)
        {
            for (int i = 0; i < 4; i++)
            {
                lum_check_1(nodes[id].child[i], limit);
            }
        }
    }
    void lum_check_0(int header_id)
    {
        if (nodes[header_id].leaf == false)
        {
            for (int i = 0; i < 4; i++)
            {
                lum_check_1(nodes[header_id].child[i], nodes[header_id].lum * pg_rho);
            }
        }
    }
};
struct Spatio_tree
{
    vector<Spatio_tree_node> nodes;
    quad_tree_group* q_tree_group;
    Spatio_tree_node& getNode(int index)
    {
        if (index >= nodes.size())
        {
            printf("invalid spatio tree query: %d/%d", index, nodes.size());
        }
        return nodes[index];
    }
    Spatio_tree_node& getHeader()
    {
        return nodes[0];
    } 
    Spatio_tree_node& training(PG_training_mat& mat)
    { 
        int c_id = 0;
        nodes[c_id].count++;
        while (nodes[c_id].leaf == false)
        {
            c_id = nodes[c_id].traverse(mat.position);
            nodes[c_id].count++;
        }
        return nodes[c_id];

    }
    Spatio_tree_node& traverse(PG_training_mat& mat)
    {
        int c_id = 0;
        while (nodes[c_id].leaf == false)
        {
            c_id = nodes[c_id].traverse(mat.position);
        }
        return nodes[c_id];

    }
    void clear_count()
    {
        for (auto p = nodes.begin(); p != nodes.end(); p++)
        {
            p->count = 0;
        }
    }
    void subdivide(int index)
    { 
        nodes[index].leaf = false;
        for (int dx = 0; dx <= 1; dx++)
            for (int dy = 0; dy <= 1; dy++)
                for (int dz = 0; dz <= 1; dz++)
                {
                    int child_index = 4 * dz + 2 * dy + dx;
                    float3 box = (nodes[index].m_max - nodes[index].m_min) / 2;
                    float3 min_base = nodes[index].m_min;
                    if (dx == 1)
                    {
                        min_base.x += box.x;
                    }
                    if (dy == 1)
                    {
                        min_base.y += box.y;
                    }
                    if (dz == 1)
                    {
                        min_base.z += box.z;
                    }
                    Spatio_tree_node child(min_base, min_base + box);
                    nodes[index].child[child_index] = nodes.size();
                    child.quad_tree_id = q_tree_group->tree_copy(nodes[index].quad_tree_id);
                    nodes.push_back(child);
                }
    }
    void subdivide_check(int index, int count_limit)
    { 
        if (!nodes[index].leaf)
        {
            for (int i = 0; i < 8; i++)
            {
                subdivide_check(nodes[index].child[i], count_limit);
            }
        }
        else if (nodes[index].count > count_limit)
        {
            subdivide(index);
        }
    }

    void quad_struct_transform()
    {
        for (auto p = nodes.begin(); p != nodes.end(); p++)
        { 
            if (p->leaf == true && q_tree_group->nodes[p->quad_tree_id].lum>0.001)
            {  
                q_tree_group->struct_transform(p->quad_tree_id);
                q_tree_group->lum_check_0(p->quad_tree_id);
            }
        }
    }
};

struct QuadArea_tree
{
    vector<quad_tree_node> nodes;
    quad_tree_group* q_tree_group;
    quad_tree_node& getNode(int index)
    {
        if (index >= nodes.size())
        {
            printf("invalid spatio tree query: %d/%d", index, nodes.size());
        }
        return nodes[index];
    }
    quad_tree_node& getHeader()
    {
        return nodes[0];
    }
    quad_tree_node& training(PG_training_mat& mat)
    {
        int c_id = mat.light_id;
        nodes[c_id].count++;
        nodes[c_id].lum += mat.lum;
        while (nodes[c_id].leaf == false)
        {
            c_id = nodes[c_id].traverse(mat.uv_light);
            nodes[c_id].count++;
            nodes[c_id].lum += mat.lum;
        }
        q_tree_group->training(mat,nodes[c_id].quad_tree_id);
        return nodes[c_id];

    }
    quad_tree_node& traverse(PG_training_mat& mat)
    {
        int c_id = 0;
        while (nodes[c_id].leaf == false)
        {
            c_id = nodes[c_id].traverse(mat.uv_light);
        }
        return nodes[c_id];

    }
    void clear_count()
    {
        for (auto p = nodes.begin(); p != nodes.end(); p++)
        {
            p->count = 0;
            p->lum = 0;
        }
        q_tree_group->clear();
    }
    void subdivide(int index)
    {
        nodes[index].leaf = false;
        for (int dx = 0; dx <= 1; dx++)
            for (int dy = 0; dy <= 1; dy++) 
                {
                    int child_index = 2 * dy + dx;
                    float2 box = (nodes[index].m_max - nodes[index].m_min) / 2;
                    float2 min_base = nodes[index].m_min;
                    if (dx == 1)
                    {
                        min_base.x += box.x;
                    }
                    if (dy == 1)
                    {
                        min_base.y += box.y;
                    } 
                    quad_tree_node child(min_base, min_base + box);
                    child.lum = nodes[index].lum / 4;
                    nodes[index].child[child_index] = nodes.size();
                    child.quad_tree_id = q_tree_group->tree_copy(nodes[index].quad_tree_id);
                    nodes.push_back(child);
                }
    }
    void subdivide_check(int index, int count_limit)
    {
        if (!nodes[index].leaf)
        {
            for (int i = 0; i < 4; i++)
            {
                subdivide_check(nodes[index].child[i], count_limit);
            }
        }
        else if (nodes[index].count > count_limit)
        {
            subdivide(index);
        }
    }

    void subdivide_check(int count_limit)
    {
        for (int i = 0; i < MAX_LIGHT; i++)
        {
            subdivide_check(i, count_limit);
        }
    }
    void quad_struct_transform()
    {
        for (auto p = nodes.begin(); p != nodes.end(); p++)
        {
            if (p->leaf == true && q_tree_group->nodes[p->quad_tree_id].lum > 0.001)
            {
                q_tree_group->struct_transform(p->quad_tree_id);
                q_tree_group->lum_check_0(p->quad_tree_id);
            }
        }
    }
};
struct SD_PGTrainer
{
    Buffer training_mats_buffer;
    Buffer spatio_tree_buffer;
    Buffer quad_tree_buffer;
    Buffer spatio_tree_buffer_light;
    Buffer quad_tree_buffer_light;
    Buffer L_trees_buffer;
    Buffer L_tree_dir_buffer;
    Spatio_tree s_tree;
    Spatio_tree s_tree_light;
    quad_tree_group q_tree_group;
    quad_tree_group q_tree_group_light;

    QuadArea_tree L_tree;
    quad_tree_group L_tree_d;

    Spatio_tree *s_tree_p;
    quad_tree_group *q_tree_group_p;
    vector<PG_training_mat> mats;
    int path_k;
    int mats_pathStep = 10;
    int default_mat_paths_num = 1000000;
    int max_process_mats_num =  10000000;

    void init(Context& context)
    {
        s_tree_p = &s_tree;
        q_tree_group_p = &q_tree_group;
        path_k = 1;

        training_mats_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, default_mat_paths_num * mats_pathStep);
        training_mats_buffer->setElementSize(sizeof(PG_training_mat));
        context["PG_mats"]->setBuffer(training_mats_buffer);

        spatio_tree_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        spatio_tree_buffer->setElementSize(sizeof(Spatio_tree_node));
        context["spatio_tree_nodes"]->setBuffer(spatio_tree_buffer);


        spatio_tree_buffer_light = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        spatio_tree_buffer_light->setElementSize(sizeof(Spatio_tree_node));
        context["spatio_tree_nodes_light"]->setBuffer(spatio_tree_buffer_light);

        quad_tree_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        quad_tree_buffer->setElementSize(sizeof(quad_tree_node));
        context["quad_tree_nodes"]->setBuffer(quad_tree_buffer);


        L_trees_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        L_trees_buffer->setElementSize(sizeof(quad_tree_node));
        context["L_trees"]->setBuffer(L_trees_buffer);


        L_tree_dir_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        L_tree_dir_buffer->setElementSize(sizeof(quad_tree_node));
        context["L_tree_dir"]->setBuffer(L_tree_dir_buffer);

        quad_tree_buffer_light = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
        quad_tree_buffer_light->setElementSize(sizeof(quad_tree_node));
        context["quad_tree_nodes_light"]->setBuffer(quad_tree_buffer_light);

        context["PG_max_step"]->setInt(mats_pathStep);
        s_tree.q_tree_group = &q_tree_group;
        s_tree_light.q_tree_group = &q_tree_group_light;

        s_tree.nodes.push_back(Spatio_tree_node(context["min_box"]->getFloat3(), context["max_box"]->getFloat3()));
        s_tree_light.nodes.push_back(Spatio_tree_node(context["min_box"]->getFloat3(), context["max_box"]->getFloat3()));
        q_tree_group.nodes.push_back(quad_tree_node(make_float2(0), make_float2(1)));
        q_tree_group_light.nodes.push_back(quad_tree_node(make_float2(0), make_float2(1)));
        
        for (int i = 0; i < MAX_LIGHT; i++)
        {
            L_tree.nodes.push_back(quad_tree_node(make_float2(0), make_float2(1)));
            L_tree_d.nodes.push_back(quad_tree_node(make_float2(0), make_float2(1)));
            L_tree.nodes[L_tree.nodes.size() - 1].quad_tree_id = i;
        }
        L_tree.q_tree_group = &L_tree_d;
        //s_tree.nodes.reserve(100000);
        //q_tree_group.nodes.reserve(100000);


        spatio_tree_buffer->setSize(s_tree.nodes.size());
        spatio_tree_buffer_light->setSize(s_tree_light.nodes.size());
        quad_tree_buffer->setSize(q_tree_group.nodes.size());
        quad_tree_buffer_light->setSize(q_tree_group_light.nodes.size());
        auto s_p = reinterpret_cast<Spatio_tree_node*>(spatio_tree_buffer->map());
        auto s_p_light = reinterpret_cast<Spatio_tree_node*>(spatio_tree_buffer_light->map());
        auto d_p = reinterpret_cast<quad_tree_node*>(quad_tree_buffer->map());
        auto d_p_light = reinterpret_cast<quad_tree_node*>(quad_tree_buffer_light->map());
         


        memcpy(s_p, &(*s_tree.nodes.begin()), sizeof(Spatio_tree_node) * s_tree.nodes.size());
        memcpy(s_p_light, &(*s_tree_light.nodes.begin()), sizeof(Spatio_tree_node) * s_tree_light.nodes.size());
        memcpy(d_p, &(*q_tree_group.nodes.begin()), sizeof(quad_tree_node) * q_tree_group.nodes.size());
        memcpy(d_p_light, &(*q_tree_group_light.nodes.begin()), sizeof(quad_tree_node) * q_tree_group_light.nodes.size());
        //unmap buffer 
        spatio_tree_buffer->unmap();
        spatio_tree_buffer_light->unmap();
        quad_tree_buffer->unmap();
        quad_tree_buffer_light->unmap();
    }

    int LT_trace(Context& context, int core = PCPT_CORE_NUM, int core_size = LIGHT_VERTEX_PER_CORE, bool M_FIX = false)
    {
        static int LVC_frame = 0;
        if (LVC_frame == 0)
        {
            context["LVC_frame"]->setInt(0);
        }
        LVC_frame = context["LVC_frame"]->getInt();
        context["LVC_frame"]->setInt(++LVC_frame);
        //int origin_M = context["M_FIX"]->getInt();

        context["LT_CORE_NUM"]->setInt(core);
        context["LT_CORE_SIZE"]->setInt(core_size);
        context["M_FIX"]->setInt(M_FIX == true ? 1 : 0);
        context->launch(lightBufferGen, core, 1);

        //context["M_FIX"]->setInt(origin_M);
        return core_size * core;
    }

    float2 dir2uv(float3 dir)
    {
        float2 uv;
        float theta = atan2f(dir.x, dir.z);
        float phi = M_PIf * 0.5f - acosf(dir.y);
        float u = (theta + M_PIf) * (0.5f * M_1_PIf);
        float v = 0.5f * (1.0f + sin(phi));
        uv = make_float2(u, v);

        return uv;
    }
    void get_mats_light(Context& context, int num, bool s_tree_guide = false, bool d_tree_guide = true)
    {
        mats.clear();
        while (mats.size() < num)
        {
            int trace_num = LT_trace(context);

            auto p = reinterpret_cast<RAWVertex*> (context["raw_LVC"]->getBuffer()->map());
            for (int i = 0; i < trace_num; i++)
            {
                if (p[i].valid == true && p[i].v.type != DIRECTION && p[i].v.depth >= 1)
                {
                    PG_training_mat mat;
                    mat.valid = true;
                    mat.lum = luminance( p[i].v.flux / p[i].v.pdf);
                    mat.position = p[i].v.position;
                    mat.uv = dir2uv(normalize(p[i].v.lastPosition - p[i].v.position));
                    mat.lum /= 100;
                    if(mat.lum<1000000000 && mat.lum > 0 && !isinf(mat.lum) && !isnan(mat.lum))
                        mats.push_back(mat);
                }
                if (mats.size() >= num)
                {
                    break;
                }
            }
            context["raw_LVC"]->getBuffer()->unmap();

            if (mats.size() > max_process_mats_num)
            {
                num -= mats.size(); 
                if(d_tree_guide)
                    sd_light_field_construct();
                if (s_tree_guide)
                    sd_struct_train();
                mats.clear();
            }
        }
    }
    void sd_struct_train()
    {
        for (int i = 0; i < mats.size(); i++)
        {
            if (mats[i].valid)
            { 
                s_tree_p->training(mats[i]);
            }
        }
    }
    void sd_light_field_construct()
    {
        for (int i = 0; i < mats.size(); i++)
        {
            if (mats[i].valid)
            { 
                auto s_node = s_tree_p->traverse(mats[i]);
                auto d_id = s_node.quad_tree_id;
                q_tree_group_p->training(mats[i], d_id);
            }
        }
    }
    void get_mats(Context& context,int num, bool s_tree_guide = true, bool d_tree_guide = true,bool light_source_guide = true)
    {
        mats.clear();
        while (mats.size() < num)
        {
            static int pg_seed = 1;
            context["PG_seed"]->setInt(pg_seed++); 
            context->launch(PGTrainingProg, default_mat_paths_num, 1);
            auto mats_p = reinterpret_cast<PG_training_mat*>(training_mats_buffer->map());
            int current_paths_A = 0;
            int current_paths_B = 0;
            for (int i = 0; i < default_mat_paths_num * mats_pathStep; i++)
            {
                if (i % 10 == 0)
                    current_paths_A++;
                if (light_source_guide && mats_p[i].light_source)
                {
                    L_tree.training(mats_p[i]);
                    mats_p[i].light_source = false;
                    continue;
                }
                if (mats_p[i].valid)
                {
                    mats.push_back(mats_p[i]);
                    if (i % 10 == 1)
                        current_paths_B++;
                    mats_p[i].valid = false;
                }
                if (mats.size() >= num)
                {
                    break;
                }               
            }
            //printf("%d / %d\n", current_paths_B, current_paths_A);
            current_paths_A = 0;
            current_paths_B = 0;
            training_mats_buffer->unmap();
 
            if (mats.size() > max_process_mats_num)
            {
                num -= mats.size();
                if (d_tree_guide)
                    sd_light_field_construct();
                if (s_tree_guide)
                    sd_struct_train();
                mats.clear();
            }
        }
    }
    void get_mats_importon(Context& context, int num, bool s_tree_guide = true, bool d_tree_guide = true)
    {
        context["PG_inverse_mat"]->setInt(1);
        get_mats(context, num, s_tree_guide, d_tree_guide,false);
        context["PG_inverse_mat"]->setInt(0);

    }
    bool build_tree(Context& context)
    { 
#ifndef PG_ENABLE
        return false;
#endif

        int max_k = 12;
        if (path_k >= max_k)
        {
            return false;
        }
        int k_2 = pow(2, path_k);
        int path_num = 12000 * k_2;
        int div_limit = 24000;
        //lunch_for_training materials 

        s_tree_p = &s_tree;
        q_tree_group_p = &q_tree_group;
        //traverse mats for training
        L_tree.clear_count();
        s_tree_p->clear_count();
        q_tree_group_p->clear();
        get_mats(context, path_num);
        sd_struct_train();
        //get_mats_light(context, path_num);
        sd_light_field_construct();


        int valid_mats_num = 0;
        float lum_sum = 0;
        for (int i = 0; i < mats.size(); i++)
        {
            if (mats[i].valid)
            {
                valid_mats_num++;
                lum_sum += mats[i].lum;
            }
        }
        printf("%d/%d valid pgMats %f luminance\n", valid_mats_num, path_num, lum_sum); 
        s_tree_p->quad_struct_transform();
        s_tree_p->subdivide_check(0, div_limit);

        if(false)
        {
            s_tree_p = &s_tree_light;
            q_tree_group_p = &q_tree_group_light;
            //traverse mats for training
            s_tree_p->clear_count();
            q_tree_group_p->clear();
            //get_mats_light(context, path_num, true, false);
            //get_mats_importon(context, path_num, false, true);
            get_mats_importon(context, path_num, true, true);
            sd_struct_train();
            sd_light_field_construct();
            //get_mats_light(context, path_num);

            s_tree_p->quad_struct_transform();
            s_tree_p->subdivide_check(0, div_limit);
            context["pg_lightSide_enable"]->setInt(1);
            q_tree_group_p->valid_check();
            
        }

        //light_source_guide
        if (false)
        {
            L_tree.quad_struct_transform();
            L_tree.subdivide_check(div_limit);
            L_trees_buffer->setSize(L_tree.nodes.size());
            L_tree_dir_buffer ->setSize(L_tree_d.nodes.size());
            auto p1 = reinterpret_cast<quad_tree_node*>(L_trees_buffer->map());
            auto p2 = reinterpret_cast<quad_tree_node*>(L_tree_dir_buffer->map());
            memcpy(p1, &(*L_tree.nodes.begin()), sizeof(quad_tree_node) * L_tree.nodes.size());
            memcpy(p2, &(*L_tree_d.nodes.begin()), sizeof(quad_tree_node) * L_tree_d.nodes.size());
            L_trees_buffer->unmap();
            L_tree_dir_buffer->unmap();
            context["pg_lightSource_enable"]->setInt(1);

            printf("lightSource--train %d complete, s_tree now has %d nodes, and d_tree now has %d nodes\n", path_k, L_tree.nodes.size(), L_tree_d.nodes.size()); 
        }
        spatio_tree_buffer->setSize(s_tree.nodes.size());
        spatio_tree_buffer_light->setSize(s_tree_light.nodes.size());
        quad_tree_buffer->setSize(q_tree_group.nodes.size());
        quad_tree_buffer_light->setSize(q_tree_group_light.nodes.size());
        auto s_p = reinterpret_cast<Spatio_tree_node*>(spatio_tree_buffer->map());
        auto s_p_light = reinterpret_cast<Spatio_tree_node*>(spatio_tree_buffer_light->map());
        auto d_p = reinterpret_cast<quad_tree_node*>(quad_tree_buffer->map());
        auto d_p_light = reinterpret_cast<quad_tree_node*>(quad_tree_buffer_light->map());

        q_tree_group_p->valid_check();


        memcpy(s_p, &(*s_tree.nodes.begin()), sizeof(Spatio_tree_node) * s_tree.nodes.size());
        memcpy(s_p_light, &(*s_tree_light.nodes.begin()), sizeof(Spatio_tree_node) * s_tree_light.nodes.size());
        memcpy(d_p, &(*q_tree_group.nodes.begin()), sizeof(quad_tree_node) * q_tree_group.nodes.size());
        memcpy(d_p_light, &(*q_tree_group_light.nodes.begin()), sizeof(quad_tree_node) * q_tree_group_light.nodes.size());
        //unmap buffer 
        spatio_tree_buffer->unmap();
        spatio_tree_buffer_light->unmap();
        quad_tree_buffer->unmap();
        quad_tree_buffer_light->unmap();
        printf("train %d complete, s_tree now has %d nodes, and d_tree now has %d nodes\n", path_k, s_tree.nodes.size(), q_tree_group.nodes.size());
        printf("light--train %d complete, s_tree now has %d nodes, and d_tree now has %d nodes\n", path_k, s_tree_light.nodes.size(), q_tree_group_light.nodes.size());
        context["pg_enable"]->setInt(1);
        path_k++;
        path_k = min(path_k, max_k);

        //context["pg_enable"]->setInt(0);
        return true;
    }
};

SD_PGTrainer PGTrainer_api;
#endif // !PG_HOST
