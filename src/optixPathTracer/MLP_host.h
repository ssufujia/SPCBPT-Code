#ifndef MLP_HOST_H
#define MLP_HOST_H

#include"MLP_common.h"
#include"BDPT_STRUCT.h"

#include<random>
#include<vector>
#include <sutil.h>
#include"hello_cuda.h"
#include<fstream>
using namespace optix;

namespace MLP
{
    using std::default_random_engine;

    default_random_engine random_generator;
    float rnd_float()
    {
        return float(random_generator()) / random_generator.max();
    }
    struct MLP_HOST
    {
        int batch_size;
        int network_num;
        int sample_sum;
        int batch_num;
        MLP_HOST()
        {
            batch_size = 10000;
            network_num = 1000;

            sample_sum = 2000000;
            batch_num = sample_sum / batch_size;
        }

        void init(Context& context)
        {
            //optimal_E_train(3, 5, 3, 3);

            auto weight_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, SUBSPACE_NUM);
            weight_buffer->setElementSize(sizeof(MLP_network));
            context["MLP_buffer"]->setBuffer(weight_buffer);

            auto gradient_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, SUBSPACE_NUM);
            gradient_buffer->setElementSize(sizeof(MLP_network));
            context["gradient_buffer"]->setBuffer(gradient_buffer);

            auto feed_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, sample_sum);
            feed_buffer->setElementSize(sizeof(feed_token));
            context["feed_buffer"]->setBuffer(feed_buffer);

            auto BP_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_USER, batch_size);
            BP_buffer->setElementSize(sizeof(BP_token));
            context["BP_buffer"]->setBuffer(BP_buffer);

            auto L1_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT2, batch_num, network_num);
            context["MLP_index_L1_buffer"]->setBuffer(L1_buffer);

            auto L2_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, sample_sum);
            context["MLP_index_L2_buffer"]->setBuffer(L2_buffer);


            auto CC_lookup_table = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, SUBSPACE_NUM, 32);
            context["CC_lookup_table"]->setBuffer(CC_lookup_table);

            auto E_table = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, SUBSPACE_NUM, SUBSPACE_NUM);
            context["E_table"]->setBuffer(E_table); 

        }
        void gen_fake_input(Context& context)
        {

            {
                auto p = reinterpret_cast<MLP_network*> (context["MLP_buffer"]->getBuffer()->map());
                for (int i = 0; i < SUBSPACE_NUM; i++)
                {
                    auto pp = reinterpret_cast<float*>(&p[i]);
                    for (int j = 0; j < sizeof(MLP_network) / sizeof(float); j++)
                    {
                        pp[j] = random_generator();
                    }
                }
                context["MLP_buffer"]->getBuffer()->unmap();
            }//weight buffer

            //feed_token
            {
                auto p = reinterpret_cast<feed_token*>(context["feed_buffer"]->getBuffer()->map());
                for (int i = 0; i < batch_size; i++)
                {
                    p[i].position = make_float3(rnd_float(), rnd_float(), rnd_float());
                    p[i].pdf0 = rnd_float();
                    p[i].ratio = rnd_float();
                    p[i].f_square = rnd_float();
                    p[i].grid_label = random_generator() % SUBSPACE_NUM;

                    p[i].dual_label = random_generator() % SUBSPACE_NUM;
                }
                context["feed_buffer"]->getBuffer()->unmap();
            }

            //lookup table
            {
                auto p = reinterpret_cast<int*>(context["CC_lookup_table"]->getBuffer()->map());
                for (int i = 0; i < 32 * SUBSPACE_NUM; i++)
                {
                    p[i] = random_generator() % SUBSPACE_NUM;
                }
                context["CC_lookup_table"]->getBuffer()->unmap();
            }

            //E_table
            {
                auto p = reinterpret_cast<float*>(context["E_table"]->getBuffer()->map());
                for (int i = 0; i < SUBSPACE_NUM * SUBSPACE_NUM; i++)
                {
                    p[i] = rnd_float();
                }
            }
            context["E_table"]->getBuffer()->unmap();
        }

        void arrange_index(Context& context)
        { 
            auto p = reinterpret_cast<feed_token*>(context["feed_buffer"]->getBuffer()->map());
            auto L1 = reinterpret_cast<int2*>(context["MLP_index_L1_buffer"]->getBuffer()->map());
            auto L2 = reinterpret_cast<int*>(context["MLP_index_L2_buffer"]->getBuffer()->map());

            int* class_sum = new int[network_num];
            int* class_acc = new int[network_num];
            int* class_cur = new int[network_num];
            for (int i = 0; i < network_num; i++)
            {
                class_sum[i] = 0; 
                class_acc[i] = 0;
                class_cur[i] = 0;
            }
            for (int i = 0; i < sample_sum; i++)class_sum[p[i].grid_label] += 1;
            for (int i = 1; i < network_num; i++)class_acc[i] = class_acc[i - 1] + class_sum[i];

            for (int i = 0; i < sample_sum; i++)
            {
                int label = p[i].grid_label;
                int base = class_acc[label];
                int bias = class_cur[label];
                int id = base + bias;
                L2[id] = i;
                class_cur[label]++;
            }
            for (int i = 0; i < network_num; i++)
            {
                int num = class_acc[i]; 
                for (int j = 0; j < batch_num; j++)
                {
                    L1[j * network_num + i].x = num;
                    while (L2[num] < (j + 1) * batch_size && num<class_acc[i+1])
                    {
                        num++; 
                    }
                    L1[j * network_num + i].y = num;

                }
            } 
             
            for (int i = 0; i < sample_sum; i++)L2[i] = L2[i] % batch_size;

            delete[] class_sum;
            delete[] class_acc;
            delete[] class_cur;
            context["feed_buffer"]->getBuffer()->unmap(); 
            context["MLP_index_L1_buffer"]->getBuffer()->unmap(); 
            context["MLP_index_L2_buffer"]->getBuffer()->unmap();
        } 
        void test_run(Context& context)
        {
            float t;
            //int dev = 0;
            //cudaDeviceProp devProp;
            //RTCHECK(cudaGetDeviceProperties(&devProp, dev));
 
                // hello from cpu
 
            //cudaMalloc
            t = sutil::currentTime();
            //hello_print();
            //batch_gemm_test();
            //thrust_bp_test_run();
            //optimal_E_train(15, 15, 3, 3);
            //printf("C %f\n", sutil::currentTime() - t);
            return;
            gen_fake_input(context);
            arrange_index(context);
            printf("A");
            context["batch_size"]->setInt(batch_size);
            context->launch(forwardProg,batch_size,1);
            for (int i = 0; i < batch_num * 1; i++)
            {
                context["batch_id"]->setInt(i % batch_num);
                context->launch(forwardProg, batch_size, 1);
                //context->launch(backwardProg, network_num, 1);
            }
            printf("B");
            t = sutil::currentTime();
            for (int i = 0; i < batch_num * 20; i++)
            {
                context["batch_id"]->setInt(i % batch_num);
                context->launch(forwardProg, batch_size, 1);
                //context->launch(backwardProg, network_num, 1);
            }
            printf("C %f\n", sutil::currentTime() - t);
            //auto p = reinterpret_cast<BP_token*>(context["BP_buffer"]->getBuffer()->map());
            //printf("%f\n", p[12887].feature_0[1]);
            //printf("D %f\n", sutil::currentTime() - t);

            //context["BP_buffer"]->getBuffer()->unmap();
        }
    };
}
MLP::MLP_HOST mlp_api;

namespace MLP
{
    struct MLP_DEBUG
    {
        std::vector<float> f_square; //size N
        std::vector<float> pdf_0;    //size N

        std::vector<float> pdf_peak; //size M
        std::vector<int>   label_E;   //size M
        std::vector<int>   label_P;  //size M
        std::vector<int>   label_light;  //size M
        std::vector<int>   label_eye;  //size M
        std::vector<int>   P2N_ind;  //size N P2N_ind[i] record the begin index of path i
        std::vector<float> ans;


        std::vector<int> close_set;
        std::vector<float> positions; //size M * 3/5
        //std::vector<float> ref_E;
        int close_dim;
        void load_data_old()
        {

            std::ifstream inFile;
            inFile.open("./exp_record/OPTP_data/optimal_E_dataset.txt");
            int N, D_A, D_B;
            inFile >> N >> D_A >> D_B;
            for (int pp = 0; pp < N; pp++)
            {
                int MM;
                float t_f_square, t_pdf_0;
                inFile >> MM  >> t_pdf_0 >> t_f_square;
                f_square.push_back(t_f_square);
                pdf_0.push_back(t_pdf_0 / 3.0);
                P2N_ind.push_back(label_P.size());
                for (int i = 0; i < MM; i++) label_P.push_back(pp);
                for (int i = 0; i < MM; i++)
                {
                    int label;
                    float peak;
                    inFile >> label >> peak;
                    pdf_peak.push_back(peak);
                    label_E.push_back(label);
                }
            } 
            inFile.close();
            printf("dataLoad_complete\n");
        }
        void load_data()
        {
            close_dim = 32;
            std::ifstream inFile;
            inFile.open("./exp_record/OPTP_data/eye_MLP_dataset.txt");
            int N, D_A, D_B;
            inFile >> N >> D_A >> D_B;
            for (int i = 0; i < D_B; i++)
            {
                for (int j = 0; j < close_dim; j++)
                {
                    int c;
                    inFile >> c;
                    close_set.push_back(c);

                }
            }
            float f_max[3] = { FLT_MIN,FLT_MIN,FLT_MIN };
            float f_min[3] = { FLT_MAX,FLT_MAX,FLT_MAX };
            for (int pp = 0; pp < N; pp++)
            {
                int MM;
                float t_f_square, t_pdf_0;
                inFile >> MM >> t_pdf_0 >> t_f_square;
                f_square.push_back(t_f_square);
                pdf_0.push_back(t_pdf_0 / 3.0);
                P2N_ind.push_back(label_P.size());
                for (int i = 0; i < MM; i++) label_P.push_back(pp);
                for (int i = 0; i < MM; i++)
                {
                    int label_light_,label_eye_;
                    float peak;
                    float x[3];
                    
                    inFile >> x[0] >> x[1] >> x[2] >> label_light_ >> label_eye_ >> peak;
                    
                    for (int j = 0; j < 3; j++)
                    {
                        f_max[j] = max(x[j], f_max[j]);
                        f_min[j] = min(x[j], f_min[j]);
                    }
                    positions.push_back(x[0]);
                    positions.push_back(x[1]);
                    positions.push_back(x[2]);
                    pdf_peak.push_back(peak);
                    label_E.push_back(label_eye_ * D_A + label_light_);
                    label_eye.push_back(label_eye_);
                    label_light.push_back(label_light_);
                }
            }
            for (int i = 0; i < positions.size(); i++)
            {
                float t = (positions[i] - f_min[i % 3]) / (f_max[i % 3] - f_min[i % 3]);
                positions[i] = (t - 0.5) * 2;
            }
            //exit(0);
            P2N_ind.push_back(label_P.size());

            inFile.close();
            printf("dataLoad_complete\n");
        }
        void test_run()
        {
            load_data();
            //learn_by_data(ans, f_square, pdf_0, pdf_peak, label_E, label_P, P2N_ind);
            learn_by_position(ans, f_square, pdf_0, pdf_peak, positions, label_light, label_eye, label_P, P2N_ind,close_set);
            
        }
    };
}

#include"ZGC.h"
#include"classTree_host.h" 
#include <thrust/copy.h> 
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//#include <thrust/device_ptr.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include<thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include "hello_cuda.h"
template<typename T>
struct buffer_pointer
{
    T* p;
    size_t size;
    Buffer b;
    bool map;
    buffer_pointer(Context& context, std::string buffer_name)
    {
        b = context[buffer_name]->getBuffer();
        p = reinterpret_cast<T*>(b->map());
        b->getSize(size);
        map = true;
    }
    void unmap()
    {
        b->unmap();
        map = false;
    }
    ~buffer_pointer()
    {
        if(map)
            unmap();
    }
    T& operator[](int i)
    {
        return p[i];
    }
};
namespace HS_algorithm
{
    std::vector<int> stra_hs(std::vector<gamma_point> points_p, float* weight, int sample_size, int cluster_num_sum,std::vector<gamma_point> &accm_centers)
    { 
        int origin_cluster_num = accm_centers.size();
        float mean_weight = 0.0;
        for (int i = 0; i < sample_size; i++)
        { 
            mean_weight += weight[i] / sample_size;
        }
        float var_weight = 0.0;
        for (int i = 0; i < sample_size; i++)
        {
            var_weight += (mean_weight - weight[i]) * (mean_weight - weight[i]) / (sample_size - 1);
        }
        float sigma = sqrt(var_weight);
        printf("mean %f var %f sigma%f\n", mean_weight, var_weight, sigma);
        //if (3 * sigma > mean_weight)sigma = mean_weight / 3;

        //vector<HS_algorithm::gamma_point> accm_centers(0);
        std::vector<HS_algorithm::gamma_point> v_stra[6];
        float layer_weight[6] = { 0 };
        float layer_sum = 0;
        for (int i = 0; i < sample_size; i++)
        {
            float judge_line = weight[i];
            int target_id = 5 - int((judge_line - mean_weight + 3 * sigma) / sigma);
            target_id = clamp(target_id, 0, 5);
            v_stra[target_id].push_back(points_p[i]);
            layer_weight[target_id] += judge_line;
            layer_sum += judge_line;
        }

        for (int i = 0; i < 6; i++)
        {
            int cluster_num = layer_weight[i] / layer_sum * cluster_num_sum;
            vector<int> n_ids(0);
            if (cluster_num <= 0)continue;
            for (int j = 0; j < accm_centers.size(); j++)
            {
                n_ids.push_back(v_stra[i].size());
                v_stra[i].push_back(accm_centers[j]);
            }
            cluster_num = min(cluster_num, v_stra[i].size());
            printf("input %d cluster in layer %d\n", n_ids.size(), i);
            n_ids = HS_algorithm::Hochbaum_Shmonys(v_stra[i].data(), n_ids.data(), v_stra[i].size(), n_ids.size(), min(origin_cluster_num + cluster_num_sum, n_ids.size() + cluster_num));
            printf("get %d cluster in layer %d\n", n_ids.size(), i);
            for (int j = accm_centers.size(); j < n_ids.size(); j++)
            {
                accm_centers.push_back(v_stra[i][j]);
            }
        }

        if (false)//sort
        {
            for (int i = 0; i < accm_centers.size(); i++)
            {
                for (int j = 0; j < accm_centers.size() - i - 1; j++)
                {
                    float weightA = length(accm_centers[j].position);
                    float weightB = length(accm_centers[j + 1].position);
                    if (weightA < weightB)
                    {
                        auto t = accm_centers[j];
                        accm_centers[j] = accm_centers[j + 1];
                        accm_centers[j + 1] = t;
                    }
                }
            } 
        }
        vector<int> final_center_ids(0);
        for (int i = 0; i < accm_centers.size(); i++)
        {
            final_center_ids.push_back(points_p.size());
            points_p.push_back(accm_centers[i]);
        }
        return HS_algorithm::label_with_center(points_p.data(), final_center_ids.data(), points_p.size(), final_center_ids.size()); 
    }
}
namespace MLP
{
    struct Train_API
    {
        struct data_obtain_api
        {
            std::vector<classTree::tree> pre_tracing_tree;


            template<typename T>
            float get_position_variance(std::vector<T>& pos,int it)
            {
                float3 mean = make_float3(0.0);
                for (int i = 0; i < it; i++)
                {
                    mean += pos[i].position / it;
                }
                float3 var = make_float3(0);
                for (int i = 0; i < it; i++)
                {
                    float3 diff = mean - pos[i].position;
                    var += diff * diff / (it - 1);
                }
                return max(var.x,max(var.y,var.z));
            }
            void get_only_light_sample(std::vector<HS_algorithm::gamma_point>& accm_centers,int sample)
            {
                std::vector<classTree::VPL> vpls;
                vpls = data_obtain_cudaApi::get_light_cut_sample(sample);
                for (int i = 0; i < min(sample, vpls.size()); i++)
                {
                    HS_algorithm::gamma_point g;
                    g.position = vpls[i].position;
                    g.direction = vpls[i].dir;
                    g.normal = vpls[i].normal;
                    g.diag2 = 0;
                    accm_centers.push_back(g);
                }

            }
            void get_rough_weighted_sample(std::vector<classTree::divide_weight_with_label>& exist_samples, std::vector<HS_algorithm::gamma_point> &accm_centers,
                int num_cluster,bool eye_side,float diag2 = 1)
            {
                exist_samples.clear();
                std::vector<classTree::divide_weight> dw_eye;  

                //get the suffix/prefix sub-paths from the full paths
                data_obtain_cudaApi::classification_data_get_flat(dw_eye, 1.0, eye_side);
                auto& weight_points = dw_eye;
                int point_M = weight_points.size();
                std::vector<HS_algorithm::gamma_point> v(point_M);
                printf("eye buffer size %d\n", point_M);
                diag2 = get_position_variance(weight_points, 10000);
                printf("classification diag2 %f\n",diag2);
                for (int i = 0; i < point_M; i++)
                {
                    v[i].diag2 = diag2;
                    v[i].direction = weight_points[i].dir;
                    v[i].position = weight_points[i].position;
                    v[i].normal = weight_points[i].normal;
                }  
                exist_samples.resize(point_M);  
                std::vector<float > n_weight;
                float weight_sum = 0.0;
                for (int i = 0; i < point_M; i++)
                { 
                    //weight_points[i].weight = 1;
                    //static float acc_weight = 0;
                    //acc_weight += weight_points[i].weight;
                    //printf("acc_weight %d %f\n",i , acc_weight);
                    n_weight.push_back(weight_points[i].weight);   
                    //weight_sum += weight_points[i].weight;
                }

                if(false)
                {
                    float acc_weight = 0;
                    for (int i = 0; i < point_M; i++)
                    {
                        acc_weight += weight_points[i].weight;
                        if (acc_weight > weight_sum / 650)
                        {
                            acc_weight -= weight_sum;
                            accm_centers.push_back(v[i]);
                        }
                    }
                }

                auto n_labels = HS_algorithm::stra_hs(v, n_weight.data(), n_weight.size(), num_cluster, accm_centers);
                for (int i = 0; i < point_M; i++)
                {
                    *(classTree::divide_weight*)(&exist_samples[i]) = weight_points[i];
                    exist_samples[i].label = n_labels[i];
                }
            }
            void get_further_weighted_sample(std::vector<classTree::divide_weight_with_label>& exist_samples, std::vector<HS_algorithm::gamma_point>& accm_centers,
                std::vector<float> E, std::vector<float> Q, 
                int num_cluster, bool eye_side, float diag2 = 1)
            {
                exist_samples.clear();
                std::vector<classTree::divide_weight> dw_eye;
                //data_obtain_cudaApi::classification_data_get_flat(dw_eye, 1.0, eye_side);
                data_obtain_cudaApi::classification_weighted_function(dw_eye, E.data(), SUBSPACE_NUM, SUBSPACE_NUM, Q.data(), eye_side);
                auto& weight_points = dw_eye;
                int point_M = weight_points.size();
                std::vector<HS_algorithm::gamma_point> v(point_M);
                printf("eye buffer size %d\n", point_M);
                for (int i = 0; i < point_M; i++)
                {
                    v[i].diag2 = diag2;
                    v[i].direction = weight_points[i].dir;
                    v[i].position = weight_points[i].position;
                    v[i].normal = weight_points[i].normal;
                }
                exist_samples.resize(point_M);
                std::vector<float > n_weight;
                for (int i = 0; i < point_M; i++)
                {
                    n_weight.push_back(weight_points[i].weight);
                }
                auto n_labels = HS_algorithm::stra_hs(v, n_weight.data(), n_weight.size(), num_cluster, accm_centers);
                for (int i = 0; i < point_M; i++)
                {
                    *(classTree::divide_weight*)(&exist_samples[i]) = weight_points[i];
                    exist_samples[i].label = n_labels[i];
                }
            }
                
            classTree::tree_node* set_device_tree(Context& context,classTree::tree &t)
            {
                printf("%d\n", t.size);
                auto b = context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, t.size);
                b->setElementSize(sizeof(classTree::tree_node));
                auto p = reinterpret_cast<classTree::tree_node*>(b->map());
                for (int i = 0; i < t.size; i++)
                {
                    p[i] = t.v[i];
                }
                b->unmap();
                
                return reinterpret_cast<classTree::tree_node*>(b->getDevicePointer(0));
            }

            std::vector<float> get_data(Context& context, std::vector<float> &Q, float &take_time)
            {

                /// <summary>
                /// comstruct the path dataset D for training
                /// </summary>
                float LVC_construct_time = 0;
                float pre_tracing_time = 0;
                int pre_tracing_count = 0;
                float build_tree_time = 0;
                float optimal_Gamma_time = 0;
                printf("path construct begin\n");

                RTsize size_one;
                context["raw_LVC"]->getBuffer()->getSize(size_one);
                int target_samples = PATH_DATASET_SIZE;
                int target_light_sample = 1000000;

                int light_samples_num = size_one * (target_light_sample / PATH_M);
 
                MLP::data_buffer b;
                b.launch_size = make_int2(500, 500);
                b.res_padding = make_int2(3, 15);
                data_obtain_cudaApi::set_data_buffer(b);
                data_obtain_cudaApi::result_buffer_setting();


                float t = sutil::currentTime();

                for (int i = 0; i < light_samples_num / size_one; i++)
                { 
                    int size = LT_trace(context);
                    thrust::device_ptr<RAWVertex> p(reinterpret_cast<RAWVertex*> (context["raw_LVC"]->getBuffer()->getDevicePointer(0)));
                    data_obtain_cudaApi::LVC_process(p, size * i, size);

                    if (i != 0)//skip compile time
                    {
                        LVC_construct_time += (sutil::currentTime() - t) * (light_samples_num / size_one) / (light_samples_num / size_one - 1);
                    }
                    t = sutil::currentTime();
                }
 
                int acc_samples = 0;
 
                //trace multiple time to get the paths enough for training
                while (acc_samples < target_samples)
                { 
                    data_obtain_cudaApi::buffer_pointer_validation();
                    context["mlp_data_buffer"]->setUserData(sizeof(data_buffer), &b);
                    context->launch(MLPPathConstructProg, b.launch_size.x, b.launch_size.y);
                    acc_samples = data_obtain_cudaApi::valid_sample_gather();

                    if (pre_tracing_count != 0)//skip compiling time, optix needs extra time for its runtime compiling, we should not take the compling time into our overhead 
                    {
                        pre_tracing_time += (sutil::currentTime() - t);
                    }
                    pre_tracing_count++;
                    t = sutil::currentTime();
                }
                pre_tracing_time *= (pre_tracing_count) / float(pre_tracing_count - 1); 

                /// <summary>
                /// to get the classification function \kappa, we first sample the subspace centroid sub-path 
                /// </summary> 
                data_obtain_cudaApi::sample_reweight();
                std::vector<classTree::divide_weight> dw_eye;
                std::vector<classTree::divide_weight> dw_light;
                std::vector<HS_algorithm::gamma_point> accm_centers_eye;
                std::vector<HS_algorithm::gamma_point> accm_centers_light; 
                std::vector<classTree::divide_weight_with_label> exist_samples;

                  
                
                int step_A[2] = { SUBSPACE_NUM  ,0 };
                int step_B[3] = { 0.1 * SUBSPACE_NUM , 0.65 * SUBSPACE_NUM , 0 };

                //stop growing tree when the depth is more than 12  
                int tree_limit_depth = 12; 
                get_only_light_sample(accm_centers_light, step_B[0]);
                
                //classify the sub-paths based on their distance from the centoid, then build the decision tree
                get_rough_weighted_sample(exist_samples, accm_centers_eye, step_A[0], true);
                auto t_eye = classTree::buildTreeBaseOnExistSample()(exist_samples, 0.9999, tree_limit_depth);
                get_rough_weighted_sample(exist_samples, accm_centers_light, step_B[1], false);
                auto t_light = classTree::buildTreeBaseOnExistSample()(exist_samples, 0.9999, tree_limit_depth); 


                classTree::tree_node* light_tree_p = data_obtain_cudaApi::light_tree_to_device(t_light.v, t_light.size); 
                classTree::tree_node* eye_tree_p = data_obtain_cudaApi::eye_tree_to_device(t_eye.v, t_eye.size); 


                build_tree_time = sutil::currentTime() - t;
                t = sutil::currentTime();


                /// <summary>
                /// now we get the function kappa to judge the subspace of a sub-path, then we label the prefix and suffix of full paths in dataset D,
                /// train the matrix Gamma and normalization term Q
                /// </summary> 
                data_obtain_cudaApi::node_label(eye_tree_p, light_tree_p); 
                Q = data_obtain_cudaApi::get_Q(light_tree_p);  

                data_obtain_cudaApi::build_optimal_E_train_data(target_samples);  

                //a little trick: when training the MIS-aware Matrix, starting traing from the contribution-based Gamma makes better convergence
                auto optimal_E = data_obtain_cudaApi::get_fake_E();
                optimal_E = data_obtain_cudaApi::train_optimal_E(optimal_E);  
                 


                data_obtain_cudaApi::clear_thrust_vector(); 
                
                if(true)
                {

                    //transfer the kappa to the device for runtime rendering.
                    classTree::tree_node* light_tree_p = set_device_tree(context, t_light);
                    classTree::tree_node* eye_tree_p = set_device_tree(context, t_eye); 
                    context["classTree::temple_eye"]->setUserData(sizeof(classTree::tree_node*), &eye_tree_p);
                    context["classTree::temple_light"]->setUserData(sizeof(classTree::tree_node*), &light_tree_p);
                }
                optimal_Gamma_time = sutil::currentTime() - t;
                printf("LVC_time %fs \n", LVC_construct_time);
                printf("pre tracing %fs \n", pre_tracing_time);
                printf("build_tree %fs \n", build_tree_time);
                printf("optimalGamma %fs \n", optimal_Gamma_time);
                take_time = LVC_construct_time + pre_tracing_time + build_tree_time + optimal_Gamma_time;
                return optimal_E;
            }
        } data;

    };
}

MLP::Train_API train_api;
MLP::MLP_DEBUG mlp_debug_api;
#endif