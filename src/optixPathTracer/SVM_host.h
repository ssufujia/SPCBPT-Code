#ifndef SVM_HOST
#define SVM_HOST
#include "SVM_common.h"
#include"BDPT_STRUCT.h" 
#include"ZGC.h"
#include<random>
#include<vector>
#include <sutil.h>
#include<fstream>
#include"classTree_host.h" 

#include "MLP_host.h"
using namespace optix;

struct SVMHost
{
    Buffer GammaVec_buffer;
    Buffer Source_buffer;
    Buffer Target_buffer;
    Buffer OPTP_buffer;
    int target_num = 100000;
    int source_num = GAMMA_DIM;
    int OPTP_num = 1000000;
    int launch_num = 0;
    int expect_num = 5000000;
    int path_per_file = 1000000;
    void init(Context& context)
    {
        GammaVec_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, target_num); 
        GammaVec_buffer->setElementSize(sizeof(GammaVec));
        context["SVM_GammaVec_buffer"]->setBuffer(GammaVec_buffer);


        Source_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, source_num);
        Source_buffer->setElementSize(sizeof(BDPTVertex));
        context["SVM_Source_buffer"]->setBuffer(Source_buffer);


        Target_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, target_num);
        Target_buffer->setElementSize(sizeof(BDPTVertex));
        context["SVM_Target_buffer"]->setBuffer(Target_buffer);

        OPTP_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_USER, OPTP_num);
        OPTP_buffer->setElementSize(sizeof(OptimizationPathInfo));
        context["SVM_OPTP_buffer"]->setBuffer(OPTP_buffer);


    }
    void divide_weight_process(OptimizationPathInfo& p,
        std::vector<classTree::divide_weight>& w,bool light_side)
    {
        int begin_idx = light_side == true ? 2 : 1;
        int end_idx = light_side == true ? p.path_length : p.path_length - 1;
        for (int i = begin_idx; i < end_idx; i++)
        {
            classTree::divide_weight a;
            a.position = p.positions[i];
            a.weight = 1;
            w.push_back(a);
        }
    }
    std::vector<OptimizationPathInfo> GetPathInfo(Context& context)
    {
        std::vector<OptimizationPathInfo> res;
        //GetSubpathInfo(context);
        return res;

        int evc_frame = context["EVC_frame"]->getUint();
        while (res.size() < expect_num)
        {
            //float t = sutil::currentTime();
            context->launch(OPTPProg, OPTP_num, 1);
            //printf("%f\n", sutil::currentTime() - t);
            //std::vector<classTree::buildTreeBaseOnWeightedSample::divide_weight> weights(0);


            evc_frame++;
            context["EVC_frame"]->setUint(evc_frame);

            OptimizationPathInfo* p = reinterpret_cast<OptimizationPathInfo*> (context["SVM_OPTP_buffer"]->getBuffer()->map());
            for (int i = 0; i < OPTP_num; i++)
            {
                if (p[i].valid == true)
                {
                    res.push_back(p[i]);
                    //divide_weight_process(p[i], weights, true);
                    if (res.size() == expect_num)
                    {
                        break;
                    }
                }
            }
             

            //thrust::device_vector<classTree::tree_node> d_v(tree.v,tree.v + tree.size);
            //classTree::tree tree = classTree::buildTreeBaseOnWeightedSample()(weights, 10, 1200);
            //printf("tree_size:%d\n", tree.size);
            //classTree::tree_node* dv_ptr = device_to(tree.v, tree.size);
            //context["classTree::temple_tree"]->setUserData(sizeof(classTree::tree_node*), reinterpret_cast<void*>(&dv_ptr));
            //exit(0); 

            launch_num += OPTP_num;
            printf("ssss %d %d\n", res.size(),launch_num);
            context["SVM_OPTP_buffer"]->getBuffer()->unmap();
        }


        //write to file        
        for (int i = 0; i * OPTP_num < expect_num;i++)
        {
            int begin = path_per_file * i;
            int end = min(begin + path_per_file,res.size());
            std::vector<OptimizationPathInfo> res_t(res.begin() + begin,res.begin() + end);
            std::ofstream inFile;
            char file_name[1100];
            sprintf(file_name, "./exp_record/OPTP_data/%d.txt", i);
            inFile.open(file_name);
            for (auto p = res_t.begin(); p != res_t.end(); p++)
            {
                inFile << p->contri << " " << p->path_length << " " << p->actual_pdf << " "<< p->pixiv_lum << " ";

                for (int i = 0; i < OPT_PATH_LENGTH; i++)
                {
                    inFile << p->ss_ids[i] << " ";
                }
                for (int i = 0; i < OPT_PATH_LENGTH; i++)
                {
                    inFile << p->pdfs[i] << " ";
                }
                for (int i = 0; i < OPT_PATH_LENGTH; i++)
                {
                    inFile << p->light_contris[i] << " ";
                }

                for (int i = 0; i < OPT_PATH_LENGTH; i++)
                {
                    inFile << p->light_pdfs[i] << " ";
                }

                for (int i = 0; i < OPT_PATH_LENGTH; i++)
                {
                    inFile << p->positions[i].x << " " << p->positions[i].y << " " << p->positions[i].z << " ";
                }
                inFile << std::endl;
            }


            inFile.close();
        }

        return res;
    } 
    void GetSubpathInfo(Context& context)
    {

        auto light_target = get_random_light_vertex(context, 5000000);
        std::ofstream inFile;
        inFile.open("./exp_record/OPTP_data/subpath_info.txt");

        for (auto p = light_target.begin(); p != light_target.end(); p++)
        {
            float3 weight0 = (p->flux / p->pdf);
            float weight = weight0.x + weight0.y + weight0.z;
            float3 in_dir = normalize(p->position - p->lastPosition);
            inFile << p->position.x << " " << p->position.y << " " << p->position.z << " " << in_dir.x << " " <<in_dir.y << " " <<in_dir.z << " " << weight << " "<<p->zoneId;

            inFile << std::endl;
        }


        inFile.close();
    }
    void recordEQ(Context& context)
    {
        return;
        std::ofstream inFile;
        inFile.open("./exp_record/OPTP_data/LF.txt");
        
        inFile << ((expect_num - 1) / path_per_file) + 1 <<" "<< float(expect_num)/launch_num << std::endl;

        ZoneSampler* zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());
        int path_sum = context["Q_light_path_sum"]->getInt();
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            inFile << zoneLVC[i].Q / path_sum<<" ";
        }
        inFile << std::endl;

        context["zoneLVC"]->getBuffer()->unmap();

        ZoneMatrix* M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
        for (int i = 0; i < SUBSPACE_NUM; i++)
        { 
            for (int j = 0; j < SUBSPACE_NUM; j++)
            { 
                inFile << M2[i].r[j] << " ";
            }
            inFile << std::endl; 
        }
        context["M2_buffer"]->getBuffer()->unmap();
        inFile.close();
    }
    std::vector<GammaVec> getGamma(Context& context)
    {
        std::vector<GammaVec> t;
        return t;


        if(false)
        {
            auto eye_target = get_random_eye_vertex(context, target_num);
            auto p1 = reinterpret_cast<BDPTVertex*>(Target_buffer->map());
            memcpy(p1, eye_target.data(), sizeof(BDPTVertex) * target_num);
            Target_buffer->unmap();


            context["gamma_need_dense"]->setInt(1);
            auto light_source = get_random_light_vertex(context, source_num);
            p1 = reinterpret_cast<BDPTVertex*>(Source_buffer->map());
            memcpy(p1, light_source.data(), sizeof(BDPTVertex) * source_num);
            Source_buffer->unmap();
            context["gamma_need_dense"]->setInt(0);

            context["light_is_target"]->setInt(0);
        }
        else
        {
            auto light_target = get_random_light_vertex(context, target_num);
            auto p1 = reinterpret_cast<BDPTVertex*>(Target_buffer->map());
            memcpy(p1, light_target.data(), sizeof(BDPTVertex) * target_num);
            Target_buffer->unmap();

            context["gamma_need_dense"]->setInt(0);
            auto eye_source = get_random_eye_vertex(context, source_num);
            p1 = reinterpret_cast<BDPTVertex*>(Source_buffer->map());
            memcpy(p1, eye_source.data(), sizeof(BDPTVertex) * source_num);
            Source_buffer->unmap();
            context["gamma_need_dense"]->setInt(0);

            context["light_is_target"]->setInt(1);
        }

        //context->launch(GammaComputeProg, target_num, 1);
        //double t1 = float(sutil::currentTime());
        //context->launch(GammaComputeProg, target_num, 1);
        //t1 = (sutil::currentTime()) - t1;
        //printf("%f\n", float(t1));

        auto p2 = reinterpret_cast<GammaVec*>(GammaVec_buffer->map());
        std::vector<GammaVec> ans(p2, p2 + target_num);
        GammaVec_buffer->unmap();
        printf("gammaProgram workable:%d\n", ans.size());

        {
            std::ofstream inFile;
            inFile.open("gamma_test.txt");

            for (auto p = ans.begin(); p != ans.end(); p++)
            {
                inFile <<p->contri<<" "<< p->default_lable<<" "<<p->objId << " " << p->surface.x << " " << p->surface.y << " " << p->dir.x << " " << p->dir.y << " ";
                for (int i = 0; i < GAMMA_DIM; i++)
                {
                    inFile << p->gamma[i] << " ";
                }
                inFile << std::endl;
            }


            inFile.close();
        }
        return ans;
    }

    void load_optimal_E(Context& context)
    {
        std::ifstream inFile;
        inFile.open("./exp_record/OPTP_data/optimalE.txt");

        ZoneMatrix* M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            for (int j = 0; j < SUBSPACE_NUM; j++)
            {
                inFile >> M2[i].r[j];
                M2[i].r[j] = M2[i].r[j] * 0.9 + 0.1 * 1.0 / SUBSPACE_NUM;
            }
            M2[i].validation();
        }
        context["M2_buffer"]->getBuffer()->unmap();
        inFile.close();
    }

    void load_optimal_E(Context& context,std::vector<float>& E)
    {
        ZoneMatrix* M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            for (int j = 0; j < SUBSPACE_NUM; j++)
            {
                M2[i].r[j] = E[i * SUBSPACE_NUM + j];
                M2[i].r[j] = M2[i].r[j] * 0.9 + 0.1 * 1.0 / SUBSPACE_NUM;
            }
            M2[i].validation();
        }
        context["M2_buffer"]->getBuffer()->unmap(); 
    }
};
SVMHost svm_api;
#endif
