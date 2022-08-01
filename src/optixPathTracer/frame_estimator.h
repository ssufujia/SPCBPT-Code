#ifndef ESTIMATOR_1028_H
#define ESTIMATOR_1028_H
//#define WRITETOFILE
//#define SAVE_VIDEO
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include<fstream>
#include<vector>
#include<set>
#include<queue>
#include<sutil.h>
#include<string>
#include"BDPT_STRUCT.h"
using namespace std;
using namespace optix;
static bool isInit = false;
static queue<int> estimate_frame;
static queue<float> estimate_time;
static double current_time;
static string render_way;
static std::ofstream outFile;
static std::ofstream outFile_with_headinfo;
static float currentMape = 1000;
string path = "./exp_record2/";
string scene_name = "FIREPLACE";
std::string get_scene_name()
{
    std::string filename(SCENE_FILE_PATH);
    int index_begin = filename.find_last_of("/");
    int index_end = filename.find_last_of(".");
    int slic_len = index_end - index_begin - 1;
    return filename.substr(index_begin+1, slic_len);

}
struct estimate_status
{
    std::string scene_name;
    std::string algorithm;
    int n_vertexConnectNum = 1;
    int N_subspace = SUBSPACE_NUM;
    int M_lightPathVertex = LIGHT_VERTEX_NUM;
    int M_lightPath = 4;
    float average_path_length = 1;
    bool isIndirectOnly = true;
    bool useClassifier = true;
    float subspaceDivTime = 0.0;
    int t0_strategyNum = 0;
    std::string log_head()
    {
        int retain_sub_path = false;
#ifdef LIGHTVERTEX_REUSE
        retain_sub_path = true;
#endif
        char q[11000];
        sprintf(q, 
            "{\n\
            \"scene\" : \"%s\",\n \
            \"algorithm\" : \"%s\",\n \
            \"N_conn\" : %d,\n \
            \"N_subspace\" : %d,\n \
            \"M_lightvertex\" : %d,\n \
            \"M_lightpath\" : %d,\n \
            \"isIndirectOnly\" : %d,\n \
            \"useClassifier\" : %d,\n  \
            \"subspaceDivTime\" : %f,\n \
            \"t0_strategy\" : %d\n \
            \"dataset_size\": %d\n\
            \"use_retain\": %d\n\
            } \n",scene_name, algorithm, n_vertexConnectNum, N_subspace, M_lightPathVertex, M_lightPath, isIndirectOnly ? 1 : 0,useClassifier?1:0,
            subspaceDivTime,t0_strategyNum,PATH_DATASET_SIZE, retain_sub_path
        );
        return std::string(q);
    }

    estimate_status()
    {
        algorithm = "UNDEFINED";

    }
    estimate_status(std::string sceneName,float ssDivTime,float M_lightPath = 1,float average_path_length = 1):
        scene_name(sceneName),
        subspaceDivTime(ssDivTime),n_vertexConnectNum(iterNum),N_subspace(SUBSPACE_NUM),M_lightPathVertex(LIGHT_VERTEX_NUM),M_lightPath(M_lightPath)
    {
        algorithm = "UNDEFINED";
#ifdef ZGCBPT
        algorithm = "ZGCBPT";
#endif
#ifdef PCBPT
        algorithm = "PCBPT";
#endif
#ifdef LVCBPT
        algorithm = "LVCBPT";
#endif

        useClassifier = false;
#ifdef SLIC_CLASSIFY
        useClassifier = true;
#endif // SLIC_CLASSIFY
        isIndirectOnly = false;
#ifdef INDIRECT_ONLY
        isIndirectOnly = true;
#endif

        t0_strategyNum = 0;
#ifdef LTC_STRA
        t0_strategyNum = LTC_SAVE_SUM / average_path_length;
#endif
    }
    std::string save_file_name()
    {
        int retain_sub_path = false;
#ifdef LIGHTVERTEX_REUSE
        retain_sub_path = true;
#endif
        char q[11000];
        sprintf(q,
            "%s_%s_%d_%d_%d_%d_%d_%d_%d_%d", scene_name,algorithm, n_vertexConnectNum, N_subspace, M_lightPathVertex, isIndirectOnly ? 1 : 0,t0_strategyNum == 0?0:1,useClassifier,PATH_DATASET_SIZE,retain_sub_path
        ); 
        return std::string(q);
    }
};

static void init(estimate_status& es)
{
//    auto test_use= estimate_status();
//    printf("%s",test_use.log_head().c_str());
    vector<int> ef{66,2500};
    vector<float> et{25, 50, 60, 120,180,300,660};
//    vector<float> et{25, 50, 60, 70};
    for(int i = 0;i<ef.size();i++)
    {
        estimate_frame.push(ef[i]);
    }
    for(int i = 0;i<et.size();i++)
    {
        estimate_time.push(et[i]);
    }
    render_way = "PT";
#ifdef PCBPT
    render_way = "PCBPT";
#endif // PCBPT
#ifdef LVCBPT
        render_way = string("LVCBPT");
#endif
#ifdef ZGCBPT
    render_way = "ZGCBPT";
#endif // ZGCBPT

#ifdef KITCHEN
    scene_name = "KITCHEN";
#endif
#ifdef HALLWAY
    scene_name = "HALLWAY";
#endif
#ifdef CONFERENCE
    scene_name = "CONFERENCE";
#endif
#ifdef BATHROOM
    scene_name = "BATHROOM";
#endif
#ifdef CLASSROOM
    scene_name = "CLASSROOM";
#endif
#ifdef FIREPLACE
    scene_name = "FIREPLACE";
#endif
#ifdef HOUSE
    scene_name = "house";
#endif
#ifdef DOOR
    scene_name = "door";
#endif

#ifdef GARDEN
    scene_name = "garden";
#endif

#ifdef SPONZA
    scene_name = "sponza";
#endif

#ifdef BEDROOM
    scene_name = "bedroom";
#endif
#ifdef GLASSROOM
    scene_name = "glassroom";
#endif

    path = string("./exp_record/") + render_way;
#ifdef WRITETOFILE
    path = string("./exp_record/") + scene_name + "/" + render_way;
#endif
    string path2 = path + "/false_log.txt";

    string path_with_info = string("./exp_record_logs_only/") + es.save_file_name() + ".txt";
    outFile_with_headinfo.open(path_with_info.c_str());
    outFile_with_headinfo << es.log_head();
}
//assumption that the standrd frame is loaded
static std::string frame_estimate(Context & context)
{
    float minLimit = 0.01;
    context["estimate_min_limit"]->setFloat(minLimit);
    RTsize OW, OH;
    context["standrd_float_buffer"]->getBuffer()->getSize(OW, OH);
    if (OW != 1920 || OH != 1001)
    {
        std::cout << "estimate fall for wrong screen size" << std::endl;
        return string("no result");
    }
    float4* p1 = reinterpret_cast<float4*>(context["accum_buffer"]->getBuffer()->map());
    float4* p2 = reinterpret_cast<float4*>(context["standrd_float_buffer"]->getBuffer()->map());
    uchar4* p3 = reinterpret_cast<uchar4*>(context["standrd_buffer"]->getBuffer()->map());
    float error = 0.0f;
    float re_error = 0.0f;
    float mae = 0.0f; 
    float valid_pixiv = 0;
    int filter_count = 0;
    for (int i = 0; i < OW * OH; i++)
    {

        float3 a = make_float3(p1[i]);
        if (a.x + a.y + a.z > 0)
            valid_pixiv += 1;
        float3 b = make_float3(p2[i]);
        float3 bias = a - b;
        float3 r_bias = (a - b) / (b + make_float3(minLimit));
        float diff = length(a - b) / 3.0;
        float diff_b = length((a - b) / (b + make_float3(minLimit))) / 3;
        float diff_c = (abs(bias.x) + abs(bias.y) + abs(bias.z)) / 3;
        float diff_d = (abs(r_bias.x) + abs(r_bias.y) + abs(r_bias.z)) / 3;
        //warning : this option may be buggy
        //if ((b.x + b.y + b.z) > 5)
        //{
        //    diff = diff_b = diff_c = diff_d = 0;

        //}

        error += diff * diff;
        // re_error += diff_b * diff_b;
        float real_diff = clamp(diff_d, 0.0, 10.0);
        re_error += real_diff;// < 1.0 ? diff_d : 1.0;
        mae += real_diff * real_diff;
//        mae += diff_c;
        filter_count++;
    }
    std::cout<<"valid_pixiv_num : "<< valid_pixiv << std::endl << std::endl;
    error /= filter_count;
    re_error /= filter_count;
    mae /= filter_count;
    //float rmae = sqrtf(mae); 
    float rmae = mae; 
    
    context["standrd_buffer"]->getBuffer()->unmap();
    context["accum_buffer"]->getBuffer()->unmap();
    context["standrd_float_buffer"]->getBuffer()->unmap();

    char output_b[1000];
    sprintf_s(output_b, "%d %f %f %f %f%%", context["frame"]->getUint(), current_time, error, rmae, re_error * 100);
    currentMape = re_error * 100;
    return string(output_b); 
     
}

void estimate( Context& context, double& elapsedTime,double & lastTime, int frame,estimate_status &es)
{
#ifndef AUTO_TEST_MOD
    return;
#endif // !AUTO_TEST_MOD
    elapsedTime += sutil::currentTime() - lastTime;
    lastTime = sutil::currentTime();
    if (isInit == false)
    {

        elapsedTime = 0;
#ifdef ZGCBPT
        elapsedTime = es.subspaceDivTime; 
#endif

        isInit = true;
        init(es);

    }
    current_time = elapsedTime;
    if (outFile_with_headinfo)
    {
        auto test_text = frame_estimate(context); 
        outFile_with_headinfo << test_text << endl;
        float cut_line = 11.9;

//        if (currentMape < cut_line && (estimate_frame.empty() == true || estimate_time.empty() == true))
        if ((estimate_frame.empty() == true || estimate_time.empty() == true))
        { 
            outFile_with_headinfo.close();
            //exit(0);
        }
    }
    bool img_save = false;
    int saveTime = 0;
    lastTime = sutil::currentTime();
}

extern bool standrdImageIsLoad;
void load_standrd(Context & context)
{
#ifndef AUTO_TEST_MOD
    return;
#endif // !AUTO_TEST_MOD

    if (!standrdImageIsLoad)
    {
        float4* p2 = reinterpret_cast<float4*>(context["standrd_float_buffer"]->getBuffer()->map());
        uchar4* p3 = reinterpret_cast<uchar4*>(context["standrd_buffer"]->getBuffer()->map());
        int i = 0;
        float a, b, c, d;
        std::ifstream inFile;
        inFile.open(REFERENCE_FILE_PATH);
        while (inFile >> a >> b >> c >> d)
        { 
            p2[i] = optix::make_float4(a, b, c, d);

            float lum = 0.3 * a + 0.6 * b + 0.1 * c;
            float limit = 1.5;
            float4 tone = p2[i] * 1.0f / (1.0f + lum / limit);
            float kInvGamma = 1.0f / 2.2f;
            float3 gamma_color = make_float3(pow(tone.x, kInvGamma), pow(tone.y, kInvGamma), pow(tone.z, kInvGamma));
            gamma_color.x = fminf(1.0f, gamma_color.x);
            gamma_color.y = fminf(1.0f, gamma_color.y);
            gamma_color.z = fminf(1.0f, gamma_color.z);
            uchar4 color = make_uchar4(uint(gamma_color.z * 255), uint(gamma_color.y * 255), uint(gamma_color.x * 255), 255);
            p3[i] = color;
            i++;

        }

        standrdImageIsLoad = true;
        inFile.close();
        context["standrd_float_buffer"]->getBuffer()->unmap();
        context["standrd_buffer"]->getBuffer()->unmap();
        sutil::writeBufferToFile("./theImageYouLoad.png", context["standrd_buffer"]->getBuffer());
    }
    return;
}
#endif


