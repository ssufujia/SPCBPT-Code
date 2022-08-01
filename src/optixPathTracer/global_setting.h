#ifndef __GLOBAL_SETTING_H
#define __GLOBAL_SETTING_H


//#define AUTO_TEST_MOD
#define TOBEREWRITE 

//#define LVCBPT
//#define PCBPT
//#define ZGCBPT
#define ZGCBPT

//#define FIREPLACE
//#define CLASSROOM
//#define BATHROOM
//#define KITCHEN
//#define CONFERENCE
//#define HALLWAY
//#define HOUSE
//#define GARDEN
//#define DOOR
//#define SPONZA
//#define BEDROOM
//#define HOUSE
//#define GLASSROOM
//#define CONFERENCE
#define DOOR

#define PCBPT_DIRECT_FACTOR 1
#ifdef KITCHEN
//#define SCENE_FILE_PATH "/data/kitchen/kitchen.scene"
#define SCENE_FILE_PATH "/data/kitchen/kitchen_gr2.scene"
//#define SCENE_FILE_PATH "/data/glossy_kitchen/glossy_kitchen.scene"
//#define REFERENCE_FILE_PATH "./standard_float/new_glossy_kitchen_118000.txt"
#define REFERENCE_FILE_PATH "./standard_float/kitchen-glass-14000.txt"
#define SCENE_PATH_AVER 3.6
//#define BD_ENV
#define PG_ENABLE
//#define PCBPT_OPTIMAL
#endif

#ifdef BEDROOM
//#define SCENE_FILE_PATH "/data/kitchen/kitchen.scene"
#define SCENE_FILE_PATH "/data/bedroom11.scene"
#define REFERENCE_FILE_PATH "./standard_float/bedroom_140000_1.txt"
#define SCENE_PATH_AVER 1.6
#define BD_ENV
//#define PG_ENABLE
//#define PCBPT_OPTIMAL
#endif
#ifdef HALLWAY
#define SCENE_FILE_PATH "/data/hallway/hallway_env2.scene" 
//#define REFERENCE_FILE_PATH "./standard_float/teaser_new_22500.txt" 
#define REFERENCE_FILE_PATH "./standard_float/n_hallway_154200.txt" 
#define SCENE_PATH_AVER 2.1
#define BD_ENV
#endif
#ifdef CONFERENCE
#define SCENE_FILE_PATH "/data/glassroom/glassroom_env.scene"
#define REFERENCE_FILE_PATH "./standard_float/glassroom_100k.txt"
#define SCENE_PATH_AVER 2.4

#define BD_ENV
#endif

#ifdef GLASSROOM
#define SCENE_FILE_PATH "/data/glassroom/glassroom_env.scene"
#define REFERENCE_FILE_PATH "./standard_float/glassroom_100k.txt"
#define SCENE_PATH_AVER 2.4

#define BD_ENV
#endif

//#ifdef BATHROOM
#ifdef DOOR
#define SCENE_FILE_PATH "/data/door/door_refine.scene"
#define REFERENCE_FILE_PATH "./standard_float/door-dark.txt"
#define SCENE_PATH_AVER 4
//#define PCBPT_OPTIMAL
#endif


#ifdef BATHROOM
#define SCENE_FILE_PATH "/data/bathroom/bathroom.scene"
#define REFERENCE_FILE_PATH "./standard_float/bathroom-50000.txt"
#define SCENE_PATH_AVER 2.8

#endif

#ifdef CLASSROOM
#define SCENE_FILE_PATH "/data/classroom/classroom.scene"
#define REFERENCE_FILE_PATH "./standard_float/classroom-90000.txt" 
#define SCENE_PATH_AVER 2.5

#endif

#ifdef FIREPLACE
#define SCENE_FILE_PATH "/data/fireplace/fireplace-3.0.scene"
#define REFERENCE_FILE_PATH "./standard_float/fireplace-alpha-11000.txt"
#define SCENE_PATH_AVER 2.8
#endif


#ifdef HOUSE
#define SCENE_FILE_PATH "/data/house/house_uvrefine2.scene"
#define REFERENCE_FILE_PATH "./standard_float/house-50000.txt"
#define SCENE_PATH_AVER 2 
#define PCBPT_DIRECT_FACTOR 1
//#define INDIRECT_ONLY
#endif

#ifdef GARDEN
#define SCENE_FILE_PATH "/data/garden/garden_sky_fire.scene"
#define REFERENCE_FILE_PATH "./standard_float/garden-fire-15k.txt"
#define SCENE_PATH_AVER 2 
#define PCBPT_OPTIMAL
#define BD_ENV
//#define INDIRECT_ONLY
#endif

#ifdef SPONZA
#define SCENE_FILE_PATH "/data/sponza_crytek/sponza_crytek2.scene"
#define REFERENCE_FILE_PATH "./standard_float/sponza-127000.txt"
#define SCENE_PATH_AVER 2 
//#define PCBPT_OPTIMAL 
#define BD_ENV
//#define INDIRECT_ONLY
#endif
//#define SCENE_FILE_PATH "/data/hallway/hallway2.scene"
//#define SCENE_FILE_PATH "/data/classroom/classroom.scene"
//#define SCENE_FILE_PATH "/data/conference/conference.scene"
//#define SCENE_FILE_PATH "/data/bathroom/bathroom.scene"
//#define SCENE_FILE_PATH "/data/fireplace/fireplace-3.0.scene"
//#define SCENE_FILE_PATH "/data/house/house.scene"
///shadingnormal
//#define REFERENCE_FILE_PATH "./standard_float/bedroom_30000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/stair-35000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/bedroom11-pt50000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt”
//#define REFERENCE_FILE_PATH "./standard_float/livingroom-3.0.txt"



///geo normal
//#define REFERENCE_FILE_PATH "./standard_float/fireplace_30000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/breafast_30000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/hallway-25000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/classroom-90000.txt" 


////#define REFERENCE_FILE_PATH "./standard_float/hallway-50000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/hallway-300000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/classroom-90000.txt" 
//#define REFERENCE_FILE_PATH "./standard_float/conference_42000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/bathroom-50000.txt"
//#define REFERENCE_FILE_PATH "./standard_float/fireplace3.0-60000.txt"

//#define CONTINUE_RENDERING_BEGIN_FRAME 123410
#define CONTINUE_RENDERING_BEGIN_FRAME 123410

#define DIR_JUDGE 0
#define LABEL_BY_STREE false
//#define LTC_STRA
//#define BD_ENV
#define UNBIAS_RENDERING
//#define DIFFUSE_ONLY
//#define GLOSSY_ONLY
#define RR_MIN_LIMIT
#ifdef RR_MIN_LIMIT
#define MIN_RR_RATE 0.3f
#endif

#ifndef UNBIAS_RENDERING

#ifndef RR_MIN_LIMIT
#define RR_DISABLE
#endif
#endif

#define RIS_M2 200
#define RIS_M1 10000
//200 500 1000 2000
#define SUBSPACE_NUM 1000
#define iterNum 3
//PATH_M 10000 50000 100000  200000 500000 1000000
#define PATH_M 100000
//#define iterNum int(SCENE_PATH_AVER + 1)
#define PATH_DATASET_SIZE 1000000
//#define BACK_ZONE
//#define LVC_RR

#define MAX_LIGHT_VERTEX_PER_TRIANGLE 200
#ifdef ZGCBPT
//iterNum 1 2 3 4 8
//#define iterNum 3
#define SLIC_CLASSIFY 
#define LABEL_BY_STREE true
//#define INDIRECT_ONLY
#define EYE_DIRECT_FRAME 2000
#define ZGC_SAMPLE_ON_LUM
#define TRACER_PROGRAM_NAME "ZGCBPT_pinhole_camera"
//#define LIGHTVERTEX_REUSE 
//#define TRACER_PROGRAM_NAME "unbias_pt_pinhole_camera"
//#define CONTINUE_RENDERING 

//#define UBER_RMIS
//#define TRACER_PROGRAM_NAME "ZGCBPT_test_pinhole_camera"
//
//#define PRIM_DIRECT_VIEW

#else


#ifdef PCBPT
//#define NOISE_DISCARD
//#define LTC_STRA
#define iterNum 3 
//#define INDIRECT_ONLY
#define PCBPT_MIS
#define PCBPT_STANDARD_MIS
#define KD_3
#define connectRate_kd3(r,ind) min(((r / KDPMFCaches[ind.x].Q + r / KDPMFCaches[ind.y].Q +r / KDPMFCaches[ind.z].Q)/3),1000.0)
#define TRACER_PROGRAM_NAME "PCBPT_pinhole_camera"
//#define PRIM_DIRECT_VIEW
#else
#ifdef BDPT
#define TRACER_PROGRAM_NAME "BDPT_pinhole_camera"
#else 
#ifdef LVCBPT
//#define TRACER_PROGRAM_NAME "unbias_pt_pinhole_camera"
#define TRACER_PROGRAM_NAME "LVCBPT_pinhole_camera"
//#define CONTINUE_RENDERING
//#define CONTINUE_RENDERING_BEGIN_FRAME 75700
#else
#define TRACER_PROGRAM_NAME "pinhole_camera"
//#define NO_COLOR
#ifdef RRPT
#define TRACER_PROGRAM_NAME "unbias_pt_pinhole_camera"
#endif
#endif
#endif

#endif // PCBPT
#endif



//画面评估参数设置
#define ESTIMATE_INVALID
#define ESTIMATE_FRAME {10,20,40,80,160,320,640}
#define ACCM_VAL_ESTIMATE

//heat map is unavailable in this optix ver 
//#define VIEW_HEAT_pos 
//#define VIEW_HEAT 

//是否使用PCPT
//#define USE_PCPT

//是否重新划分区域
//#define ZONE_ALLOC

//是否使用视顶点矩阵
//#define USE_ML_MATRIX

//是否使用朴素方法划分区域
#define RAW_CLUSTER
#ifdef RAW_CLUSTER
#define KMEANS_ITER_NUM 1
#ifndef ZONE_ALLOC
#define ZONE_ALLOC
#endif // !ZONE_ALLOC
#else
#define KMEANS_ITER_NUM 10
#endif // RAW_CLUSTER


//场景中是否有透明介质
#define BRDF
#define STACKSIZE 24
#define M 1.0f
#define RR_RATE 1.0f
#define RR_BEGIN_DEPTH 2

//降噪参数设置
//#define USE_DENOISER
#ifdef USE_DENOISER
#define DENOISE_BEGIN_FRAME 1000
#endif // USE_DENOISER
#ifdef PCBPT
#define PCPT_CORE_NUM RIS_M2
#define LIGHT_VERTEX_PER_CORE 20
#else
#define M_PER_CORE 100
#define PCPT_CORE_NUM int(PATH_M / M_PER_CORE)
#define LIGHT_VERTEX_PER_CORE int(SCENE_PATH_AVER * M_PER_CORE)
#endif
#define LTC_CORE_NUM 1000
#define LTC_SPC 100
#define LTC_SAVE_SUM (LTC_CORE_NUM * LTC_SPC)

#define PMF_DEPTH 5
#define UberWidth 100
#define LIGHT_VERTEX_NUM (PCPT_CORE_NUM * LIGHT_VERTEX_PER_CORE) 
#define ZONE_OPTIMAL_DIRECTOR_NUM 1000
#define UBER_VERTEX_NUM (LIGHT_VERTEX_NUM / UberWidth *10) 
#define MAX_TRIANGLE 6000000
#define MAX_TRI_AREA 2000000
#define MGPT_GUIDE_RATE 0.9f
#define UNIFORM_GUIDE_RATE 0.25f
#define FIX_SAMPLE_RATE 0.25
#define FIX_RANGE 15
#define PMFCaches_RATE 0.063f
#define MAX_PATH_LENGTH 25
#define VISIBILITY_TEST_NUM 100
#define VISIBILITY_TEST_SLICE 10
#define MAX_LIGHT 201
#define RECORD_DEPTH 1 

#define BUFFER_WEIGHT 1.f 
#define DISCARD_VALUE 1000.0f
#define POWER_RATE 1.0f
#define PG_RATE 0.5
#define PG_LIGHTSOURCE_RATE 0.5
#define  PPM_X         ( 1 << 0 )
#define  PPM_Y         ( 1 << 1 )
#define  PPM_Z         ( 1 << 2 )
#define  PPM_LEAF      ( 1 << 3 )
#define  PPM_NULL      ( 1 << 4 )
#define ENERGY_WEIGHT(a) (a.x + a.y + a.z)

#ifdef KITCHEN_DISCARD
#define EYE_DIRECT
#else
#define LIGHTVERTEX_REUSE 
#endif


//#define SAMPLE_ONLY_SOURCE
#endif