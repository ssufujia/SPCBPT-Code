#ifndef SVM_DEVICE
#define SVM_DEVICE
#include "SVM_common.h"
#include "random.h"

rtBuffer<GammaVec> SVM_GammaVec_buffer;
rtBuffer<BDPTVertex> SVM_Source_buffer;
rtBuffer<BDPTVertex> SVM_Target_buffer;
rtBuffer<OptimizationPathInfo> SVM_OPTP_buffer;
rtDeclareVariable(int, light_is_target, , ) = {1};
rtDeclareVariable(int, gamma_need_dense, , ) = { 0 };
#endif