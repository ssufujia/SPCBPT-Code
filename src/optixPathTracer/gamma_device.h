#ifndef GAMMA_DEVICE
#define GAMMA_DEVICE
#include"gamma_common.h"
#include"BDPTVertex.h"
#include"random.h"

struct lightSelectionFunction_device:public lightSelectionFunction
{
    RT_FUNCTION float operator()(int subspaceID_eye, int subspaceID_light, float lum)
    {
        return Matrix[subspaceID_eye][subspaceID_light] * lum / Q[subspaceID_light];
    }

    RT_FUNCTION float eval(int subspaceID_eye, int subspaceID_light, float lum)
    {
        return (*this)(subspaceID_eye, subspaceID_light, lum);
    }
    RT_FUNCTION float computeMISweight(BDPTVertex& z_bar, BDPTVertex& y_bar)
    {
        return (*this)(z_bar.zoneId, y_bar.zoneId, y_bar.contri_float());
    }
};

#endif // !GAMMA_DEVICE
