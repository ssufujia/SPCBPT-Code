#ifndef SUBSPACE_DEVICE
#define SUBSPACE_DEVICE
#include "subspace_common.h"
#include "random.h"

rtBuffer<BDPTVertex, 1>           subspace_LVC;
rtBuffer<jumpUnit, 1>                jump_buffer;
struct subspaceSamplerDevice :subspaceSampler
{
    RT_FUNCTION BDPTVertex& sample(unsigned int &seed, float& pmf)
    {
        float index = rnd(seed);
        int mid = size / 2 - 1, l = 0, r = size;
        while (r - l > 1)
        {
            if (index < jump_buffer[mid + base].cmf)
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }

        int id = l + base;
        jumpUnit& ju = jump_buffer[id];
        pmf = ju.pmf;
        return subspace_LVC[ju.p];
    }
    RT_FUNCTION
    bool empty()
    {
        return size == 0;
    }
};
rtBuffer<subspaceSamplerDevice,1>   vertex_sampler;
#endif 