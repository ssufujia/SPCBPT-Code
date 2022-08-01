#ifndef BDENV_DEVICE
#define BDENV_DEVICE
#include "BDenv_common.h"
#include "random.h"

rtBuffer<float, 1> env_sampling_buffer;
rtBuffer<int, 1> env_label_buffer;
rtTextureSampler<float4, 2> envmap;
rtDeclareVariable(float, env_lum, , ) = {1};
struct envInfo_device:envInfoBase
{

	RT_FUNCTION float2 coord2uv(int2 coord,unsigned int &seed)
	{
		float r1 = rnd(seed), r2 = rnd(seed);
		float u, v;
		u = float(coord.x + r1) / float(width);
		v = float(coord.y + r2) / float(height);
		return make_float2(u, v);
	}
	RT_FUNCTION float3 sample(unsigned int &seed)
	{
        float index = rnd(seed);
        int mid = size / 2 - 1, l = 0, r = size;
        while (r - l > 1)
        {
            if (index < env_sampling_buffer[mid])
            {
                r = mid + 1;
            }
            else
            {
                l = mid + 1;
            }
            mid = (l + r) / 2 - 1;
        }
		int2 coord = index2coord(l);
		float2 uv = coord2uv(coord, seed);
		return uv2dir(uv);
	}
	RT_FUNCTION float3 sample_projectPos(float3 dir, unsigned int& seed)
	{
		const float r1 = rnd(seed);
		const float r2 = rnd(seed);
		float3 pos;
		optix::Onb onb(dir);
		cosine_sample_hemisphere(r1, r2, pos);

		return 10* r * (-dir) + pos.x * r * onb.m_tangent + pos.y * r * onb.m_binormal + center;
	}
	RT_FUNCTION float projectPdf()
	{
		return 1/(M_PI * r * r);
	}

	RT_FUNCTION int getLabel(optix::float3 dir)
	{
		float2 uv = dir2uv(dir);
		int2 coord = uv2coord(uv);
		int index = coord2index(coord);
		int res_id = env_label_buffer[index];
		if (SUBSPACE_NUM < 1000)
		{
			int target = SUBSPACE_NUM / float(1000) * 200;
			res_id = res_id * target / 200.0;
		} 

		return SUBSPACE_NUM - 1 - res_id;
	}
	RT_FUNCTION float3 getColor(optix::float3 dir)
	{
		float2 uv = dir2uv(dir);
#ifndef BD_ENV
		return make_float3(tex2D(envmap, uv.x, uv.y));
#else 
		return env_lum * make_float3(tex2D(envmap, uv.x, uv.y));
#endif
	}

	RT_FUNCTION float3 color(optix::float3 dir)
	{
		return getColor(dir);
	}
	RT_FUNCTION float pdf(optix::float3 dir)
	{
		float2 uv = dir2uv(dir);
		int2 coord = uv2coord(uv);
		int index = coord2index(coord);

		float pdf1 = index == 0 ? env_sampling_buffer[index] : env_sampling_buffer[index] - env_sampling_buffer[index - 1];

		//if (luminance(color(dir)) / (pdf1 * size / (4 * M_PI)) > 1000)
		//{
			//rtPrintf("%d %d\n", coord.x,coord.y); 
			//return 1000;
		//}
		return pdf1 * size / (4 * M_PI);
	}

};
rtBuffer<envInfo_device, 1> env_info;
#define sky env_info[0]
#endif