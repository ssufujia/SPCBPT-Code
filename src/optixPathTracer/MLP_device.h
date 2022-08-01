#ifndef MLP_DEVICE_H
#define MLP_DEVICE_H
#include"MLP_common.h" 
#include"classTree_device.h"
#include "BDPT.h"
#include"BDenv_device.h"

rtBuffer<feed_token,1> feed_buffer;
rtBuffer<BP_token,1> BP_buffer;
rtBuffer<MLP_network, 1> MLP_buffer;
rtBuffer<MLP_network, 1> gradient_buffer;
rtBuffer<int2, 2> MLP_index_L1_buffer;
rtBuffer<int, 1> MLP_index_L2_buffer;

rtBuffer<int, 2> CC_lookup_table;
rtBuffer<float, 2> E_table;

rtDeclareVariable(int, batch_id, , ) = { 0};
rtDeclareVariable(int, batch_size, , ) = { 100000};

namespace MLP
{
	struct nVertex_device:nVertex
	{
		RT_FUNCTION nVertex_device(const nVertex& a, const nVertex_device& b,bool eye_side)
		{
			position = a.position;
			dir = normalize(b.position - a.position);
			normal = a.normal;
			weight = eye_side? b.forward_eye(a): b.forward_light(a); 
			pdf = eye_side ? weight.x : b.forward_light_pdf(a);
			color = a.color;
			materialId = a.materialId;
			valid = true;
			label_id = a.label_id;
			isBrdf = a.isBrdf;

			depth = b.depth+1;
			save_t = 0;
		
			brdf_tracing_branch(a, b, eye_side);
			
		}
		RT_FUNCTION void brdf_tracing_branch(const nVertex& a, const nVertex_device& b, bool eye_side)
		{
			if (b.isBrdf == false && a.isBrdf == true)
			{
				save_t = length(a.position - b.position);
			}
			else if (b.isBrdf == true)
			{
				save_t = b.save_t + length(a.position - b.position);
			}
			if (a.isBrdf == true && b.isBrdf == false)
			{
				if (eye_side == false)
				{
					weight = b.weight * abs(dot(dir, b.normal)) * (b.isLightSource() ? make_float3(1.0) : b.color);
				}
				else
				{
					weight = b.weight * b.color;
				}
			}
			else if (a.isBrdf == true && b.isBrdf == true)
			{
				weight = b.weight * b.color;
			}
			else if (a.isBrdf == false && b.isBrdf == true)
			{
				weight = b.weight * abs(dot(a.normal, dir)) * b.color;
				weight /= save_t * save_t;
				save_t = 0;
			}

		}
		RT_FUNCTION int get_label()
		{
			if (isLightSource())
			{
				return label_id;
			}
			return classTree::getLightLabel(position);
		}
		RT_FUNCTION nVertex_device(const BDPTVertex&a,bool eye_side):nVertex(a,eye_side){} 
		RT_FUNCTION nVertex_device(){} 

		RT_FUNCTION float forward_light_pdf(const nVertex& b)const
		{
			float3 vec = b.position - position;
			float3 c_dir = normalize(vec);
			float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);

			if (isLightSource())
			{
				if (isAreaLight())
				{
					g *= abs(dot(normal, c_dir));
					return pdf * g * 1.0 / M_PI;
				}
				if (isDirLight())
				{
					g = abs(dot(c_dir, b.normal));
					return pdf * g * sky.projectPdf();
				}
			}

			MaterialParameter mat = sysMaterialParameters[materialId];
			mat.color = color;
			float d_pdf = DisneyPdf(mat, normal, dir, c_dir, position, false);
			float RR_rate = max(fmaxf(color), MIN_RR_RATE);
			return pdf * d_pdf * RR_rate * g;
		}

		RT_FUNCTION float3 forward_eye(const nVertex& b)const
		{
			if (b.isDirLight())
			{
				float3 c_dir = -b.normal;

				MaterialParameter mat = sysMaterialParameters[materialId];
				mat.color = color;
				float d_pdf = DisneyPdf(mat, normal, dir, c_dir, position, true);
				float RR_rate = max(fmaxf(color), MIN_RR_RATE);
				return weight * d_pdf * RR_rate;
			}

			float3 vec = b.position - position;
			float3 c_dir = normalize(vec);
			float g = abs(dot(c_dir, b.normal)) / dot(vec, vec);
			
			MaterialParameter mat = sysMaterialParameters[materialId];
			mat.color = color;
			float d_pdf = DisneyPdf(mat, normal, dir, c_dir, position, true);
			float RR_rate = max(fmaxf(color), MIN_RR_RATE);
			//d_pdf /= isBrdf ? abs(dot(normal, c_dir)) : 1;
			return weight * d_pdf * RR_rate * g;
		}
			
		RT_FUNCTION float3 forward_areaLight(const nVertex & b)const
		{
			float3 vec = b.position - position;
			float3 c_dir = normalize(vec);
			float g =  abs(dot(c_dir, b.normal)) * abs(dot(c_dir, normal)) / dot(vec, vec);

			return weight * g; 
		}

		RT_FUNCTION float3 forward_dirLight(const nVertex& b)const
		{  
			return weight * abs(dot(normal, b.normal)); 
		}

		RT_FUNCTION float3 forward_light_general(const nVertex& b)const
		{
			float3 vec = b.position - position;
			float3 c_dir = normalize(vec);
			
			float g = isBrdf?
				abs(dot(c_dir, b.normal)) / dot(vec, vec):
				abs(dot(c_dir, b.normal)) * abs(dot(c_dir,normal)) / dot(vec, vec);

			
			MaterialParameter mat = sysMaterialParameters[materialId];
			mat.color = color;
			float3 d_contri = DisneyEval(mat, normal, dir, c_dir);
			return weight * g * d_contri;
			
		}
		RT_FUNCTION float3 forward_light(const nVertex& b)const
		{
			if (isAreaLight())
				return forward_areaLight(b);
			if (isDirLight())
				return forward_dirLight(b);
			return forward_light_general(b);
		}
		RT_FUNCTION float3 local_contri(const nVertex_device& b) const
		{ 
			float3 vec = b.position - position;
			float3 c_dir = b.isDirLight()? -b.normal: normalize(vec);
			MaterialParameter mat = sysMaterialParameters[materialId];
			mat.color = color; 
			return DisneyEval(mat, normal, dir, c_dir);
		}

	};

	struct contri_reseter
	{
		nVertex current_vertex;
		float3 current_contri;
		float3 dir;
		
		float save_distance;
		float initial_metric;
		bool init;
		RT_FUNCTION
		contri_reseter(nVertex &a, nVertex &b)
		{
			//current_position = b.position;
			dir = normalize(a.position - b.position);
			save_distance = 0;
			current_contri = make_float3(1);
			//materialId = b.materialId;
			current_vertex = b;
			
			float3 diff = a.position - b.position;
			initial_metric = abs(dot(dir, a.normal)) * abs(dot(dir, b.normal)) / dot(diff, diff);

			init = false;
		}
		RT_FUNCTION
		void iteration(nVertex& a)
		{
			if (a.isBrdf == false && current_vertex.isBrdf == false)
			{
				MaterialParameter mat = sysMaterialParameters[current_vertex.materialId];
				mat.color = current_vertex.color;

				float3 diff = a.position - current_vertex.position;
				float3 n_dir = normalize(diff);
				if (init == true)
				{
					init = false;
				}
				else
				{
					current_contri *= DisneyEval(mat, current_vertex.normal, dir, n_dir); 
				}
				current_contri *= abs(dot(n_dir, current_vertex.normal)) * abs(dot(n_dir, a.normal)) / dot(diff, diff);
				dir = -n_dir;
				current_vertex = a;
			}
			else if(a.isBrdf == true && current_vertex.isBrdf == false)
			{
				float3 diff = a.position - current_vertex.position;
				float3 n_dir = normalize(diff);
				current_contri *= current_vertex.color * abs(dot(current_vertex.normal,n_dir));
				save_distance += length(diff);
				dir = -n_dir;
				current_vertex = a;
			}
			else if (a.isBrdf == false && current_vertex.isBrdf == true)
			{
				float3 diff = a.position - current_vertex.position;
				float3 n_dir = normalize(diff);
				current_contri *= current_vertex.color * abs(dot(a.normal, n_dir));
				save_distance += length(diff);
				current_contri *= 1.0 / (save_distance * save_distance);
				//printf("distance %f\n", ENERGY_WEIGHT(current_contri));
				save_distance = 0;
				dir = -n_dir;
				current_vertex = a; 
			}
			else
			{
				float3 diff = a.position - current_vertex.position;
				float3 n_dir = normalize(diff);
				current_contri *= current_vertex.color;
				save_distance += length(diff);  
				dir = -n_dir;
				current_vertex = a;

			}
		}
		RT_FUNCTION
		float3 hit_light(float3 flux)
		{
			current_contri *= flux;
			return current_contri;
		}
		RT_FUNCTION
		float3 merge_with(contri_reseter& a)
		{
			return current_contri * a.current_contri * initial_metric;
		}
	};
}

#endif