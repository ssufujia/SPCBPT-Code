/*Copyright (c) 2016 Miles Macklin

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.*/

#pragma once

#include <stdio.h>

#include <map>
#include <set>
#include <string>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "commonStructs.h"
#include "material_parameters.h"
#include "properties.h"
#include "light_parameters.h"
#include "Picture.h"
#include "Texture.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdint.h>


struct Scene
{
	std::vector<std::string> mesh_names;
	std::vector<std::string> uv_mesh_names;
	std::vector<optix::Matrix4x4> transforms;
	std::vector<MaterialParameter> materials;
	std::vector<LightParameter> lights;
	std::vector<Texture> textures;
	std::map<int, std::string> texture_map;
	std::string env_file;
	Properties properties;
    optix::float3 eye;
    optix::float3 lookat;
    optix::float3 up;
    float fov;
    bool use_camera;
	bool use_geometry_normal;
	float env_factor;
	optix::float3 dirLightDir;
	Scene() :use_geometry_normal(false) {};
};

Scene* LoadScene(const char* filename);