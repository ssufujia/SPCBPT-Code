#ifndef ZGC_HOST
#define ZGC_HOST
#include"BDPT_STRUCT.h"

#include<random>
#include<vector>
#include<sutil.h>
struct DivTris
{
private:
    std::vector<triangleStruct> tri;
    std::vector<int> divLevel;
    float scene_area;
    int insert(triangleStruct a, int depth = 0)
    {
        float area = a.area();
        if (area > MAX_TRI_AREA )
        {
            triangleStruct b, c;
            b.normalFlip = !a.normalFlip;
            c.normalFlip = !a.normalFlip;
            b.objectId = a.objectId;
            c.objectId = a.objectId;

            float3 newPos = (a.position[1] + a.position[2]) / 2;
            c.position[0] = b.position[0] = newPos;
            b.position[1] = a.position[1];
            b.position[2] = a.position[0];

            c.position[1] = a.position[0];
            c.position[2] = a.position[2];

            float2 newUV = (a.uv[1] + a.uv[2]) / 2;
            c.uv[0] = b.uv[0] = newUV;
            b.uv[1] = a.uv[1];
            b.uv[2] = a.uv[0];

            c.uv[1] = a.uv[0];
            c.uv[2] = a.uv[2];
            insert(b, depth + 1);
            //insert(c,depth + 1);
            return insert(c, depth + 1);
        }
        else
        {
            tri.push_back(a); 
            count++;
            return depth;
        }
    }
public:
    float get_scene_area() { return scene_area; }
    int count;
    DivTris() :count(0)
    {

    }
    triangleStruct& get(int idx)
    {
        return tri[idx];
    }
    void createBuffer(Context& context)
    {
        if (LABEL_BY_STREE)
        {
            Buffer div_tris = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, 1);
            div_tris->setElementSize(sizeof(triangleStruct));
            context["div_tris"]->set(div_tris);

            Buffer div_level = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 1);
            context["divLevel"]->set(div_level);

            Buffer triBase = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, 1);
            context["triBase"]->set(triBase);

        }
        else
        {

            Buffer div_tris = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, MAX_TRIANGLE);
            div_tris->setElementSize(sizeof(triangleStruct));
            context["div_tris"]->set(div_tris);

            Buffer div_level = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, MAX_TRIANGLE);
            context["divLevel"]->set(div_level);

            Buffer triBase = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, MAX_TRIANGLE);
            context["triBase"]->set(triBase);
        }
    }
    void add(triangleStruct a)
    {
        divLevel.push_back(insert(a));
        scene_area += a.area();
    }
    //return the number of div_tris
    int validation(Context &context)
    {
        if (LABEL_BY_STREE)return tri.size();
        triangleStruct* div_tris = reinterpret_cast<triangleStruct*>(context["div_tris"]->getBuffer()->map());
        int * divLevelP = reinterpret_cast<int*>(context["divLevel"]->getBuffer()->map());
        int * triBase = reinterpret_cast<int*>(context["triBase"]->getBuffer()->map());
        for (int i = 0; i < tri.size(); i++)
        {
            div_tris[i] = tri[i];
        }
        for (int i = 0; i < divLevel.size(); i++)
        {
            divLevelP[i] = divLevel[i];
            if (i == 0)
            {
                triBase[i] = 0;
            }
            else
            {
                triBase[i] = triBase[i - 1] + (1 << divLevelP[i - 1]);
            } 
        }


        float area_sum = 0;

        ZoneSampler * zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());

        for (int j = 0; j < SUBSPACE_NUM; j++)
        {
            zoneLVC[j].area = 0;
        }
        for (int i = 0; i < tri.size(); i++)
        {
            float area = tri[i].area();
            area_sum += area;
            zoneLVC[tri[i].zoneNum].area += area;
        } 
        context["zoneLVC"]->getBuffer()->unmap();
        context["div_tris"]->getBuffer()->unmap();
        context["divLevel"]->getBuffer()->unmap();
        context["triBase"]->getBuffer()->unmap();
        context["scene_area"]->setFloat(area_sum);
        return tri.size();
    }

    void rawAlloc(Context &context)
    {
        triangleStruct* triangle_samples = reinterpret_cast<triangleStruct*>(context["triangle_samples"]->getBuffer()->map());
        int triZoneSum = SUBSPACE_NUM - MAX_LIGHT;
        float area = scene_area;
        float area_c = 0.0f;
        for (int i = 0; i < tri.size(); i++)
        {
            tri[i].zoneNum = (area_c / area) * triZoneSum; 
            area_c += tri[i].area(); 
        }
        printf("DivTris:scene area %f is divided into %d triangle\n", scene_area,tri.size());

        context["triangle_sum"]->setInt(tri.size());

        context["triangle_samples"]->getBuffer()->unmap() ;
    }
    void reRawAlloc(bool zoneValid[])
    {
        int triZoneSum = SUBSPACE_NUM - MAX_LIGHT;
        float area = 0.0f, area_c = 0.0f;;

        for (int i = 0; i < tri.size(); i++)
            tri[i].zoneNum = zoneValid[tri[i].zoneNum] == true ? 1 : 0;

        for (int i = 0; i < tri.size(); i++)
            area += tri[i].zoneNum == 1 ? tri[i].area() : 0;

        for (int i = 0; i < tri.size(); i++)
        {
            if (tri[i].zoneNum == 1)
            {
                tri[i].zoneNum = area_c / area * (triZoneSum - 1) + 1;
                area_c += tri[i].area();
            }
        } 
    }
};


void ZoneLVCInit(Context &context, int triangle_num, ZoneMatrix* StableP,DivTris div_tris)
{
    int loopTime = 10;
    RTsize LVCSIZE;
    int *triangle2Zone = new int[triangle_num];
    if (!triangle2Zone)
    {
        printf("so space for triangles %d\n", triangle_num);
        return;
    }
    context["LVC"]->getBuffer()->getSize(LVCSIZE);

    bool validZone[SUBSPACE_NUM] = { false };
    float zone_energy[SUBSPACE_NUM] = { 0 };
    float zone_area[SUBSPACE_NUM] = { 0 };
    float energy_sum = 0.0;
    for (int loop = 0; loop < loopTime; loop++)
    {
        int LVC_frame = context["LVC_frame"]->getInt();
        context["LVC_frame"]->setInt(++LVC_frame);
        context->launch(lightBufferGen, PCPT_CORE_NUM, 1);

        RAWVertex * RAWLVC = reinterpret_cast<RAWVertex*>(context["raw_LVC"]->getBuffer()->map());

        for (int i = 0; i < LVCSIZE; i++)
        {
            if (RAWLVC[i].valid)
            {
                BDPTVertex& vertex = RAWLVC[i].v;
#ifdef INDIRECT_ONLY
                if (vertex.depth == 0)
                    continue;
#endif // INDIRECT_ONLY
#ifdef  BACK_ZONE
                int z_id = vertex.zoneId;
                if (z_id >= SUBSPACE_NUM - MAX_LIGHT)
                {
                    validZone[z_id] = true;
                }
                else
                {
                    int mid_div = (SUBSPACE_NUM - MAX_LIGHT) / 2;
                    if (z_id >= mid_div)
                    {
                        z_id = z_id / 2;
                    }
                    int z1 = z_id * 2;
                    int z2 = z_id * 2 + 1;
                    validZone[z1] = true;
                    validZone[z2] = true;
                    //printf("%d\n",z2);
                    //validZone[vertex.zoneId] = true;

                    float3 flux = vertex.flux / vertex.pdf;
                    float weight = flux.x + flux.y + flux.z;

                    if (isinf(weight) || isnan(weight))
                    {
                        continue;
                    }
                    zone_energy[z1] += weight/2;
                    zone_energy[z2] += weight/2;
                    energy_sum += weight;
                }
#else
                validZone[vertex.zoneId] = true;
#endif //  BACK_ZONE

            }
        }

        context["raw_LVC"]->getBuffer()->unmap();
    }

    triangleStruct * triangle_sample = reinterpret_cast<triangleStruct *>(context["div_tris"]->getBuffer()->map());
    float * tridiv_weight = new float[triangle_num];
    float area = 0.0f, current_area = 0.0f,current_weight = 0.0,weight_sum = 0;

    for (int i = 0; i < triangle_num; i++)
    {
        triangleStruct & tri = triangle_sample[i];
        zone_area[tri.zoneNum] += tri.area();
        if (validZone[tri.zoneNum] == false)
        {
            triangle2Zone[i] = 0;
        }
        else
        {
            triangle2Zone[i] = 1;
            area += tri.area();
        }
    }
    for (int i = 0; i < triangle_num; i++)
    {
        triangleStruct & tri = triangle_sample[i];
        float tri_e = tri.area() / zone_area[tri.zoneNum] * zone_energy[tri.zoneNum];
        tridiv_weight[i] = tri.area() / area + tri_e / energy_sum * 0.8;
        weight_sum += tridiv_weight[i];
    }
    int div_sum = SUBSPACE_NUM - MAX_LIGHT - 1;

    for (int i = 0; i < triangle_num; i++)
    {
        if (triangle2Zone[i] == 0)
        {
            continue;
        }
        triangleStruct & tri = triangle_sample[i];
        int zoneId = current_area / area * div_sum + 1;
        int zoneId2 = current_weight / weight_sum * div_sum + 1;
        current_area += tri.area();
        current_weight += tridiv_weight[i];
        triangle2Zone[i] = zoneId2;
        
    }
    context["div_tris"]->getBuffer()->unmap();
    delete[] triangle2Zone;
    delete[] tridiv_weight;

    int valid_zone_count = 0;
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        if (validZone[i])
        {
            valid_zone_count++;
        }
    }
    div_tris.reRawAlloc(validZone);
    div_tris.validation(context);
    printf("realloc complete,original validZone num:%d\n", valid_zone_count);

    static int pathSum = 0;
    static int vertexSum = 0;
    for (int loop = 0; loop < 1; loop++)
    {
        int LVC_frame = context["LVC_frame"]->getInt();
        context["LVC_frame"]->setInt(++LVC_frame);
        context->launch(lightBufferGen, PCPT_CORE_NUM, 1);

        RAWVertex * RAWLVC = reinterpret_cast<RAWVertex*>(context["raw_LVC"]->getBuffer()->map());
        ZoneMatrix * M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());
        ZoneSampler *zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());
        if (loop == 1)
        {
            for (int i = 0; i < SUBSPACE_NUM; i++)
            {

                zoneLVC[i].size = 0;
                zoneLVC[i].Q = 0;
                pathSum = 0;
                vertexSum = 0;
            }
        }
        for (int i = 0; i < LVCSIZE; i++)
        {
            if (RAWLVC[i].valid)
            {
                BDPTVertex& vertex = RAWLVC[i].v;
                if (RAWLVC[i].v.depth == 0)
                {
                    pathSum++;
#ifdef INDIRECT_ONLY
                    continue;
#endif // INDIRECT_ONLY
                }

                float3 flux = vertex.flux / vertex.pdf;
                float weight = flux.x + flux.y + flux.z;

                if (isinf(weight) || isnan(weight))
                {
                    continue;
                }
                vertexSum++;
                if (vertex.depth >= 1)
                {
                    StableP[vertex.zoneId].r[RAWLVC[i].lastZoneId] += weight;
                    StableP[vertex.zoneId].sum += weight;
                }
                ZoneSampler & vZone = zoneLVC[vertex.zoneId];
                vZone.v[vZone.size % MAX_LIGHT_VERTEX_PER_TRIANGLE] = vertex;
                vZone.Q += weight;
                vZone.size++;

            }
        }

        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            for (int j = 0; j < SUBSPACE_NUM; j++)
            {
                M2[i].r[j] = StableP[i].r[j] / StableP[i].sum;
                M2[i].m[j] = StableP[i].m[j] / StableP[i].sum;
                M2[i].sum += M2[i].r[j];
            }
        }
        context["raw_LVC"]->getBuffer()->unmap();
        context["M2_buffer"]->getBuffer()->unmap();
        context["zoneLVC"]->getBuffer()->unmap();
        context["light_path_sum"]->setInt(pathSum);
    }
}

template<typename T>
int binary_search(T * a, int size, float rnd)
{

    T index = rnd * a[size - 1];
    int mid = size / 2 - 1, l = 0, r = size;
    while (r - l > 1)
    {
        if (index < a[mid])
        {
            r = mid + 1;
        }
        else
        {
            l = mid + 1;
        }
        mid = (l + r) / 2 - 1;
    }
    return l;
}
void UberVertexGen(Context &context ,int vertex_num, std::default_random_engine & random_generator)
{
#ifndef UBER_RMIS
    return;

#endif // UBER_RMIS

    
    triangleStruct * triangle_sample = reinterpret_cast<triangleStruct *>(context["div_tris"]->getBuffer()->map());
    UberLightVertex *res = new UberLightVertex[vertex_num];
    UberZoneLVC *uberLVC = reinterpret_cast<UberZoneLVC *>(context["uberLVC"]->getBuffer()->map()); 
    int triangle_num = context["triangle_sum"]->getInt();
    float* div_weight = new float[triangle_num] ;  

    for (int i = 0; i < triangle_num; i++)
    {
        triangleStruct & tri = triangle_sample[i];
        
        div_weight[i] = tri.area();
        if (i != 0)
            div_weight[i] += div_weight[i - 1]; 
    }   
    float area_sum = div_weight[triangle_num - 1]; 
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        uberLVC[i].realSize = 0;
        uberLVC[i].maxSize = UBER_VERTEX_PER_ZONE; 
    }

    
    for (int i = 0; i < vertex_num; i++)
    {
        float r1 = float(random_generator()) / random_generator.max();
        float r2 = float(random_generator()) / random_generator.max();
        float r3 = float(random_generator()) / random_generator.max();
        int tri_id = binary_search(div_weight, triangle_num, r1); 
        triangleStruct & tri = triangle_sample[tri_id];
        res[i].position = tri.getPos(r2, r3);
        res[i].normal = tri.normal();
        res[i].singlePdf = vertex_num / area_sum;
        res[i].zoneId = tri.zoneNum;

        UberZoneLVC &LVC = uberLVC[res[i].zoneId];
        if (LVC.realSize >= LVC.maxSize)
        {
            LVC.realSize++;
            continue;
        }
        else
        {
            LVC.v[LVC.realSize] = res[i];
            LVC.realSize++;
        }
    } 
    delete[] div_weight;
    delete[] res;
    context["div_tris"]->getBuffer()->unmap();
    context["uberLVC"]->getBuffer()->unmap(); 
//    context->launch(uberVertexProcessProg, SUBSPACE_NUM, UBER_VERTEX_PER_ZONE);
    return;
}

static float corput_compute(int index, int base)
{
    float result = 0;
    float f = 1.0f / base;
    int i = index;
    while (i > 0)
    {
        result = result + f * (i % base);
        i = i / base;
        f = f / base;
    }
    return result;
}
void corputBufferGen(Context& context)
{
    static int count = 20000;    
    static bool isInit = false;
    if (isInit == false)
    {
        Buffer corput_buffer = //sutil::createOutputBuffer(context, RT_FORMAT_UNSIGNED_BYTE4, scene->properties.width, scene->properties.height, use_pbo);
            context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT,count);
        context["corput_buffer"]->set(corput_buffer);
        isInit = true;
    }
    else
    {
        uint frame = context["frame"]->getUint();
        if (frame < count / 2)
        {
            return;
        }
        else
        {
            count *= 2;
            context["corput_buffer"]->getBuffer()->setSize(count);
        }
    }
     
    float* p = reinterpret_cast<float*>(context["corput_buffer"]->getBuffer()->map());
    for (int i = 0; i < count; i++)
    {
        p[i] = corput_compute(i, 2);
    }
    context["corput_buffer"]->getBuffer()->unmap(); 
}

struct TriangleWithID :triangleStruct
{
    int ID;
    int m_materialID;
    float3 m_normal;
    float3 m_center;
    float2 m_uv_center;
    TriangleWithID(triangleStruct& a, int id) :triangleStruct(a), ID(id),m_materialID(a.objectId) 
    {
        m_center = a.center();
        m_normal = a.normal();
        m_uv_center = a.uvCenter();
    }    
    optix::float3 center()
    {
        return m_center;
    }
    optix::float2 uvCenter()
    {
        return m_uv_center;
    }
    optix::float3 normal()
    { 
        return m_normal;
    }
};
struct BlockerNormalSet
{
    std::vector<TriangleWithID> tris;
    float3 normal;
    float area;
    bool valid = true;
    BlockerNormalSet(float3 normal) :normal(normal), area(0)
    {
        tris.clear();
    }
    BlockerNormalSet() {}
    void insert(TriangleWithID tri)
    {
        tris.push_back(tri);
        area += tri.area();
    }
    int size()
    {
        return tris.size();
    }
};
// 0 for 001 1 for 00-1
// 2 for 010 3 for 0-10
// 4 for 100 5 for -100

int3 Tid[] = {
    make_int3(0,0,1),
    make_int3(0,0,-1),
    make_int3(0,1,0),
    make_int3(0,-1,0),
    make_int3(1,0,0),
    make_int3(-1,0,0),
};
//each blocker has 6 normalSet and 6 beside blocker
//for each insert,triangle will be put on tris instead of alloc to normalSet Directly,alloc step will be taken in tris_alloc func
//if the area of normalSet is below the limit ,tri will be taken from p to tris,destroy the tris and alloc again 
struct Blocker
{
    std::vector<BlockerNormalSet> p;
    std::vector<TriangleWithID> tris;
    bool valid;
    float area;
    bool alloc_complete() { return tris.empty(); }
    float3 min_box;
    float3 max_box;
    bool crossValid[6];
    void destroyP(std::vector<BlockerNormalSet>::iterator& pset)
    {
        for (auto iter = pset->tris.begin(); iter != pset->tris.end(); iter++)
        {
            insert(*iter);
        }
        area -= pset->area;
        p.erase(pset);
        tris_alloc();
    }
    //destroy every P without realloc,tri will be put on tris
    void fresh()
    {
        for (auto pset = p.begin(); pset != p.end(); pset++)
        {
            for (auto iter = pset->tris.begin(); iter != pset->tris.end(); iter++)
            {
                insert(*iter);
            }
            area -= pset->area;
            pset->tris.clear();
            //p.erase(pset);
        }
    }
    void insert(TriangleWithID& tri)
    {
        area += tri.area();
        tris.push_back(tri);
    }

    void tris_alloc(TriangleWithID& tri)
    {
        auto tp = p.begin();
        float max_sim = dot(tp->normal, tri.normal());
        for (auto iter = p.begin(); iter != p.end(); iter++)
        {
            float sim = dot(iter->normal, tri.normal());
            if (sim > max_sim)
            {
                tp = iter;
            }
        }
        tp->insert(tri);
    }
    void tris_alloc()
    {
        for (auto iter = tris.begin(); iter != tris.end(); iter++)
        {
            tris_alloc(*iter);
        }
        tris.clear();
    }
    int getSide(TriangleWithID& t)
    {

        bool isInit = true;
        float min_distance = 0.0f;
        int minIdx = 0;

        float candi[6] = {
            abs(t.center().z - max_box.z),
            abs(t.center().z - min_box.z),
            abs(t.center().y - max_box.y),
            abs(t.center().y - min_box.y),
            abs(t.center().x - max_box.x),
            abs(t.center().x - min_box.x) };
        for (int i = 0; i < 6; i++)
        {
            if (crossValid[i] == true && (isInit || candi[i] < min_distance))
            {
                min_distance = candi[i];
                minIdx = i;
                isInit = false;
            }
        }
        return minIdx;
    }
    Blocker() {}
    Blocker(float3 min_box, float3 max_box) :min_box(min_box), max_box(max_box), area(0), valid(true)
    {
        for (int i = 0; i < 6; i++)
        {
            crossValid[i] = true;
        }
        p.push_back(BlockerNormalSet(make_float3(0, 0, 1)));
        p.push_back(BlockerNormalSet(make_float3(0, 0, -1)));
        p.push_back(BlockerNormalSet(make_float3(0, 1, 0)));
        p.push_back(BlockerNormalSet(make_float3(0, -1, 0)));
        p.push_back(BlockerNormalSet(make_float3(-1, 0, 0)));
        p.push_back(BlockerNormalSet(make_float3(1, 0, 0)));
    }
    ~Blocker()
    {
    }
    void realloc(float min_area)
    {
        for (auto it = p.begin(); it != p.end(); it++)
        {
            if (it->area < min_area)
            {
                destroyP(it);
                it = p.begin();
            }
        }
    }
    int size()
    {
        int count = tris.size();
        for (auto it = p.begin(); it != p.end(); it++)
        {
            count += it->size();
        }
        return count;
    }
};
struct Blockers
{
    float3 min_box;
    float3 max_box;
    int3 divSize;
    float3 slice_box;
    std::vector<Blocker> b;
    Blocker invalidB;
    float area;
    Blocker& get(int3 idx)
    {
        if (idx.x < 0 || idx.y < 0 || idx.z < 0 || idx.x >= divSize.x || idx.y >= divSize.y || idx.z >= divSize.z)
        {
            return invalidB;
        }
        int t1 = divSize.z;
        int t2 = t1 * divSize.y;
        return b[idx.x * t2 + idx.y * t1 + idx.z];
    }
    Blocker& operator[](int3 idx)
    {
        return get(idx);
    }
    void breakB(int3 idx)
    {
        get(idx + Tid[1]).crossValid[0] = false;
        get(idx + Tid[0]).crossValid[1] = false;
        get(idx + Tid[3]).crossValid[2] = false;
        get(idx + Tid[2]).crossValid[3] = false;
        get(idx + Tid[5]).crossValid[4] = false;
        get(idx + Tid[4]).crossValid[5] = false;
        Blocker& db = get(idx);
        db.fresh();
        if (!db.alloc_complete())
        {
            for (int i = 0; i < db.tris.size(); i++)
            {
                int next_id = db.getSide(db.tris[i]);
                get(idx + Tid[next_id]).insert(db.tris[i]);
                get(idx + Tid[next_id]).tris_alloc();
            }
            db.tris.clear();
        }
        db.valid = false;
    }
    Blockers(int3 size, float3 min_box, float3 max_box) :min_box(min_box), max_box(max_box), divSize(size), area(0)
    {
        b.resize(divSize.x * divSize.y * divSize.z);
        float x_delta = (max_box.x - min_box.x) / divSize.x;
        float y_delta = (max_box.y - min_box.y) / divSize.y;
        float z_delta = (max_box.z - min_box.z) / divSize.z;

        slice_box = make_float3(x_delta, y_delta, z_delta);
        for (int i = 0; i < divSize.x; i++)
        {
            for (int j = 0; j < divSize.y; j++)
            {
                for (int k = 0; k < divSize.z; k++)
                {

                    float3 slice_min = min_box + make_float3(i, j, k) * slice_box;

                    float3 slice_max = min_box + make_float3(i, j, k) * slice_box + slice_box;
                    get(make_int3(i, j, k)) = Blocker(slice_min, slice_max);
                }
            }
        }
        for (int i = -1; i <= divSize.x; i++)
        {
            for (int j = -1; j <= divSize.y; j++)
            {
                for (int k = -1; k <= divSize.z; k++)
                {
                    if (i == -1 || j == -1 || k == -1 || i == divSize.x  || j == divSize.y  || k == divSize.z )
                    {
                        breakB(make_int3(i, j, k));
                    }
                }
            }
        }
    }
    void insert(TriangleWithID& tri)
    {
        float3 c = tri.center() - min_box;
        int3 idx = make_int3(c.x / slice_box.x, c.y / slice_box.y, c.z / slice_box.z);
        idx.x = clamp(idx.x,0, divSize.x-1);
        idx.y = clamp(idx.y,0, divSize.y-1);
        idx.z = clamp(idx.z,0, divSize.z-1);
        area += tri.area();
        get(idx).insert(tri);

    }
    void validation()
    {
        int begin_size = size();
        float min_area = area / (6 * divSize.x * divSize.y * divSize.z);
        //min_area = 0.0001;
        for (int i = 0; i < divSize.x; i++)
        {
            for (int j = 0; j < divSize.y; j++)
            {
                for (int k = 0; k < divSize.z; k++)
                {
                    int3 idx = make_int3(i, j, k);
                    if (get(idx).area < min_area)
                    {
                        breakB(idx);
                    }
                }
            }
        }
        for (auto it = b.begin(); it != b.end(); it++)
        {
            it->tris_alloc();
            it->realloc(min_area);
        }

        if (size() != begin_size)
        {
            printf("BUG FOUND\n");
        }

    }
    int size()
    {
        int count = 0;
        for (auto it = b.begin(); it != b.end(); it++)
        {
            count += it->size();
        }
        return count;
    }
};
void tri_alloc_by_grid(Context& context, DivTris& div_tris)
{
    if (LABEL_BY_STREE)return;
    triangleStruct* div_tris_p = reinterpret_cast<triangleStruct*>(context["div_tris"]->getBuffer()->map());
    float3 min_box = context["min_box"]->getFloat3();
    float3 max_box = context["max_box"]->getFloat3();
    int3 blkIdx = make_int3(10, 10, 10);
    {
        int n = 2000;
        float3 box = max_box - min_box;
        float V = box.x * box.y * box.z / n;
        float lbox = pow(V, 1.0 / 3);
        blkIdx = make_int3(int(box.x / lbox), int(box.y / lbox), int(box.z / lbox));
    }
    Blockers blockers(blkIdx, min_box, max_box);
    int tricount = 0;
    for (int i = 0; i < div_tris.count; i++)
    {
        TriangleWithID tri(div_tris.get(i), i);
        blockers.insert(tri);
        tricount++;
    }
    blockers.validation();
    int subspaceId = 0;
    for (auto it = blockers.b.begin(); it != blockers.b.end(); it++)
    {
        if (it->valid == false)
        {
            continue;
        }
        for (auto nit = it->p.begin(); nit != it->p.end(); nit++)
        {
            if(true)
            {
                for (auto trit = nit->tris.begin(); trit != nit->tris.end(); trit++)
                {
                    div_tris_p[trit->ID].zoneNum = subspaceId;
                }
            }
            subspaceId++;
        }
    }

    printf("GridAlloc: %d-%d triangles is divided into %d subspace\n", blockers.size(),div_tris.count, subspaceId);
    context["div_tris"]->getBuffer()->unmap();
}

std::vector<BDPTVertex> get_random_eye_vertex(Context& context,int vertex_num)
{
    uint launch_width = 20, launch_height = 20, launch_depth = 5, evc_frame = 0;
    context["EVC_frame"]->setUint(evc_frame);
    context["EVC_width"]->setUint(launch_width);
    context["EVC_height"]->setUint(launch_height);
    context["EVC_max_depth"]->setUint(launch_depth);

    assert(vertex_num != 0);
    std::vector<BDPTVertex> res;
    while (res.size() < vertex_num)
    {

        launch_width = context["EVC_width"]->getUint();
        launch_height = context["EVC_height"]->getUint();
        launch_depth = context["EVC_max_depth"]->getUint();
        evc_frame = context["EVC_frame"]->getUint(); 
        context->launch(EVCLaunch, launch_width, launch_height); 
        evc_frame++;
        context["EVC_frame"]->setUint(evc_frame);

        RAWVertex* p = reinterpret_cast<RAWVertex*> (context["raw_LVC"]->getBuffer()->map());
        for (int i = 0; i < launch_depth * launch_width * launch_height;i++)
        {
            if (p[i].valid == true)
            {
                res.push_back(p[i].v);
            }
        } 
        context["raw_LVC"]->getBuffer()->unmap();
    }
    if (res.size() == vertex_num)
    {
        return res;
    }
    std::vector<BDPTVertex> res2;
    res2.push_back(res[0]);
    float select_step = float(res.size()) / vertex_num;
    int selectPos = select_step;

    int pos = 1;
    while (res2.size() != vertex_num)
    {
        if (pos > selectPos||vertex_num - res2.size() >= res.size() - pos)
        {
            res2.push_back(res[pos]);
            selectPos += select_step;
        }
        pos++;
    }
    assert(res2.size() == vertex_num);
    return res2;
}

int LT_trace(Context& context, int core = PCPT_CORE_NUM, int core_size = LIGHT_VERTEX_PER_CORE, bool M_FIX = false)
{
    static bool is_init = true;
    if (is_init == true)
    {
        is_init = false;
        context["LVC_frame"]->setInt(0);
    }

    int target_raw_size = core * core_size;
    RTsize raw_size;
    context["raw_LVC"]->getBuffer()->getSize(raw_size);
    if (target_raw_size > raw_size)
    {
        context["raw_LVC"]->getBuffer()->setSize(target_raw_size);
    }
    int LVC_frame = 0;
    LVC_frame = context["LVC_frame"]->getInt();
    context["LVC_frame"]->setInt(++LVC_frame); 

    context["LT_CORE_NUM"]->setInt(core);
    context["LT_CORE_SIZE"]->setInt(core_size);
    context["M_FIX"]->setInt(M_FIX == true ? 1 : 0); 
    context->launch(lightBufferGen, core, 1); 
     
    return core_size * core;
}


std::vector<BDPTVertex> get_random_light_vertex(Context& context, int vertex_num)
{
    assert(vertex_num != 0);
    std::vector<BDPTVertex> res;
    int M_count = 0;
    while (res.size() < vertex_num)
    {
        int trace_num = LT_trace(context);
        
        RAWVertex* p = reinterpret_cast<RAWVertex*> (context["raw_LVC"]->getBuffer()->map());
        for (int i = 0; i < trace_num; i++)
        {
            if (p[i].valid == true&&p[i].v.type != DIRECTION)
            {
                if (p[i].v.depth == 0)
                {
                    M_count += 1;
                }
                res.push_back(p[i].v);
            }
        }
        context["raw_LVC"]->getBuffer()->unmap();
    }

    for (auto p = res.begin(); p != res.end(); p++)
    {
        p->pdf *= float(M_count) / res.size() * vertex_num / res.size();
    }
    if (res.size() == vertex_num)
    {
        return res;
    }
    std::vector<BDPTVertex> res2;
    res2.push_back(res[0]);
    float select_step = float(res.size()) / vertex_num;
    int selectPos = select_step;

    int pos = 1;
    while (res2.size() != vertex_num)
    {
        if (pos > selectPos || vertex_num - res2.size() >= res.size() - pos)
        {
            res2.push_back(res[pos]);
            selectPos += select_step;
        }
        pos++;
    }
    assert(res2.size() == vertex_num);
    return res2;
}

class MeshClassifier
{
public:
    void virtual insert(TriangleWithID& a) = 0;
    void virtual validation() = 0;
};


struct SLICClassifier_MeshGroupWithFlagPoint
{
private:
    float3 position;
    float3 normal;
    float2 uv;
    int objectID;
    int materialID;
    float m_area;
    float scene_max_length;
    float space_factor;
public:

    std::vector<TriangleWithID> tris;
    std::vector<TriangleWithID> tris_back;
    
    void insert(TriangleWithID& tri)
    {
        //if (dot(tri.normal(), normal) > 0)
            tris.push_back(tri);
        //else
        //    tris_back.push_back(tri);
        m_area += tri.area();
    }
    int getMaterialID() { return materialID; }
    float diffEval(TriangleWithID& tri)
    {
        float diff = 0.0;
        float space_inch = 2.5;
        float material_diff = 2;
        float3 diffVec[] = { tri.position[0] - position ,tri.position[1] - position,tri.position[2] - position};
        float space_diff = 0, uv_diff = 0, mat_diff = 0, normal_diff = 0;
        for (int i = 0; i < 3; i++)
        {
            space_diff += dot(diffVec[i], diffVec[i]);
        }
        diff += space_diff;
        space_diff /= space_factor / 3;

        float uv_diff_length = length(uv - tri.uvCenter());
        uv_diff = uv_diff_length * uv_diff_length;
        uv_diff = uv_diff > 1 ? 1 : uv_diff;
        
        mat_diff = tri.m_materialID == materialID ? 0 : 0.25;
        //uv_diff = tri.m_materialID == materialID ? 0 : uv_diff;

        normal_diff = -dot(normal, tri.normal());
        //normal_diff = 1 - abs(dot(normal, tri.normal()));
        if (normal_diff > -0.0)
        {
            normal_diff += 1;
            normal_diff *= normal_diff;
        }
        //printf("space diff %f sum diff %f\n", space_diff, mat_diff + normal_diff + space_diff);
        return mat_diff + normal_diff + space_diff;


        diff -= space_inch * dot(normal, tri.normal());
        uv_diff = length(uv - tri.uvCenter());
        if (uv_diff > 1)
        {
            uv_diff = 1;
        }
        uv_diff = 0;
        diff += (tri.m_materialID == materialID ? uv_diff * space_inch : space_inch) * material_diff;
        return diff;
    }
    SLICClassifier_MeshGroupWithFlagPoint(BDPTVertex &a,float scene_max_length = 5):
        position(a.position),normal(a.normal),m_area(0),materialID(a.materialId),uv(a.uv),scene_max_length(scene_max_length),space_factor(scene_max_length * scene_max_length / 25){}
    float area()
    {
        return m_area;
    }
    int size()
    {
        return tris.size() + tris_back.size();
    }
    void set_scene_max_length(float a)
    {
        space_factor = a * a / 16;
        return;
    }
    std::vector<TriangleWithID> get_all_tris()
    {
        std::vector<TriangleWithID> vec3;
        vec3.insert(vec3.end(), tris.begin(), tris.end());
        vec3.insert(vec3.end(), tris_back.begin(), tris_back.end());

        return vec3;
    }
    void inverse_normal()
    {
        normal = -normal;
    }
};

class SimpleSLICClassifier :MeshClassifier
{
private:
    std::vector<TriangleWithID> tris;
    std::vector<SLICClassifier_MeshGroupWithFlagPoint> groups;
    std::vector<std::vector<SLICClassifier_MeshGroupWithFlagPoint>::iterator> zone_groups[SUBSPACE_NUM];
    float minArea;
    float scene_max_length;
    void alloc_tri(TriangleWithID& tri)
    {
        if (true||zone_groups[tri.m_materialID].size() == 0 )
        {
            float minDiff = groups[0].diffEval(tri);
            auto minP = groups.begin();
            for (auto p = groups.begin(); p != groups.end(); p++)
            {
                float diff = p->diffEval(tri);
                if (diff < minDiff)
                {
                    minDiff = diff;
                    minP = p;
                }
            }
            minP->insert(tri);
        }
        else
        {
            auto& itgroup = zone_groups[tri.m_materialID];
            float minDiff = itgroup[0]->diffEval(tri);
            auto minP = itgroup.begin();
            for (auto p = itgroup.begin(); p != itgroup.end(); p++)
            {
                float diff = (*p)->diffEval(tri);
                if (diff < minDiff)
                {
                    minDiff = diff;
                    minP = p;
                }
            }
            (*minP)->insert(tri);
        }
    }
    void destroyG(std::vector<SLICClassifier_MeshGroupWithFlagPoint>::iterator g)
    {
        auto group_tris = g->get_all_tris();
        tris.insert(tris.end(), group_tris.begin(), group_tris.end());
        groups.erase(g);
    }
    void alloc()
    {
        for (auto p = tris.begin(); p != tris.end(); p++)
        {
            alloc_tri(*p);
        }
        tris.clear();
    }
public:
    SimpleSLICClassifier() :MeshClassifier(), scene_max_length(5) {}
    void insert(TriangleWithID& tri)
    {
        tris.push_back(tri);
    }
    void set_flagPoints(std::vector<BDPTVertex>& a)
    {
        for (auto p = a.begin(); p != a.end(); p++)
        {
            auto flags = SLICClassifier_MeshGroupWithFlagPoint(*p, scene_max_length);
            groups.push_back(flags);
            flags.inverse_normal();
            groups.push_back(flags);
            
            
        }
        for (auto p = groups.begin(); p != groups.end(); p++)
        {
          //  zone_groups[p->getMaterialID()].push_back(p);
        }
    }
    void setMinArea(float min_area)
    {
        minArea = min_area;
    }
    void validation()
    {

        alloc();
        printf("firstStage_alloc_complete\n");
        for (auto p = groups.begin(); p != groups.end(); p++)
        {
            if (p->area() < minArea)
            {
                printf("secondStage_alloc_begin allocsize: %d\n",p->size());
                destroyG(p);
                alloc();
                printf("secondStage_alloc_complete\n");
                p = groups.begin();
            }
        }
        printf("validationComplete\n");
    }
    std::vector<std::vector<TriangleWithID>> get_classify_result()
    {
        std::vector<std::vector<TriangleWithID>> res;
        for (auto p = groups.begin(); p != groups.end(); p++)
        {
            res.push_back(p->tris);
            //res.push_back(p->tris_back);
        }
        return res;
    }
    int size()
    {
        int count = tris.size();
        for (auto p = groups.begin(); p != groups.end(); p++)
        {
            count += p->size();
        }
        return count;
    }
    void set_scene_max_length(float a)
    {
        scene_max_length = a;
        for (auto p = groups.begin(); p != groups.end(); p++)
        {
            p->set_scene_max_length(a);
        }
        return;
    }
};

void tri_alloc_by_simpleSLIC(Context& context, DivTris& div_tris)
{
    if (LABEL_BY_STREE)return;
    SimpleSLICClassifier classifier;
    classifier.set_scene_max_length(context["sceneMaxLength"]->getFloat());
    //classifier.set_scene_max_length(4);
    //int max_class = (SUBSPACE_NUM-MAX_LIGHT)/2;
    //max_class *= 1.5;
    int max_class = SUBSPACE_NUM / 2;
#ifdef BACK_ZONE
    max_class /= 2;
#endif
    int eyeFlags = max_class * 0.75;
    int lightFlags = max_class - eyeFlags;
    auto flagPoints = get_random_eye_vertex(context, eyeFlags);
    auto LPoints = get_random_light_vertex(context, lightFlags);
    flagPoints.insert(flagPoints.end(),LPoints.begin(), LPoints.end());
    classifier.set_flagPoints(flagPoints);
    classifier.setMinArea(div_tris.get_scene_area() / max_class);
    classifier.setMinArea(div_tris.get_scene_area() / max_class / 10);
    //classifier.setMinArea(2);
    //classifier.setMinArea(0);
    int tricount = 0;
    for (int i = 0; i < div_tris.count; i++)
    {
        TriangleWithID tri(div_tris.get(i), i);
        classifier.insert(tri);
        tricount++;
    }
    classifier.validation();
    auto classifyResult = classifier.get_classify_result();
    int subspaceId = 0;
    triangleStruct* div_tris_p = reinterpret_cast<triangleStruct*>(context["div_tris"]->getBuffer()->map());
    for (auto it = classifyResult.begin(); it != classifyResult.end(); it++)
    {
        for (auto nit = it->begin(); nit != it->end(); nit++)
        {
            div_tris_p[nit->ID].zoneNum = subspaceId; 
            //printf("objectId %d", nit->objectId);
        }
        subspaceId++;
    }

    printf("GridAlloc: %d-%d triangles is divided into %d subspace\n", classifier.size(), div_tris.count, subspaceId);
    if (subspaceId > SUBSPACE_NUM - MAX_LIGHT)
    {
        for(int i=0;i<100;i++)
            printf("WARNNING,zone alloc EXCEED LIMIT\n");
    }
    context["div_tris"]->getBuffer()->unmap();
}

struct simpleSlicWithGPU
{
    float min_area;
    int max_label;
    Buffer label_buffer;
    Buffer unlabel_index_buffer;
    Buffer cluster_center_buffer;
    Buffer div_tris_buffer;
    void init(Context& context)
    { 
        max_label = SUBSPACE_NUM - MAX_LIGHT;
        label_buffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_INT, MAX_TRIANGLE);
        context["slic_label_buffer"]->set(label_buffer);
        unlabel_index_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT, MAX_TRIANGLE);
        context["slic_unlabel_index_buffer"]->set(unlabel_index_buffer);
        cluster_center_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER, max_label);
        cluster_center_buffer->setElementSize(sizeof(BDPTVertex));
        context["slic_cluster_center_buffer"]->set(cluster_center_buffer);
        div_tris_buffer = context["div_tris"]->getBuffer();
        context["slic_tris"]->setBuffer(div_tris_buffer);
    }
    void segmentation(Context& context, DivTris& div_tris)
    {
        if (LABEL_BY_STREE)return;
        {//space factor setting
            float space_normal_item = context["sceneMaxLength"]->getFloat();
            //space_normal_item /= 10;
            space_normal_item *= space_normal_item / 16;
            space_normal_item = 1;
            context["space_normal_item"]->setFloat(space_normal_item);
             
            min_area = div_tris.get_scene_area() / max_label / 100;
        }

        {//flags Setting
            int face_label_num = 0.5 * max_label;
            int eyeFlags = 0.75 * face_label_num;
            int lightFlags = face_label_num - eyeFlags;
            auto flagPoints = get_random_eye_vertex(context, eyeFlags);
            printf("segmentation begin\n");
            auto LPoints = get_random_light_vertex(context, lightFlags); 

            flagPoints.insert(flagPoints.end(), LPoints.begin(), LPoints.end());
            auto flags_p = reinterpret_cast<BDPTVertex*>(cluster_center_buffer->map());
            for (int i = 0; i < flagPoints.size(); i++)
            {
                int id_i = 2 * i, id_j = 2 * i + 1;
                flags_p[id_i] = flagPoints[i];
                flags_p[id_j] = flagPoints[i];
                flags_p[id_j].normal *= -1;
            }
            cluster_center_buffer->unmap();
            context["slic_cluster_num"]->setInt(2 * flagPoints.size());
        }


        auto unlabel_id_p = reinterpret_cast<int*>(unlabel_index_buffer->map());
        for (int i = 0; i < div_tris.count; i++)
        {
            unlabel_id_p[i] = i;
        }
        unlabel_index_buffer->unmap();

        int label_count = div_tris.count;
        context->launch(SlicProg, label_count, 1);



        auto label_p = reinterpret_cast<int*>(label_buffer->map());
        unlabel_id_p = reinterpret_cast<int*>(unlabel_index_buffer->map());
        auto tris_p = reinterpret_cast<triangleStruct*>(div_tris_buffer->map());
        float ss_areas[SUBSPACE_NUM] = {0};
        for (int i = 0; i < label_count; i++)
        {
            auto& tri = tris_p[unlabel_id_p[i]];
            tri.zoneNum = label_p[i];
            ss_areas[label_p[i]] += tri.area();
        }
        label_count = 0;
        for (int i = 0; i < div_tris.count; i++)
        {
            auto& tri = tris_p[i];
            if (ss_areas[tri.zoneNum] < min_area)
            {
                unlabel_id_p[label_count] = i;
                label_count++;
            }
        }
        label_buffer->unmap();
        unlabel_index_buffer->unmap();
        div_tris_buffer->unmap();
        context->launch(SlicProg, label_count, 1);

        label_p = reinterpret_cast<int*>(label_buffer->map());
        unlabel_id_p = reinterpret_cast<int*>(unlabel_index_buffer->map());
        tris_p = reinterpret_cast<triangleStruct*>(div_tris_buffer->map()); 
        for (int i = 0; i < label_count; i++)
        {
            auto& tri = tris_p[unlabel_id_p[i]];
            tri.zoneNum = label_p[i]; 
        } 
        label_buffer->unmap();
        unlabel_index_buffer->unmap();
        div_tris_buffer->unmap();
        printf("segmentation complete\n");
    }
};
simpleSlicWithGPU slic_gpu_api;

struct UV_GRID:std::vector<float>
{
    Buffer zgc_gridInfo_buffer;
    void table_write(Context& context,int begin_label,int end_label)
    {
        int label_sum = end_label - begin_label;
        auto& t = (*this);
        float area = 0;
        for (auto p = begin(); p != end(); p++)
        {
            area += *p;
           // printf("%f %f\n", area,*p);
        }
        std::vector<int> table_slot;
        std::vector<int> table_slot_accm;
        for (int i = 0; i < size(); i++)
        {
            int slot = t[i] / area * label_sum;
            int std_slot = 2;
            while (slot >= std_slot)
            {
                std_slot *= 2;
            }
            if (i == 0)
            {
                table_slot_accm.push_back(0);
            }
            else
            {
                table_slot_accm.push_back(table_slot.back() + table_slot_accm.back());
            }
            table_slot.push_back(std_slot / 2);
        }
        zgc_gridInfo_buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, size());
        auto p = reinterpret_cast<int3*> (zgc_gridInfo_buffer->map());
        for (int i = 0; i < size(); i++)
        {
            int divLevel = 0;
            int slot = table_slot[i];
            while (slot > 1)
            {
                divLevel++;
                slot /= 2;
            }
            p[i] = make_int3(table_slot_accm[i] + begin_label, table_slot[i], divLevel);
         //   printf("%d %d %d\n", p[i].x, p[i].y, p[i].z);
        }
        zgc_gridInfo_buffer->unmap();
        context["zgc_gridInfo"]->setBuffer(zgc_gridInfo_buffer);
    }
} uv_grid;

void Q_update(Context& context,int light_path_sum)
{ 
    context["Q_light_path_sum"]->setInt(light_path_sum);
    ZoneSampler* zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());
    for (int i = 0; i < SUBSPACE_NUM; i++)
    { 
        zoneLVC[i].Q = zoneLVC[i].Q_old; 
    }

    context["zoneLVC"]->getBuffer()->unmap();
}
void load_Q(Context& context, std::vector<float>& Q)
{
    context["Q_light_path_sum"]->setInt(1);
#ifdef BEDROOM
    context["Q_light_path_sum"]->setInt(300);

#endif
    ZoneSampler* zoneLVC = reinterpret_cast<ZoneSampler*>(context["zoneLVC"]->getBuffer()->map());
    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        zoneLVC[i].Q = Q[i];
    }

    context["zoneLVC"]->getBuffer()->unmap();

}
void E_update(Context& context, ZoneMatrix* StableP, ZoneMatrix* eyeM2P,float uniform_rate = 0.2)
{
    ZoneMatrix* M2 = reinterpret_cast<ZoneMatrix*>(context["M2_buffer"]->getBuffer()->map());

    for (int i = 0; i < SUBSPACE_NUM; i++)
    {
        M2[i].sum = 0.0;
        for (int j = 0; j < SUBSPACE_NUM; j++)
        {
            float lerp_rate = 1.0;
#ifdef EYE_DIRECT
            lerp_rate = 0.1;
#endif // EYE_DIRECT


            M2[i].r[j] = StableP[i].r[j] / StableP[i].sum * lerp_rate + eyeM2P[i].r[j] / eyeM2P[i].sum * (1.0 - lerp_rate);
            M2[i].r[j] = lerp(M2[i].r[j], 1.0 / SUBSPACE_NUM, uniform_rate);


            //M2[i].r[j] = 1;
        }
        M2[i].validation();
    }
    context["M2_buffer"]->getBuffer()->unmap();
}
#endif