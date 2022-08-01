#ifndef KD_TREE_H
#define KD_TREE_H
#include"BDPT_STRUCT.h"

#include<random>
#include<vector>
#include"select.h"

//for kd tree
static int max_component(float3 a)
{
    if (a.x > a.y) {
        if (a.x > a.z) {
            return 0;
        }
        else {
            return 2;
        }
    }
    else {
        if (a.y > a.z) {
            return 1;
        }
        else {
            return 2;
        }
    }
}
template<typename T>
void buildKDTree(T** directors, int start, int end, int depth, T* kd_tree, int current_root, splitChoice split_choice, float3 bbmin, float3 bbmax)
{
    if (end - start == 0) {
        kd_tree[current_root].axis = PPM_NULL;
        kd_tree[current_root].valid = false;
        return;
    }

    // If we have a single photon
    if (end - start == 1) {
        directors[start]->axis = PPM_LEAF;
        kd_tree[current_root] = *(directors[start]);
        return;
    }

    // Choose axis to split on
    int axis;
    switch (split_choice) {
    case RoundRobin:
    {
        axis = depth % 3;
    }
    break;
    case HighestVariance:
    {
        float3 mean = make_float3(0.0f);
        float3 diff2 = make_float3(0.0f);
        for (int i = start; i < end; ++i) {
            float3 x = directors[i]->position;
            float3 delta = x - mean;
            float3 n_inv = make_float3(1.0f / (static_cast<float>(i - start) + 1.0f));
            mean = mean + delta * n_inv;
            diff2 += delta * (x - mean);
        }
        float3 n_inv = make_float3(1.0f / (static_cast<float>(end - start) - 1.0f));
        float3 variance = diff2 * n_inv;
        axis = max_component(variance);
    }
    break;
    case LongestDim:
    {
        float3 diag = bbmax - bbmin;
        axis = max_component(diag);
    }
    break;
    default:
        axis = -1;
        std::cerr << "Unknown SplitChoice " << split_choice << " at " << __FILE__ << ":" << __LINE__ << "\n";
        exit(2);
        break;
    }

    int median = (start + end) / 2;
    T** start_addr = &(directors[start]);

    switch (axis) {
    case 0:
        select<T*, 0>(start_addr, 0, end - start - 1, median - start);
        directors[median]->axis = PPM_X;
        break;
    case 1:
        select<T*, 1>(start_addr, 0, end - start - 1, median - start);
        directors[median]->axis = PPM_Y;
        break;
    case 2:
        select<T*, 2>(start_addr, 0, end - start - 1, median - start);
        directors[median]->axis = PPM_Z;
        break;
    }

    float3 rightMin = bbmin;
    float3 leftMax = bbmax;
    if (split_choice == LongestDim) {
        float3 midPoint = (*directors[median]).position;
        switch (axis) {
        case 0:
            rightMin.x = midPoint.x;
            leftMax.x = midPoint.x;
            break;
        case 1:
            rightMin.y = midPoint.y;
            leftMax.y = midPoint.y;
            break;
        case 2:
            rightMin.z = midPoint.z;
            leftMax.z = midPoint.z;
            break;
        }
    }

    kd_tree[current_root] = *(directors[median]);
    buildKDTree(directors, start, median, depth + 1, kd_tree, 2 * current_root + 1, split_choice, bbmin, leftMax);
    buildKDTree(directors, median + 1, end, depth + 1, kd_tree, 2 * current_root + 2, split_choice, rightMin, bbmax);
}
class KD_tree
{
private:
    std::vector<KDPos> nodes;
    int build_tree()
    {
        splitChoice split_choice = RoundRobin;
        int N = nodes.size();
        int MAX_N = 2 * N;
        KDPos* Cache_pos = new KDPos[MAX_N];
        for (int i = 0; i < MAX_N; i++)
        {
            Cache_pos[i].valid = false;
        }
        KDPos** temp_A = new KDPos * [N];
        KDPos* temp_B = new KDPos[N];
        for (int i = 0; i < N; ++i)
        {
            auto& label = nodes[i];
            temp_B[i].valid = true;
            temp_B[i].position = label.position;
            temp_B[i].label = label.label;
            temp_A[i] = &(temp_B[i]);
        }
        float3 bbmin = make_float3(0.0f);
        float3 bbmax = make_float3(0.0f);
        if (split_choice == LongestDim) {
            bbmin = make_float3(std::numeric_limits<float>::max());
            bbmax = make_float3(-std::numeric_limits<float>::max());
            // Compute the bounds of the photons
            for (unsigned int i = 0; i < N; ++i) {
                float3 position = (*temp_A[i]).position;
                bbmin = fminf(bbmin, position);
                bbmax = fmaxf(bbmax, position);
            }
        }
        buildKDTree(temp_A, 0, N, 0, Cache_pos, 0, split_choice, bbmin, bbmax);
        nodes.clear();
        for (int i = 0; i < MAX_N; i++)
        {
            nodes.push_back(Cache_pos[i]);
        }

        delete[] temp_A;
        delete[] temp_B;
        delete[] Cache_pos;
        return N;
    }
public:
    KD_tree(std::vector<float3> originPos)
    {
        construct(originPos);
    }
    KD_tree() {
        nodes.clear();
    };
    int construct(std::vector<float3> originPos)
    {
        nodes.clear();
        int id = 0;
        for (auto p = originPos.begin(); p != originPos.end(); p++)
        {
            nodes.push_back(KDPos(*p, id));
            id++;
        }
        build_tree();
        return nodes.size();
    }
    int construct(std::vector<KDPos> originPos,bool need_rebuild = true)
    {
        nodes.clear();
        for (auto p = originPos.begin(); p != originPos.end(); p++)
        {
            nodes.push_back(*p);
        }
        if(need_rebuild == true)
            build_tree();
        return nodes.size();
    }

    std::vector<int> find(float3 position, int k)
    {
        std::priority_queue<std::pair<float, int> > ans;
        unsigned int stack[25];
        float dis_stack[25];
        unsigned int stack_current = 0;
        unsigned int node = 0; // 0 is the start

#define push_node(N) stack[stack_current++] = (N)
#define pop_node()   stack[--stack_current]

        float block_min = 0.0;
        dis_stack[stack_current] = 0.0;
        push_node(0);

        do {
            if (ans.size() >= k && ans.top().first < block_min)
            {
                node = pop_node();
                block_min = dis_stack[stack_current];
                continue;
            }
            KDPos& currentVDirector = nodes[node];
            uint axis = currentVDirector.axis;
            if (!(axis & PPM_NULL)) {

                float3 vd_position = currentVDirector.position;
                float3 diff = position - vd_position;
                float distance2 = dot(diff, diff);

                auto o_pair = std::make_pair(distance2, node);
                ans.push(o_pair);
                if (ans.size() > k)
                {
                    ans.pop();
                }

                // Recurse
                if (!(axis & PPM_LEAF)) {
                    float d;
                    if (axis & PPM_X) d = diff.x;
                    else if (axis & PPM_Y) d = diff.y;
                    else                      d = diff.z;

                    // Calculate the next child selector. 0 is left, 1 is right.
                    int selector = d < 0.0f ? 0 : 1;
                    if (d * d < ans.top().first) {
                    dis_stack[stack_current] = d * d;
                    push_node((node << 1) + 2 - selector);
                    }

                    node = (node << 1) + 1 + selector;
                }
                else {
                    node = pop_node();
                    block_min = dis_stack[stack_current];
                }
            }
            else {
                node = pop_node();
                block_min = dis_stack[stack_current];
            }
        } while (node);

        std::vector<int> r_ans;
        while (ans.size() > 0)
        {
            int r_id = nodes[ans.top().second].label;
            r_ans.push_back(r_id); 
            ans.pop();
        }
        return r_ans;
    }
};
#endif