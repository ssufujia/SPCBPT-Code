// ConsoleApplication11.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <vector>
using namespace std;
struct float3
{
    float x, y, z;
    
};

struct float2
{
    float x, y, z;

};
struct PG_training_mat
{
    float3 position;
    float2 uv;
    float lum;
};
struct quad_tree_node
{
    float2 m_min;
    float2 m_max;
    float2 m_mid;
    int child[4];
    bool leaf;
    float lum;
    quad_tree_node(float2 m_min,float2 m_max) :m_min(m_min), m_max(m_max), m_mid((m_min + m_max) / 2) {}
    int whichChild(float2 uv)
    {

        int base = 1;
        int index = 0;
        if (uv.x > m_mid.x)
        {
            index += base;
            base *= 2;
        }
        if (uv.y > m_mid.y)
        {
            index += base;
            base *= 2;
        }
        return index;
    }

    int getChild(int index)
    {
        return child[index];
    }
    int traverse(float2 uv)
    {
        return getChild(whichChild(uv));
    }

};
quad_tree_node quad_tree_nodes[100];
struct quad_tree
{
    int headerID;
    quad_tree_node& getHeader()
    {
        return quad_tree_nodes[headerID];
    }
    quad_tree_node& getNode(int id)
    {
        return quad_tree_nodes[id];
    }
    void training(PG_training_mat &mat)
    {
        auto p = getHeader();
        
        while (p.leaf == false)
        {
            p = getNode(p.traverse(mat.uv));
            p.count++;
        }
        return p;
    }
};
struct Spatio_tree_node
{
    float3 m_min;
    float3 m_max;
    float3 m_mid;
    bool leaf;
    int count;
    int child[8];
    Spatio_tree_node(float3 m_min, float3 m_max) :m_min(m_min), m_max(m_max), m_mid((m_min + m_max) / 2) {}
    int whichChild(float3 pos)
    {
        int base = 1;
        int index = 0;
        if (pos.x > m_mid.x)
        {
            index += base;
            base *= 2;
        }
        if (pos.y > m_mid.y)
        {
            index += base;
            base *= 2;
        }
        if (pos.z > m_mid.z)
        {
            index += base;
            base *= 2;
        }
        return index;
    }
    int getChild(int index)
    {
        return child[index];
    }
    int traverse(float3 pos)
    {
        return getChild(whichChild(pos));
    }
};

struct Spatio_tree
{
    vector<Spatio_tree_node> nodes;
    Spatio_tree_node& getNode(int index)
    {
        return nodes[index];
    }
    Spatio_tree_node& getHeader()
    {
        return nodes[0];
    }
    Spatio_tree_node& traverse(float3 pos)
    {
        auto p = getHeader();
        while (p.leaf == false)
        {
            p = getNode(p.traverse(pos));
        }
        return p;
    }
    void training(PG_training_mat &mat)
    {
        auto p = getHeader();
        p.count++;
        while (p.leaf == false)
        {
            p = getNode(p.traverse(pos));
            p.count++;
        }
        return p;
    }
    void tree_subdivide(int index)
    {
        auto FN = nodes[index];
        FN.leaf = false;
        for (int dx = 0; dx <= 1; dx++)
            for (int dy = 0; dy <= 1; dy++)
                for (int dz = 0; dz <= 1; dz++)
                {
                    int child_index = 4 * dz + 2 * dy + dx;
                    Spatio_tree_node child;
                    float3 box = (FN.m_max - FN.m_min) / 2;
                    float3 min_base;
                    if (dx == 1)
                    {
                        min_base.x += box.x;
                    }
                    if (dy == 1)
                    {
                        min_base.y += box.y;
                    }
                    if (dz == 1)
                    {
                        min_base.z += box.z;
                    }
                    child.m_min = min_base;
                    child.m_max = min_base + box;
                    child.m_mid = (child.m_max + child.m_min) / 2;
                    FN.child[child_index] = nodes.size();
                    nodes.push_back(child);
                }
    } 
}; 
Spatio_tree S_tree;

int main()
{
    S_tree.nodes.push_back(Spatio_tree_node());
    int t = S_tree.getHeader().whichChild(float3());
    printf("%d\n", t);
    std::cout << "Hello World!\n"; 
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
