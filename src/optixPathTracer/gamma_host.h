#ifndef GAMMA_HOST
#define GAMMA_HOST

#include"gamma_common.h"
//#include"BDPTVertex.h" 
#include<vector>

struct lightSelectionFunction_host :public lightSelectionFunction
{
    void setup(std::vector<float>& Gamma, std::vector<float>& Q_vector)
    {
        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            for (int j = 0; j < SUBSPACE_NUM; j++)
            {
                Matrix[i][j] = Gamma[i * SUBSPACE_NUM + j];
                Matrix[i][j] = Matrix[i][j] * 0.9 + 0.1 * 1.0 / SUBSPACE_NUM;
                
                CMFs[i][j] = Matrix[i][j];
                if (j != 0)
                {
                    //build the CMF cache for first stage sampling
                    CMFs[i][j] += CMFs[i][j - 1];
                }
            }            
        }

        for (int i = 0; i < SUBSPACE_NUM; i++)
        {
            Q[i] = Q_vector[i];
        }
    }

};

#endif