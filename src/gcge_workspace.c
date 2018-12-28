/*
 * =====================================================================================
 *
 *       Filename:  gcge_workspace.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时57分13秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "gcge_workspace.h"

//给GCGE_Para结构中各个部分分配空间
void GCGE_WORKSPACE_Create(GCGE_WORKSPACE **workspace)
{
    (*workspace) = (GCGE_WORKSPACE*)malloc(sizeof(GCGE_WORKSPACE));
    //对GCGE算法中用到的一些参数进行初始化
    (*workspace)->max_dim_x = 0;
    (*workspace)->dim_x     = 0;
    (*workspace)->dim_xp    = 0;
    (*workspace)->dim_xpw   = 0;
    (*workspace)->conv_bs   = 0;
    (*workspace)->unconv_bs = 0;
    (*workspace)->num_soft_locked_in_last_iter = 0; 

    //近似特征值
    (*workspace)->eval          = NULL;
    //小规模的临时工作空间
    (*workspace)->subspace_dtmp = NULL;
    (*workspace)->subspace_itmp = NULL;
    //存储子空间矩阵的特征向量
    (*workspace)->subspace_evec = NULL;
    //用于存储子空间矩阵
    (*workspace)->subspace_matrix = NULL;

    //存储前nev个特征对中未收敛的特征对编号
    (*workspace)->unlock = NULL;
    //正交化时用到的临时GCGE_INT*型空间,用于临时存储非0的编号
    (*workspace)->orth_ind = NULL;
    (*workspace)->V = NULL;
    (*workspace)->V_tmp = NULL;
    (*workspace)->RitzVec = NULL;
    (*workspace)->CG_p = NULL;
    (*workspace)->CG_r = NULL;
    
    //GCGE_STATISTIC_Para用于统计各部分时间
}
void GCGE_WORKSPACE_Setup(GCGE_WORKSPACE *workspace, GCGE_PARA *para, GCGE_OPS *ops, void *A)
{
    GCGE_INT   i,    nev = para->nev,
               block_size = para->block_size,
               max_dim_x_tmp = (nev*1.25 < nev+8) ? (nev*1.25) : (nev+8),
               max_dim_x = (max_dim_x_tmp > nev + 3)? max_dim_x_tmp : (nev+3);

    GCGE_INT   V_size = max_dim_x + 2 * block_size;
    GCGE_INT   V_tmp_size = max_dim_x - nev + block_size;
    GCGE_INT   CG_p_size = block_size;
    workspace->V_size = V_size;
    workspace->V_tmp_size = V_tmp_size;
    workspace->CG_p_size = CG_p_size;

    //V,V_tmp,RitzVec是向量工作空间
    ops->MultiVecCreateByMat(&(workspace->V), V_size, A, ops);
    ops->MultiVecCreateByMat(&(workspace->V_tmp), V_tmp_size, A, ops);
    ops->MultiVecCreateByMat(&(workspace->CG_p), CG_p_size, A, ops);

    //对GCGE算法中用到的一些参数进行初始化
    workspace->max_dim_x = max_dim_x;
    workspace->dim_x     = nev;
    workspace->dim_xp    = 0;
    workspace->dim_xpw   = nev;
    workspace->conv_bs   = 0;
    workspace->num_soft_locked_in_last_iter = 0; 
    workspace->unconv_bs = (nev < para->block_size) ? nev : para->block_size;

    GCGE_INT max_dim_xpw = max_dim_x + 2 * block_size;
    //近似特征值
    workspace->eval      = (GCGE_DOUBLE*)calloc(max_dim_xpw, sizeof(GCGE_DOUBLE));
    //小规模的临时工作空间
    //GCGE_INT lwork1 = 26*max_dim_xpw;
    //GCGE_INT lwork2 = 1+6*max_dim_xpw+2*max_dim_xpw*max_dim_xpw;
    workspace->subspace_dtmp = (GCGE_DOUBLE*)calloc(max_dim_xpw*max_dim_xpw+40*max_dim_xpw, sizeof(GCGE_DOUBLE));
    //+(lwork1>lwork2)?lwork1:lwork2, sizeof(GCGE_DOUBLE));
    workspace->subspace_itmp = (GCGE_INT*)calloc(100*max_dim_xpw, sizeof(GCGE_INT));
    //存储子空间矩阵的特征向量
    workspace->subspace_evec = (GCGE_DOUBLE*)calloc(max_dim_xpw*max_dim_xpw, sizeof(GCGE_DOUBLE));
    //用于存储子空间矩阵
    workspace->subspace_matrix = (GCGE_DOUBLE*)calloc(max_dim_xpw*max_dim_xpw, sizeof(GCGE_DOUBLE));

    //存储前nev个特征对中未收敛的特征对编号
    workspace->unlock = (GCGE_INT*)calloc(max_dim_x, sizeof(GCGE_INT));
    //for(i=0; i<nev; i++)
    GCGE_INT num_locked = nev - para->num_unlock;
    for(i=0; i<para->num_unlock; i++)
    {
        workspace->unlock[i] = i+num_locked;
    }
    //正交化时用到的临时GCGE_INT*型空间,用于临时存储非0的编号
    //workspace->orth_ind = (GCGE_INT*)calloc(max_dim_xpw, sizeof(GCGE_INT));

    //if(ops->DenseMatCreate)
    //{
    //   ops->DenseMatCreate(&(workspace->dense_matrix), max_dim_xpw, max_dim_xpw);
    //}

    //GCGE_STATISTIC_Para用于统计各部分时间
}
//释放GCGE_Para结构中分配的空间
void GCGE_WORKSPACE_Free(GCGE_WORKSPACE **workspace, GCGE_PARA *para, GCGE_OPS *ops)
{
    if((*workspace)->unlock)
    {
        free((*workspace)->unlock);
        (*workspace)->unlock = NULL;
    }
    if((*workspace)->subspace_matrix)
    {
        free((*workspace)->subspace_matrix);
        (*workspace)->subspace_matrix = NULL;
    }
    if((*workspace)->subspace_evec)
    {
        free((*workspace)->subspace_evec);
        (*workspace)->subspace_evec = NULL;
    }
    if((*workspace)->eval)
    {
        free((*workspace)->eval);
        (*workspace)->eval = NULL;
    }
    if((*workspace)->subspace_dtmp)
    {
        free((*workspace)->subspace_dtmp);
        (*workspace)->subspace_dtmp = NULL;
    }
    if((*workspace)->subspace_itmp)
    {
        free((*workspace)->subspace_itmp);
        (*workspace)->subspace_itmp = NULL;
    }
    //if((*workspace)->orth_ind)
    //{
    //    free((*workspace)->orth_ind);
    //    (*workspace)->orth_ind = NULL;
    //}
    
    GCGE_INT max_dim_x = (*workspace)->max_dim_x;
    GCGE_INT max_dim_xpw = max_dim_x + 2 * para->block_size;
    if((*workspace)->V)
    {
        ops->MultiVecDestroy(&((*workspace)->V), (*workspace)->V_size, ops);
        ops->MultiVecDestroy(&((*workspace)->V_tmp), (*workspace)->V_tmp_size, ops);
        //ops->MultiVecDestroy(&((*workspace)->RitzVec), max_dim_x, ops);
        ops->MultiVecDestroy(&((*workspace)->CG_p), (*workspace)->CG_p_size,  ops);
        //ops->MultiVecDestroy(&((*workspace)->CG_r), para->block_size, ops);
        //if(ops->DenseMatCreate)
        //{
        //    ops->DenseMatDestroy(&((*workspace)->dense_matrix));
        //}
    }

    free((*workspace)); (*workspace) = NULL;
}
