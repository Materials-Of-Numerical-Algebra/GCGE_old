/*
 * =====================================================================================
 *
 *       Filename:  gcge_cg.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时50分16秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef _GCGE_CG_H_
#define _GCGE_CG_H_

#include <stddef.h>
#include "gcge_config.h"

#include "gcge_para.h"
#include "gcge_ops.h"

#include "gcge_workspace.h"

#if GCGE_USE_MPI
#include <mpi.h>
#endif

/**
 * @brief 
 *
 * @param V
 * @param start
 * @param[in|out] end
 * @param B
 * @param ops
 * @param orth_para
 * @param workspace
 */
void GCGE_CG(void *Matrix, void *b, void *x, GCGE_OPS *ops, GCGE_PARA *para, 
             void **V_tmp1, void **V_tmp2);
//统一进行块形式的CG迭代
//CG迭代求解 A * x =  RHS 
//W存储在 V(:,w_start:w_start+w_length), RHS存储在V_tmp(:,0:w_length)
// V_tmp = [r, p, w]的方式来组织
//以下的BCG允许不连续收敛, 不做矩阵统一乘以向量组的优化
void GCGE_BCG(void *Matrix, GCGE_INT if_shift, GCGE_DOUBLE shift, void *B, 
      void **RHS, void**V, GCGE_INT *start, 
      GCGE_INT *end, GCGE_INT max_it, GCGE_DOUBLE rate,
      GCGE_OPS *ops, void *V_tmp1, void **V_tmp2, void **V_tmp3,
      GCGE_DOUBLE *subspace_dtmp, GCGE_INT *subspace_itmp);
//void GCGE_BCG(void *Matrix, void **RHS, void**V, GCGE_INT *start, 
//      GCGE_INT *end, GCGE_INT max_it, GCGE_DOUBLE rate,
//      GCGE_OPS *ops, void *V_tmp1, void **V_tmp2, void **V_tmp3,
//      GCGE_DOUBLE *subspace_dtmp, GCGE_INT *subspace_itmp);
//以下的GCGE_BCG_Continuous只允许连续收敛, 会做矩阵统一乘以向量组的优化
void GCGE_BCG_Continuous(void *Matrix, void **RHS, void**V, GCGE_INT *start, 
      GCGE_INT *end, GCGE_INT max_it, GCGE_DOUBLE rate,
      GCGE_OPS *ops, void *V_tmp1, void **V_tmp2, void **V_tmp3,
      GCGE_DOUBLE *subspace_dtmp, GCGE_INT *subspace_itmp);
#endif
