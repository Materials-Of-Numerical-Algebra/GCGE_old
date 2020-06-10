/*
 * =====================================================================================
 *
 *       Filename:  gcge_cg.c
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
#include <math.h>
#include "gcge_ops_orth.h"


/* 严格的单个向量重正交化的方式(数值稳定性比较好) */
void GCGE_Orth_GramSchmidt(void **V, GCGE_INT start, GCGE_INT *end, void *B, GCGE_OPS *ops)
{
   GCGE_GramSchidtOrth *gs_orth = (GCGE_GramSchidtOrth*)ops->orth_workspace;

   GCGE_INT    current, j; // 去掉V[current]中V[j]的分量
   GCGE_INT    reorth_count; //统计(重)正交化次数
   GCGE_INT    mv_s[2], mv_e[2];

   GCGE_DOUBLE norm_in, norm_out, ipv, ratio;
   void        *vec_current, *vec_j, *vec_tmp;

   for(current = start; current < (*end); ++current)
   {
      /* 提取V[current]并计算
       * norm_in = ||V[current]||_B */
      ops->GetVecFromMultiVec(V, current, &vec_current, ops); //vec_current = V[current]
      if(B == NULL)
      {
	 ops->VecInnerProd(vec_current, vec_current, &norm_in, ops);      
      }
      else
      {
	 ops->GetVecFromMultiVec(gs_orth->V_tmp, gs_orth->idx, &vec_tmp, ops);
	 ops->MatDotVec(B, vec_current, vec_tmp, ops); 
	 ops->VecInnerProd(vec_current, vec_tmp, &norm_in, ops);
	 ops->RestoreVecForMultiVec(gs_orth->V_tmp, gs_orth->idx, &vec_tmp, ops);      
      }
      norm_in = sqrt(norm_in);

      /* 重复将V[current]中剔除V[0, current-1] 的部分 */
      reorth_count = 0;
      do {
	 /* 当前向量V[current]需要和V[0, current-1]进行B正交 */
	 for (j=0; j<current; j++)
	 {
	    /* 去掉V[current]中V[j]的分量 V[current] = V[current] - ipv*V[j] */
	    ops->GetVecFromMultiVec(V, j, &vec_j, ops);
	    if(B == NULL)
	    {
	       ops->VecInnerProd(vec_current, vec_j, &ipv, ops);
	    }
	    else
	    {
	       ops->GetVecFromMultiVec(gs_orth->V_tmp, gs_orth->idx, &vec_tmp, ops);
	       ops->MatDotVec(B, vec_current, vec_tmp, ops); 
	       ops->VecInnerProd(vec_j, vec_tmp, &ipv, ops);        
	       ops->RestoreVecForMultiVec(gs_orth->V_tmp, gs_orth->idx, &vec_tmp, ops);        
	    }
	    ops->VecAxpby(-ipv, vec_j, 1.0, vec_current, ops);
	    ops->RestoreVecForMultiVec(V, j, &vec_j, ops);          
	 }
	 /* 计算新的V[current]的B模 */
	 if(B == NULL)
	 {
	    ops->VecInnerProd(vec_current, vec_current, &norm_out, ops);
	 }
	 else
	 {
	    ops->GetVecFromMultiVec(gs_orth->V_tmp, gs_orth->idx, &vec_tmp, ops);
	    ops->MatDotVec(B, vec_current, vec_tmp, ops); 
	    ops->VecInnerProd(vec_current, vec_tmp, &norm_out, ops);
	    ops->RestoreVecForMultiVec(gs_orth->V_tmp, gs_orth->idx, &vec_tmp, ops);          
	 }
	 norm_out = sqrt(norm_out);

	 ratio   = norm_out/norm_in;
	 norm_in = norm_out;
	 ++reorth_count;      
      } while((ratio < gs_orth->reorth_tol) && (reorth_count < gs_orth->max_reorth_count) 
	    && (norm_out > gs_orth->orth_zero_tol) );
//      printf ( "reorth_count = %d, ratio = %.5e\n", reorth_count, ratio );
      /* 上面的判断中，如果ratio比较大，比如接近1，
       * 也就是说，已经去掉V[current]中V[0]到V[current-1]的部分 
       * 如果norm_out接近与0, 则说明V[current]与V[0]到V[current-1]线性相关 */
      ops->RestoreVecForMultiVec(V, current, &vec_current, ops);

      /* 对V[current]做归一化，或者去掉这个向量(将之替换到最后) */
      if(norm_out > gs_orth->orth_zero_tol)
      {
	 ops->GetVecFromMultiVec(V, current, &vec_current, ops);
	 ops->VecAxpby(0.0, vec_current, 1/norm_out, vec_current, ops);     
	 ops->RestoreVecForMultiVec(V, current, &vec_current, ops);
      }
      else 
      {
	 if(current < *end-1)
	 {
	    mv_s[0] = current;
	    mv_e[0] = current+1;
	    mv_s[1] = *end-1;
	    mv_e[1] = *end;
	    ops->MultiVecSwap(V, V, mv_s, mv_e, ops);
	 }
	 /* 总向量的个数减1, 当前向量的编号减1, 因为当前向量已被替换 */
	 (*end)--;
	 current--;
      }
   }
}

/* 只需要一个临时向量，V_tmp[idx] */
void GCGE_OrthSetup_GramSchmidt(
      GCGE_INT    max_reorth_count,    
      GCGE_DOUBLE reorth_tol,       GCGE_DOUBLE orth_zero_tol, 
      void        **V_tmp,          GCGE_INT idx,
      GCGE_OPS *ops)
{
   static GCGE_GramSchidtOrth gs_orth_static = {
      .max_reorth_count = 3, .reorth_tol = 0.75, .orth_zero_tol = 1e-14, 
      .V_tmp = NULL, .idx = -1};
   ops->orth_workspace          = (void *)&gs_orth_static;

   GCGE_GramSchidtOrth *gs_orth = (GCGE_GramSchidtOrth*)ops->orth_workspace;
   gs_orth->max_reorth_count    = max_reorth_count;
   gs_orth->reorth_tol          = reorth_tol;
   gs_orth->orth_zero_tol       = orth_zero_tol;
   gs_orth->V_tmp               = V_tmp;
   gs_orth->idx                 = idx;
}
