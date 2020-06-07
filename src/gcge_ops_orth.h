/*
 * =====================================================================================
 *
 *       Filename:  gcge_linsol.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年05月24日 09时50分16秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef _GCGE_OPS_ORTH_H_
#define _GCGE_OPS_ORTH_H_

#include "gcge_config.h"
#include "gcge_ops.h"
#include "gcge_utilities.h" 

#if GCGE_USE_MPI
#include <mpi.h>
#endif

typedef struct GCGE_GramSchidtOrth_ {
   GCGE_INT    max_reorth_count;    
   GCGE_DOUBLE reorth_tol;       GCGE_DOUBLE orth_zero_tol; 
   /* 临时空间多向量，但只用到一个向量，所以idx告知V_tmp[idx]可用 */
   void        **V_tmp;          GCGE_INT idx;
}GCGE_GramSchidtOrth;

void GCGE_Orth_GramSchmidt(void **V, GCGE_INT start, GCGE_INT *end, void *B, GCGE_OPS *ops);
void GCGE_OrthSetup_GramSchmidt(
      GCGE_INT    max_reorth_count,    
      GCGE_DOUBLE reorth_tol,       GCGE_DOUBLE orth_zero_tol, 
      void        **V_tmp,          GCGE_INT idx,
      GCGE_OPS *ops);
#endif
