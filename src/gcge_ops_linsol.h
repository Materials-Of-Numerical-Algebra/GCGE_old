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
#ifndef _GCGE_OPS_LINSOL_H_
#define _GCGE_OPS_LINSOL_H_

#include "gcge_config.h"
#include "gcge_ops.h"
#include "gcge_utilities.h" 

#if GCGE_USE_MPI
#include <mpi.h>
#endif

typedef struct GCGE_PCGSolver_ {
  GCGE_INT    max_it;     GCGE_DOUBLE rate;     GCGE_DOUBLE tol; 
  void        *r;         void        *p;       void        *w;
  void        *pc;
}GCGE_PCGSolver;

typedef struct GCGE_BPCGSolver_ {
  GCGE_INT    max_it;     GCGE_DOUBLE rate;     GCGE_DOUBLE tol;
  void        **r;        void        **p;      void        **w;
  GCGE_INT    *start_rpw; GCGE_INT    *end_rpw;
  GCGE_DOUBLE *dtmp;      GCGE_INT    *itmp;
  void        *pc;
  GCGE_INT    if_shift;    GCGE_DOUBLE shift;   void        *B; 
}GCGE_BPCGSolver;

typedef struct GCGE_BAMGSolver_ {
      GCGE_INT    *max_it;      GCGE_DOUBLE *rate;        GCGE_DOUBLE *tol; 
      void        **A_array;    void        **P_array; 
      GCGE_INT    num_levels;
      void        ***rhs_array; void        ***sol_array; 
      GCGE_INT    *start_bx;    GCGE_INT    *end_bx;
      void        ***r_array;   void        ***p_array;   void        ***w_array;
      GCGE_INT    *start_rpw;   GCGE_INT    *end_rpw; 
      GCGE_DOUBLE *dtmp;        GCGE_INT    *itmp; 
      void        *pc;
}GCGE_BAMGSolver;


void GCGE_LinearSolver_PCG(void *Matrix, void *b, void *x, GCGE_OPS *ops);
void GCGE_LinearSolverSetup_PCG(
      GCGE_INT max_it, GCGE_DOUBLE rate, GCGE_DOUBLE tol, 
      void *r,         void *p,          void *w, 
      void *pc, 
      GCGE_OPS *ops);
void GCGE_MultiLinearSolver_BPCG(void *Matrix, void **RHS, void **V, 
      GCGE_INT *start, GCGE_INT *end, GCGE_OPS *ops);
void GCGE_MultiLinearSolverSetup_BPCG(
      GCGE_INT    max_it,       GCGE_DOUBLE rate,         GCGE_DOUBLE tol, 
      void        **r,          void        **p,          void        **w, 
      GCGE_INT    *start_rpw,   GCGE_INT    *end_rpw, 
      GCGE_DOUBLE *dtmp,        GCGE_INT    *itmp,
      void *pc, 
      GCGE_INT if_shift,        GCGE_DOUBLE shift,        void *B, 
      GCGE_OPS *ops);

void GCGE_MultiLinearSolver_BAMG(void *Matrix, void **RHS, void **V, 
      GCGE_INT *start, GCGE_INT *end, GCGE_OPS *ops);
void GCGE_MultiLinearSolverSetup_BAMG(
      GCGE_INT    *max_it,      GCGE_DOUBLE *rate,        GCGE_DOUBLE *tol, 
      void        **A_array,    void        **P_array, 
      GCGE_INT    num_levels,
      void        ***rhs_array, void        ***sol_array, 
      GCGE_INT    *start_bx,    GCGE_INT    *end_bx,
      void        ***r_array,   void        ***p_array,   void        ***w_array,
      GCGE_INT    *start_rpw,   GCGE_INT    *end_rpw, 
      GCGE_DOUBLE *dtmp,        GCGE_INT    *itmp, 
      void        *pc, 
      GCGE_OPS    *ops);


#endif
