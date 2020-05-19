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
#ifndef _GCGE_LINSOL_H_
#define _GCGE_LINSOL_H_

#include "gcge_config.h"
#include "gcge_ops.h"
#include "gcge_utilities.h" 

#if GCGE_USE_MPI
#include <mpi.h>
#endif

typedef struct GCGE_PCGSolver_ {
  void        *pc;
  GCGE_INT    max_it; 
  GCGE_DOUBLE rate; 
  GCGE_DOUBLE tol; 
  void        *r; 
  void        *p; 
  void        *w;
}GCGE_PCGSolver;

typedef struct GCGE_BPCGSolver_ {
  void        *pc;
  GCGE_INT    max_it; 
  GCGE_DOUBLE rate; 
  GCGE_DOUBLE tol;
  void        **r; 
  void        **p; 
  void        **w;
  GCGE_INT    *start_rpw; 
  GCGE_INT    *end_rpw;
  GCGE_DOUBLE *dtmp;
  GCGE_INT    *itmp;
  GCGE_INT    if_shift; 
  GCGE_DOUBLE shift; 
  void        *B; 
}GCGE_BPCGSolver;

void GCGE_Default_LinearSolver(void *Matrix, void *b, void *x, GCGE_OPS *ops);
void GCGE_Default_LinearSolverSetUp(void *pc, GCGE_INT max_it, GCGE_DOUBLE tol, 
      void *r, void *p, void *w, GCGE_OPS *ops);
void GCGE_Default_MultiLinearSolver(void *Matrix, void **RHS, void **V, 
      GCGE_INT *start, GCGE_INT *end, GCGE_OPS *ops);
void GCGE_Default_MultiLinearSolverSetUp(void *pc, GCGE_INT max_it, GCGE_DOUBLE tol, 
      void **r, void **p, void **w, GCGE_INT *start_rpw, GCGE_INT *end_rpw, 
      GCGE_DOUBLE *dtmp, GCGE_INT *itmp, 
      GCGE_INT if_shift, GCGE_DOUBLE shift, void *B, 
      GCGE_OPS *ops);

#endif
