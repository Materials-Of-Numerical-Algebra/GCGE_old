#ifndef __PASE_MG_SOLVER_H__
#define __PASE_MG_SOLVER_H__

#include "pase_mg.h"
#include "pase_param.h"
#include "pase.h"
#include "gcge.h"

typedef struct PASE_MG_FUNCTION_PRIVATE_ { 
  PASE_INT    (*get_initial_vector) (void *solver);
  PASE_INT    (*direct_solve)       (void *solver); 
  PASE_INT    (*presmoothing)       (void *solver); 
  PASE_INT    (*postsmoothing)      (void *solver); 
  PASE_INT    (*presmoothing_aux)   (void *solver); 
  PASE_INT    (*postsmoothing_aux)  (void *solver); 
} PASE_MG_FUNCTION_PRIVATE;
typedef PASE_MG_FUNCTION_PRIVATE * PASE_MG_FUNCTION;

typedef struct PASE_MG_SOLVER_PRIVATE_ {
  PASE_INT     cycle_type;       /* 0. two-grid (default): solve eigenvalue problem on coarsest grid and solve linear problem on finest grid */
                                  /* 1. multigrid: solve eigenvalue problem on coarsest grid and solve linear problem on finer grid */
  PASE_INT    *idx_cycle_level;
  PASE_INT     num_cycle_level;
  PASE_INT     max_cycle_level;
  PASE_INT     cur_cycle_level;
  PASE_INT     nleve;

  PASE_INT     block_size;
  PASE_INT     max_block_size;
  PASE_INT     actual_block_size;

  PASE_INT     max_pre_iter;
  PASE_INT     max_post_iter;
  PASE_INT     max_direct_iter;

  PASE_REAL    rtol;
  PASE_REAL    atol;
  PASE_REAL   *r_norm;
  PASE_INT     nconv;
  PASE_INT     nlock;
  PASE_INT     max_cycle;
  PASE_INT     ncycl;
  PASE_INT     print_level; 

  PASE_REAL    set_up_time;
  PASE_REAL    get_initvec_time;
  PASE_REAL    smooth_time;
  PASE_REAL    set_aux_time;
  PASE_REAL    prolong_time;
  PASE_REAL    direct_solve_time;
  PASE_REAL    total_solve_time;
  PASE_REAL    total_time;

  PASE_REAL    time_inner;
  PASE_REAL    time_lapack;
  PASE_REAL    time_other;
  PASE_REAL    time_orth_gcg;
  PASE_REAL    time_diag_pre;
  PASE_REAL    time_linear_diag;

  PASE_SCALAR *eigenvalues;
  PASE_SCALAR *exact_eigenvalues;

  void             **u;
  void             **u_tmp;
  GCGE_OPS         *gcge_ops;
  PASE_OPS         *pase_ops;
  PASE_MultiVector *aux_u;
  PASE_MULTIGRID   multigrid;
  //PASE_VECTOR *u;
  PASE_INT     is_u_owner; //不知道还有没有用
  //PASE_AUX_VECTOR    	**aux_u;

  //PASE_MULTIGRID    multigrid;
  PASE_MG_FUNCTION  function;

  char *method_init;
  char *method_pre;
  char *method_post;
  char *method_pre_aux;
  char *method_post_aux;
  char *method_dire;

  void *amg_data_coarsest;
  PASE_MULTIGRID multigrid_pre;
} PASE_MG_SOLVER_PRIVATE; 
typedef PASE_MG_SOLVER_PRIVATE * PASE_MG_SOLVER;

//求解
PASE_INT
PASE_EigenSolver(void *A, void *B, PASE_SCALAR *eval, void **evec, 
        PASE_INT block_size, PASE_PARAMETER param, GCGE_OPS *gcge_ops)
