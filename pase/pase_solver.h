#ifndef __PASE_MG_SOLVER_H__
#define __PASE_MG_SOLVER_H__

#include "pase_multigrid.h" 
#include "pase_config.h"

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
  GCGE_OPS         *gcge_ops;
  PASE_OPS         *pase_ops;
  PASE_MultiVector *aux_u;
  PASE_MultiGrid   multigrid;
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

PASE_MG_SOLVER PASE_Mg_solver_create(void *A, void *B, PASE_PARAMETER param);
PASE_MG_FUNCTION PASE_Mg_function_create(
    PASE_INT (*get_initial_vector) (void *solver),
    PASE_INT (*direct_solve)       (void *solver),
    PASE_INT (*presmoothing)       (void *solver), 
    PASE_INT (*postsmoothing)      (void *solver), 
    PASE_INT (*presmoothing_aux)   (void *solver), 
    PASE_INT (*postsmoothing_aux)  (void *solver));
PASE_INT PASE_Mg_solver_destroy(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_up(PASE_MG_SOLVER solver, void *A, void *B, void **x, PASE_PARAMETER param);
PASE_INT PASE_Mg_solve(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_cycle(PASE_MG_SOLVER solver);

PASE_INT PASE_Mg_set_aux_space(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT cur_level, PASE_INT last_level);
PASE_INT PASE_Mg_set_pase_aux_matrix_by_pase_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, void **u_j);
PASE_INT PASE_Mg_set_pase_aux_matrix_by_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, PASE_AUX_VECTOR *aux_u_j);
PASE_INT PASE_Mg_pase_aux_matrix_create(PASE_MG_SOLVER solver, PASE_INT i);
PASE_INT PASE_Mg_set_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT cur_level);

PASE_INT PASE_Mg_prolong_from_pase_aux_vector(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_AUX_VECTOR *aux_u_i, PASE_INT j, void **u_j);
PASE_INT PASE_Mg_prolong_from_pase_aux_vector_to_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_AUX_VECTOR *aux_u_i, PASE_INT j, PASE_AUX_VECTOR *aux_u_j);
PASE_INT PASE_Mg_prolong(PASE_MG_SOLVER solver,  PASE_INT i, void *u_i, PASE_INT j, void *u_j);
PASE_INT PASE_Mg_restrict(PASE_MG_SOLVER solver, PASE_INT i, void *u_i, PASE_INT j, void *u_j);

PASE_INT PASE_Mg_error_estimate(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_get_initial_vector(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_presmoothing(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_postsmoothing(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_direct_solve(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_orthogonalize(PASE_MG_SOLVER solver);

PASE_INT PASE_Mg_set_exact_eigenvalues(PASE_MG_SOLVER solver, PASE_SCALAR *exact_eigenvalues);
PASE_INT PASE_Mg_solver_set_multigrid(PASE_MG_SOLVER solver, PASE_MULTIGRID multigrid);
PASE_INT PASE_Mg_set_cycle_type(PASE_MG_SOLVER solver, PASE_INT cycle_type);
PASE_INT PASE_Mg_set_block_size(PASE_MG_SOLVER solver, PASE_INT block_size);
PASE_INT PASE_Mg_set_max_block_size(PASE_MG_SOLVER solver, PASE_INT max_block_size);
PASE_INT PASE_Mg_set_max_cycle(PASE_MG_SOLVER solver, PASE_INT max_iter);
PASE_INT PASE_Mg_set_max_pre_iteration(PASE_MG_SOLVER solver, PASE_INT pre_iter);
PASE_INT PASE_Mg_set_max_post_iteration(PASE_MG_SOLVER solver, PASE_INT post_iter);
PASE_INT PASE_Mg_set_max_direct_iteration(PASE_MG_SOLVER solver, PASE_INT max_direct_iter);
PASE_INT PASE_Mg_set_atol(PASE_MG_SOLVER solver, PASE_REAL atol);
PASE_INT PASE_Mg_set_rtol(PASE_MG_SOLVER solver, PASE_REAL rtol);
PASE_INT PASE_Mg_set_print_level(PASE_MG_SOLVER solver, PASE_INT print_level);

PASE_INT PASE_Mg_smoothing_by_cg(void *mg_solver, char *PreOrPost);
PASE_INT PASE_Mg_smoothing_by_cg_aux(void *mg_solver, char *PreOrPost);
PASE_INT PASE_Mg_presmoothing_by_cg(void *mg_solver);
PASE_INT PASE_Mg_postsmoothing_by_cg(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_cg_aux(void *mg_solver);
PASE_INT PASE_Mg_postsmoothing_by_cg_aux(void *mg_solver);
//-------------------------------------------
//cg这里可以改用GCGE_BCG
//PASE_INT PASE_Linear_solve_by_cg(void *A, void *b, void *x, PASE_REAL tol, PASE_INT max_iter);
//PASE_INT PASE_Linear_solve_by_cg_aux(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_b, PASE_AUX_VECTOR aux_x, PASE_REAL tol, PASE_INT max_iter);
//-------------------------------------------

PASE_INT PASE_Mg_direct_solve_by_gcg(void *mg_solver);
PASE_INT PASE_Mg_precondition_for_gcg(void *mg_solver);

/**
 * @brief 
 *
 * @param A
 * @param B
 * @param eval
 * @param evec
 * @param block_size
 * @param param
 */
PASE_INT PASE_EigenSolver(void *A, void *B, PASE_SCALAR *eval, void **evec, PASE_INT block_size, PASE_PARAMETER param);

PASE_INT PASE_Mg_print(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_print_eigenvalue_of_current_level(PASE_MG_SOLVER solver);

#endif
