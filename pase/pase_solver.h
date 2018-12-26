#ifndef __PASE_MG_SOLVER_H__
#define __PASE_MG_SOLVER_H__

#include <math.h>
#include "pase_mg.h"
#include "pase_param.h"
#include "pase.h"
#include "gcge.h"
#include "gcge_app_pase.h"

//PASE用到的函数，因为获取初值用多重网格，迭代中用二网格，所以后两个函数
//presmoothing_aux与postsmoothing_aux没有用到
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
  /* cycle_type: 
   * 0. two-grid (default): solve eigenvalue problem on coarsest grid 
   *    and solve linear problem on finest grid 
   * 1. multigrid: solve eigenvalue problem on coarsest grid 
   *    and solve linear problem on finer grid */
  PASE_INT     cycle_type;       

  PASE_INT    *idx_cycle_level; 
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

  PASE_SCALAR *eigenvalues;
  PASE_SCALAR *exact_eigenvalues;

  //新修改部分,工作空间
  PASE_SCALAR      *given_eval;
  void            **given_evec;
  void            **u;
  void           ***u_tmp;
  void           ***u_tmp_1;
  void           ***u_tmp_2;
  void           ***u_tmp_3;
  PASE_REAL        *double_tmp;
  PASE_INT         *int_tmp;
  GCGE_OPS         *gcge_ops;
  PASE_OPS         *pase_ops;
  PASE_MultiVector  aux_u;

  PASE_MULTIGRID    multigrid;
  PASE_MG_FUNCTION  function;
  PASE_INT          is_u_owner;

  char  *method_init;
  char  *method_pre;
  char  *method_post;
  char  *method_pre_aux;
  char  *method_post_aux;
  char  *method_dire;

  PASE_REAL  set_up_time;
  PASE_REAL  get_initvec_time;
  PASE_REAL  smooth_time;
  PASE_REAL  set_aux_time;
  PASE_REAL  prolong_time;
  PASE_REAL  direct_solve_time;
  PASE_REAL  total_solve_time;
  PASE_REAL  total_time;

} PASE_MG_SOLVER_PRIVATE; 
typedef PASE_MG_SOLVER_PRIVATE * PASE_MG_SOLVER;

//求解
//A,B: 要求解的矩阵
//eval,evec: 要求解的特征对
//block_size: 要求解的特征值个数
//param: 用户(main)中提供的参数
//gcge_ops: 通过gcge_ops,用户(main)中决定使用哪种矩阵向量格式
//pase_ops: pase_ops在这里给是因为测试时PrintMat函数在main中给
//Ac,Bc,P,R: 测试时使用的粗网格矩阵与延拓限制矩阵,main中给出
PASE_INT
PASE_EigenSolver(void *A, void *B, PASE_SCALAR *eval, void **evec, 
        PASE_INT block_size, PASE_PARAMETER param, GCGE_OPS *gcge_ops,
        PASE_OPS *pase_ops,
        void *Ac, void *Bc, void *P, void *R);

//创建solver,给solver中内容赋初值或者NULL,不进行空间分配
PASE_MG_SOLVER
PASE_Mg_solver_create(PASE_PARAMETER param);

//确定前后光滑等使用哪个函数，在PASE_Mg_solver_create中给出
PASE_MG_FUNCTION
PASE_Mg_function_create(PASE_INT (*get_initial_vector) (void *solver),
    PASE_INT (*direct_solve)       (void *solver),
    PASE_INT (*presmoothing)       (void *solver), 
    PASE_INT (*postsmoothing)      (void *solver), 
    PASE_INT (*presmoothing_aux)   (void *solver), 
    PASE_INT (*postsmoothing_aux)  (void *solver)) ;

//创建multigrid,并给各个工作空间分配空间
PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver, void *A, void *B, PASE_SCALAR *eval, void **x, PASE_PARAMETER param,
        void *Ac, void *Bc, void *P, void *R);

//销毁工作空间
PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver);

PASE_INT 
PASE_Mg_solve(PASE_MG_SOLVER solver);

//对最细层解u进行误差估计，计算Rayleigh商，再计算||Ax-\lambdaBx||_2/||x||_2
//洪的pase中计算的是||Ax-\lambdaBx||_2/||x||_B. ???????
PASE_INT 
PASE_Mg_error_estimate(PASE_MG_SOLVER solver);

//一次二网格迭代
PASE_INT 
PASE_Mg_cycle(PASE_MG_SOLVER solver);

PASE_INT 
PASE_Mg_presmoothing(PASE_MG_SOLVER solver);

PASE_INT 
PASE_Mg_postsmoothing(PASE_MG_SOLVER solver);

PASE_INT
PASE_Mg_print_eigenvalue_of_current_level(PASE_MG_SOLVER solver);

//用GCGE求解复合矩阵的特征对
PASE_INT
PASE_Mg_direct_solve(PASE_MG_SOLVER solver);

PASE_INT
PASE_Mg_set_aux_space(PASE_MG_SOLVER solver);

PASE_INT 
PASE_Mg_set_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT cur_level, PASE_INT last_level);

PASE_INT
PASE_Mg_set_pase_aux_matrix_by_pase_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, void **u_j);

PASE_INT
PASE_Mg_pase_aux_matrix_create(PASE_MG_SOLVER solver, PASE_INT i);

PASE_INT 
PASE_Aux_matrix_set_by_pase_matrix(PASE_Matrix aux_A, void *A_h, void **u_j, 
        PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j);

PASE_INT
PASE_Mg_set_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT cur_level);

PASE_INT
PASE_Mg_prolong_from_pase_aux_vector(PASE_MG_SOLVER solver);

PASE_INT
PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_MultiVector aux_u_i, PASE_INT j, void **u_j);

PASE_INT
PASE_Mg_presmoothing_by_amg_hypre(void *mg_solver);

PASE_INT
PASE_Mg_postsmoothing_by_amg_hypre(void *mg_solver);

PASE_INT 
PASE_Mg_smoothing_by_amg_hypre(void *mg_solver, char *PreOrPost);

PASE_INT
PASE_Mg_direct_solve_by_gcg(void *mg_solver);

PASE_INT
PASE_Mg_solver_set_multigrid(PASE_MG_SOLVER solver, PASE_MULTIGRID multigrid);

PASE_INT
PASE_Mg_set_cycle_type(PASE_MG_SOLVER solver, PASE_INT cycle_type);

PASE_INT 
PASE_Mg_set_block_size(PASE_MG_SOLVER solver, PASE_INT block_size);

PASE_INT 
PASE_Mg_set_max_block_size(PASE_MG_SOLVER solver, PASE_INT max_block_size);

PASE_INT 
PASE_Mg_set_max_cycle(PASE_MG_SOLVER solver, PASE_INT max_cycle);

PASE_INT 
PASE_Mg_set_max_pre_iteration(PASE_MG_SOLVER solver, PASE_INT max_pre_iter);

PASE_INT 
PASE_Mg_set_max_post_iteration(PASE_MG_SOLVER solver, PASE_INT max_post_iter);

PASE_INT
PASE_Mg_set_max_direct_iteration(PASE_MG_SOLVER solver, PASE_INT max_direct_iter);

PASE_INT 
PASE_Mg_set_atol(PASE_MG_SOLVER solver, PASE_REAL atol);

PASE_INT 
PASE_Mg_set_rtol(PASE_MG_SOLVER solver, PASE_REAL rtol);

PASE_INT 
PASE_Mg_set_print_level(PASE_MG_SOLVER solver, PASE_INT print_level);

PASE_INT
PASE_Mg_set_exact_eigenvalues(PASE_MG_SOLVER solver, PASE_SCALAR *exact_eigenvalues);

PASE_INT
PASE_Mg_print(PASE_MG_SOLVER solver);

PASE_INT
PASE_Mg_get_initial_vector_by_full_multigrid_hypre(void *mg_solver);

PASE_INT
PASE_Mg_get_initial_vector(PASE_MG_SOLVER solver);
#endif
