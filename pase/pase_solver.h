#ifndef __PASE_MG_SOLVER_H__
#define __PASE_MG_SOLVER_H__

#include <math.h>
#include "pase_mg.h"
#include "pase_param.h"
#include "pase.h"
#include "gcge.h"
#include "gcge_app_pase.h"

typedef struct PASE_MG_SOLVER_PRIVATE_ {

  //-------------------------------------------------------
  //num_levels表示一共用多少层
  PASE_INT      num_levels;
  //计算初值或者用户给定初值的层号
  PASE_INT      initial_level;
  //用户给的初值个数
  PASE_INT      num_given_eigs;
  //最粗网格层指标
  PASE_INT      coarest_level;  
  //最细网格层指标
  PASE_INT      finest_level;  
  //max_cycle_count_each_level表示每层最多做多少次二网格迭代
  PASE_INT     *max_cycle_count_each_level;
  //每层上最多前光滑的次数
  PASE_INT     *max_pre_count_each_level;
  //每层上最多后光滑的次数
  PASE_INT     *max_post_count_each_level;
  //每层上最多直接求解时GCGE迭代的次数
  PASE_INT     *max_direct_count_each_level;
  //第一层直接求解时的最大迭代次数
  PASE_INT     max_initial_count;

  //最多算多少个特征对
  PASE_INT     max_nev;
  //表示在实际计算的时候求解的特征对个数（在本算法中一般设置为 nev+5 和2*nev的较小值）
  PASE_INT     pase_nev; 
  //实际要求的特征对个数
  PASE_INT     nev;
  //相对残差收敛准则，||A*x-\lambda*B*x||_2/(\lambda*||x||_2) < rtol 则为收敛
  PASE_REAL    rtol;
  //绝对残差收敛准则，||A*x-\lambda*B*x||_2/||x||_2 < atol 则为收敛
  PASE_REAL    atol;
  //以上两种收敛准则满足其一即可认定为收敛
  //res_norm用于存储特征对的绝对残差
  PASE_REAL   *abs_res_norm;
  //已经收敛的特征对个数
  PASE_INT     nconv;
  //已经收敛的特征对个数, 用于smooth时的lock
  PASE_INT     nlock_smooth;
  //已经收敛的特征对个数, 用于aux_direct时的lock
  PASE_INT     nlock_direct;

  //特征值空间
  PASE_SCALAR      *eigenvalues;
  //解空间，多重网格迭代时，用于存储每一层的解
  void           ***sol;
  //sol_size用于存储每层sol中有几个向量
  PASE_INT         *sol_size;
  //每一层的右端项空间, 分配空间为bmg_step_size
  //也在复合矩阵中用作aux_A->aux_Hh 
  //因此可能会用作VH的层上，需要给rhs分配的空间为max_nev
  //细层的rhs还会用作P*u_H, 用于prolong 
  void           ***rhs;
  //rhs_size用于存储每层rhs中有几个向量
  //因为VH层上需要多分配
  PASE_INT         *rhs_size;
  //每一层BCG迭代时的P向量空间, 分配空间为bmg_step_size
  //也在复合矩阵中用作aux_B->aux_Hh 
  //因此可能会用作VH的层上，需要给cg_p分配的空间为max_nev
  void           ***cg_p;
  //cg_p_size用于存储每层cg_p中有几个向量
  //因为VH层上需要多分配
  PASE_INT         *cg_p_size;
  //每一层BCG迭代时的W向量空间
  void           ***cg_w;
  //cg_w_size用于存储每层cg_w中有几个向量
  PASE_INT         *cg_w_size;
  //每一层BCG迭代时的残差向量空间
  void           ***cg_res;
  //cg_res_size用于存储每层cg_res中有几个向量
  PASE_INT         *cg_res_size;
  //每一层BCG迭代时的double型临时空间
  PASE_REAL        *cg_double_tmp;
  //每一层BCG迭代时的int型临时空间
  PASE_INT         *cg_int_tmp;
  //普通矩阵向量操作
  GCGE_OPS         *gcge_ops;
  //pase aux 复合矩阵的矩阵向量操作
  PASE_OPS         *pase_ops;
  //pase aux 复合向量
  PASE_MultiVector  aux_u;
  //pase aux 复合矩阵
  PASE_Matrix       aux_A;
  //pase aux 复合矩阵
  PASE_Matrix       aux_B;
  //pase aux 复合矩阵特征值
  PASE_REAL        *aux_eigenvalues;
  //多重网格结构
  PASE_MULTIGRID    multigrid;
  //BMG进行分批计算时，每次最多计算的线性方程组个数
  PASE_INT          bmg_step_size;

  //打印level
  PASE_INT     print_level; 
  //统计时间
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

PASE_INT
PASE_EigenSolver(void *A, void *B, PASE_SCALAR *eval, void **evec, 
        PASE_INT nev, PASE_PARAMETER param, GCGE_OPS *gcge_ops);

PASE_MG_SOLVER
PASE_Mg_solver_create(PASE_PARAMETER param);

PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver, void *A, void *B, GCGE_OPS *gcge_ops);

PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver);

PASE_INT
PASE_Mg_solve(PASE_MG_SOLVER solver);

PASE_INT 
PASE_Mg_cycle(PASE_MG_SOLVER solver, PASE_INT coarse_level, PASE_INT fine_level);

PASE_INT
PASE_Mg_set_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT coarse_level, 
      PASE_INT fine_level);

PASE_INT
PASE_Direct_solve(PASE_MG_SOLVER solver, PASE_INT idx_level);

PASE_INT 
PASE_Matrix_create_sub(PASE_Matrix *aux_A, PASE_INT n);

PASE_INT 
PASE_Matrix_destroy_sub(PASE_Matrix *aux_A);

PASE_INT
PASE_Mg_pase_aux_matrix_create(PASE_MG_SOLVER solver, PASE_INT idx_level);

PASE_INT 
PASE_Aux_matrix_set_by_pase_matrix(PASE_Matrix aux_A, void *A_h, void **sol, 
        PASE_MG_SOLVER solver, PASE_INT coarse_level, PASE_INT fine_level);

PASE_INT
PASE_Mg_set_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT coarse_level, 
      PASE_INT current_level);

PASE_INT 
PASE_Mg_pase_aux_vector_create(PASE_MG_SOLVER solver, PASE_INT idx_level);

PASE_INT 
PASE_MultiVector_create_sub(PASE_MultiVector *aux_u, PASE_INT n);

PASE_INT 
PASE_MultiVector_destroy_sub(PASE_MultiVector *aux_u);

PASE_INT 
PASE_Aux_direct_solve(PASE_MG_SOLVER solver, PASE_INT coarse_level);

PASE_INT
PASE_Mg_prolong_from_pase_aux_vector(PASE_MG_SOLVER solver,
      PASE_INT coarse_level, PASE_INT fine_level);

PASE_INT
PASE_Mg_smoothing(PASE_MG_SOLVER solver, PASE_INT fine_level, PASE_INT max_iter);

#endif
