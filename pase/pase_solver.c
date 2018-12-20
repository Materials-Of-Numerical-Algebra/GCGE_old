#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "pase_mg_solver_hypre.h"
#include "pase_mg_solver.h"
#include "pase_lobpcg_hypre.h"
#include "pase_gcg.h"

#define CLK_TCK 1000000
#define DIAG_GCG 0

/**
 * @brief 创建 PASE_MG_SOLVER
 *
 * @param A      输入参数
 * @param B      输入参数
 * @param param  输入参数
 *
 * @return PASE_MG_SOLVER
 */
PASE_MG_SOLVER
PASE_Mg_solver_create(void *A, void *B, PASE_PARAMETER param)
{
  PASE_MG_SOLVER solver      = (PASE_MG_SOLVER)PASE_Malloc(sizeof(PASE_MG_SOLVER_PRIVATE));
  PASE_MULTIGRID multigrid   = PASE_Multigrid_create(A, B, param, NULL);

  solver->multigrid          = multigrid;
  solver->function           = PASE_Mg_function_create(//PASE_Mg_get_initial_vector_by_coarse_grid_hypre,
                               //PASE_Mg_get_initial_vector_by_coarse_grid_lobpcg_amg_hypre, 
						       PASE_Mg_get_initial_vector_by_full_multigrid_hypre, 
						       //PASE_Mg_get_initial_vector_by_full_multigrid_hypre_for_guangji, 
                               //PASE_Mg_direct_solve_by_lobpcg_aux_hypre,
						       PASE_Mg_direct_solve_by_gcg, 
                               //PASE_Mg_presmoothing_by_cg,
                               //PASE_Mg_postsmoothing_by_cg,
						       //PASE_Mg_presmoothing_by_pcg_amg_hypre, 
						       //PASE_Mg_smoothing_by_pcg_amg_hypre_for_guangji, 
						       //PASE_Mg_postsmoothing_by_pcg_amg_hypre, 
						       PASE_Mg_presmoothing_by_amg_hypre, 
						       PASE_Mg_postsmoothing_by_amg_hypre, 
                               PASE_Mg_presmoothing_by_cg_aux,
                               PASE_Mg_postsmoothing_by_cg_aux);
  solver->cycle_type                = 0;

  solver->idx_cycle_level           = NULL;
  solver->num_cycle_level           = 0;
  solver->max_cycle_level           = 0;
  solver->cur_cycle_level           = 0;
  solver->nleve                     = param->max_level;

  solver->block_size                = 1;
  solver->max_block_size            = 1;
  solver->actual_block_size         = 1;

  solver->max_pre_iter              = 1;
  solver->max_post_iter             = 1;
  solver->max_direct_iter           = 1;
  solver->rtol                      = 1e-8;
  solver->atol                      = 1e-8;
  solver->r_norm                    = NULL;
  solver->nconv                     = 0;
  solver->nlock                     = 0;
  solver->ncycl                     = 0;
  solver->max_cycle                 = 200;

  solver->print_level               = 1;
  solver->set_up_time               = 0.0;
  solver->get_initvec_time          = 0.0;
  solver->smooth_time               = 0.0;
  solver->set_aux_time              = 0.0;
  solver->prolong_time              = 0.0;
  solver->direct_solve_time         = 0.0;
  solver->total_solve_time          = 0.0;
  solver->total_time                = 0.0;

  solver->time_inner                = 0.0;
  solver->time_lapack               = 0.0;
  solver->time_other                = 0.0;
  solver->time_diag_pre             = 0.0;
  solver->time_linear_diag          = 0.0;
  solver->time_orth_gcg             = 0.0;

  solver->exact_eigenvalues         = NULL;
  solver->eigenvalues               = NULL;
  solver->u                         = NULL;
  solver->is_u_owner                = PASE_YES;
  solver->aux_u                     = NULL;

  solver->method_init               = NULL;
  solver->method_pre                = NULL;
  solver->method_post               = NULL;
  solver->method_pre_aux            = NULL;
  solver->method_post_aux           = NULL;
  solver->method_dire               = NULL;
  
  solver->amg_data_coarsest         = NULL;
  solver->multigrid_pre             = NULL;
  return solver;
}

/**
 * @brief 创建 PASE_MG_FUNCTION
 *
 * @param get_initial_vector
 * @param direct_solve
 * @param presmoothing
 * @param postsmoothing
 * @param presmoothing_aux
 * @param postsmoothing_aux
 *
 * @return PASE_MG_FUNCTION
 */
PASE_MG_FUNCTION
PASE_Mg_function_create(PASE_INT (*get_initial_vector) (void *solver),
    PASE_INT (*direct_solve)       (void *solver),
    PASE_INT (*presmoothing)       (void *solver), 
    PASE_INT (*postsmoothing)      (void *solver), 
    PASE_INT (*presmoothing_aux)   (void *solver), 
    PASE_INT (*postsmoothing_aux)  (void *solver)) 
{
  PASE_MG_FUNCTION function    = (PASE_MG_FUNCTION)PASE_Malloc(sizeof(PASE_MG_FUNCTION_PRIVATE));
  function->get_initial_vector = get_initial_vector;
  function->direct_solve       = direct_solve;
  function->presmoothing       = presmoothing;
  function->postsmoothing      = postsmoothing;
  function->presmoothing_aux   = presmoothing_aux;
  function->postsmoothing_aux  = postsmoothing_aux;
  return function;
}

/**
 * @brief 销毁 PASE_MG_SOLVER 并释放内存空间
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
/* TODO 
 * 要重写 */
PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver)
{
  PASE_INT i, j;
  if(NULL != solver) {
    if(NULL != solver->idx_cycle_level) {
      PASE_Free(solver->idx_cycle_level);
    }
    if(NULL != solver->aux_u) {
      for(i = 0; i < solver->nleve; ++i) {
	if(NULL != solver->aux_u[i]){
	  for(j = 0; j < solver->block_size; ++j) {
	    if(NULL != solver->aux_u[i][j]) {
          //改用solver->pase_ops->MultiVecDestroy
	      PASE_Aux_vector_destroy(solver->aux_u[i][j]);
	    }
	  }
	  PASE_Free(solver->aux_u[i]);
	}
      }
      PASE_Free(solver->aux_u);
    }
    if(NULL != solver->eigenvalues) {
      PASE_Free(solver->eigenvalues);
    }
    if(NULL != solver->r_norm) {
      PASE_Free(solver->r_norm);
    }
    if(NULL != solver->u) {
      //改用solver->pase_ops->MultiVecDestroy
      if(PASE_YES == solver->is_u_owner) {
        for(j = 0; j < solver->block_size; ++j) {
          if(NULL != solver->u[j]) {
            PASE_Vector_destroy(solver->u[j]);
          }
        }
      } else {
        for(j = solver->actual_block_size; j < solver->block_size; ++j) {
          if(NULL != solver->u[j]) {
            PASE_Vector_destroy(solver->u[j]);
          }
        }
      }
      PASE_Free(solver->u);
    }
    if(NULL != solver->function) {
      PASE_Free(solver->function);
    }
    if(NULL != solver->multigrid) {
      PASE_Multigrid_destroy(solver->multigrid);
    }
    if(NULL != solver->amg_data_coarsest) {
      HYPRE_BoomerAMGDestroy((HYPRE_Solver)solver->amg_data_coarsest);
    }
    PASE_Free(solver);
  }
  return 0;
}

/**
 * @brief PASE_MG_SOLVER 的准备阶段
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver, void *A, void *B, void **x, PASE_PARAMETER param)
{
  clock_t start, end;
  start = clock();
  PASE_INT i = 0;
  /* TODO */
  //Multigrid_set_up把不同层矩阵构造出来
  PASE_Multigrid_set_up(solver->multigrid, A, B, param);
  solver->nleve = solver->multigrid->actual_level;
  if(NULL == solver->eigenvalues) {
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->max_block_size*sizeof(PASE_SCALAR));
  }
  if(NULL == solver->u) {
    /* TODO */
    //solver->gcge_ops->MultiVecCreateByMat
    solver->u = (PASE_VECTOR*)PASE_Malloc(solver->max_block_size*sizeof(PASE_VECTOR));
    if(NULL != x) {
      for(i = 0; i < solver->max_block_size; ++i) {
        solver->u[i] = PASE_Vector_create_by_vector(x);
      }
    } else {
      //缺省情况, 依据 A0 创建 u[0], 再依据 u[0] 创建 u[i] (i>0).
      //对于基于 hypre 的向量结构, 特征值求解个数多的时候, 所有向量共用一个partitioning, 节省内存消耗
      solver->u[0] = PASE_Vector_create_by_matrix_and_vector_data_operator(solver->multigrid->A[0], NULL);
      for(i = 1; i < solver->max_block_size; ++i) {
        solver->u[i] = PASE_Vector_create_by_vector(solver->u[0]);
      }
    }
  }

  if(0 == solver->cycle_type) {
    solver->idx_cycle_level = (PASE_INT*)PASE_Malloc(2*sizeof(PASE_INT));
    solver->idx_cycle_level[0] = 0;
    solver->idx_cycle_level[1] = solver->nleve-1;
    solver->cur_cycle_level = 0;
    solver->max_cycle_level = 1;
  } else if(1 == solver->cycle_type) {
    solver->idx_cycle_level = (PASE_INT*)PASE_Malloc(solver->nleve*sizeof(PASE_INT));
    for(i = 0; i < solver->nleve; ++i) {
      solver->idx_cycle_level[i] = i;
    }
    solver->cur_cycle_level = 0;
    solver->max_cycle_level = solver->nleve - 1;
  }
  /* TODO  PASE_MultiVector */
  solver->aux_u       = (PASE_AUX_VECTOR**)PASE_Malloc((solver->nleve)*sizeof(PASE_AUX_VECTOR*));
  for(i = 0; i < solver->nleve; ++i) {
    solver->aux_u[i] = NULL;
  }

  end = clock();
  solver->set_up_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief MG求解, 主要通过迭代 PASE_Mg_cycle 函数实现. 
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_solve(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  start = clock();
  /* TODO  几个函数分别重写 */
  PASE_Mg_get_initial_vector(solver);
  PASE_Mg_error_estimate(solver);
  //PASE_Mg_print(solver);
  do {
    solver->ncycl++;
    PASE_Mg_cycle(solver);
    PASE_Mg_error_estimate(solver);
  } while(solver->max_cycle > solver->ncycl && solver->nconv < solver->actual_block_size);
  end = clock();
  solver->total_solve_time += ((double)(end-start))/CLK_TCK;
  solver->total_time        = solver->total_solve_time + solver->set_up_time;
  PASE_Mg_print(solver);

  return 0;
}

/**
 * @brief MG 方法的主体，采用递归定义，每层上主要包含分成三步：前光滑，粗空间校正，后光滑. 
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_cycle(PASE_MG_SOLVER solver)
{
  PASE_INT cur_cycle_level = solver->cur_cycle_level;
  PASE_INT max_cycle_level = solver->max_cycle_level;

  if(cur_cycle_level < max_cycle_level)
  {
    PASE_Mg_presmoothing(solver);

    solver->cur_cycle_level++;
    PASE_Mg_set_aux_space(solver);
    PASE_Mg_cycle(solver);
    PASE_Mg_prolong_from_pase_aux_vector(solver);
    solver->cur_cycle_level--;

    PASE_Mg_postsmoothing(solver);
  }
  else if( cur_cycle_level == max_cycle_level)
  {
    PASE_Mg_direct_solve(solver);
  }
  return 0;
}

/**
 * @brief 设置辅助矩阵和初始辅助向量
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_aux_space(PASE_MG_SOLVER solver)
{
  PASE_INT cur_level = solver->idx_cycle_level[solver->cur_cycle_level];
  PASE_INT last_level = solver->idx_cycle_level[solver->cur_cycle_level-1];
  clock_t     start, end;
  start = clock();
  PASE_Mg_set_pase_aux_matrix(solver, cur_level, last_level);
  PASE_Mg_set_pase_aux_vector(solver, cur_level);
  end = clock();
  solver->set_aux_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 设置辅助粗空间的矩阵
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT cur_level, PASE_INT last_level)
{
  if(0 == last_level) {
    PASE_Mg_set_pase_aux_matrix_by_pase_matrix(solver, cur_level, last_level, solver->u);
  } else {
    PASE_Mg_set_pase_aux_matrix_by_pase_aux_matrix(solver, cur_level, last_level, solver->aux_u[last_level]);
  }
  return 0;
}

/**
 * @brief 根据细空间矩阵创建辅助粗空间的矩阵
 *
 * @param solver  输入/输出参数
 * @param i       输入参数
 * @param j       输入参数
 * @param u_j     输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_pase_aux_matrix_by_pase_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, PASE_VECTOR *u_j)
{
  PASE_INT block_size = solver->block_size;
  PASE_INT nlock      = solver->nlock;
  PASE_INT idx_block  = 0;
  PASE_MATRIX *A = solver->multigrid->A;
  PASE_MATRIX *B = solver->multigrid->B;
  /* TODO 创建aux矩阵的函数名？ */
  PASE_Mg_pase_aux_matrix_create(solver, i);
  PASE_AUX_MATRIX aux_A = solver->multigrid->aux_A[i];
  PASE_AUX_MATRIX aux_B = solver->multigrid->aux_B[i];
  
  PASE_Aux_matrix_set_block_some(aux_A, nlock, block_size-1, A[j], u_j);
  PASE_Aux_matrix_set_block_some(aux_B, nlock, block_size-1, B[j], u_j);
  PASE_VECTOR Au = PASE_Vector_create_by_vector(u_j[0]);
  for(idx_block = nlock; idx_block < block_size; ++idx_block) {
    PASE_Matrix_multiply_vector(A[j], u_j[idx_block], Au);
    PASE_Mg_restrict(solver, j, Au, i, aux_A->vec[idx_block]);
    PASE_Matrix_multiply_vector(B[j], u_j[idx_block], Au);
    PASE_Mg_restrict(solver, j, Au, i, aux_B->vec[idx_block]);
  }
  PASE_Vector_destroy(Au);
  return 0;
}

/**
 * @brief 根据辅助细空间的矩阵创建辅助粗空间的矩阵
 *
 * @param solver   输入/输出参数
 * @param i        输入参数
 * @param j        输入参数
 * @param aux_u_j  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_pase_aux_matrix_by_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, PASE_AUX_VECTOR *aux_u_j)
{
  PASE_INT         block_size = solver->block_size;
  PASE_INT         nlock      = solver->nlock;
  PASE_INT         idx_block  = 0;
  PASE_Mg_pase_aux_matrix_create(solver, i);
  PASE_AUX_MATRIX *aux_A = solver->multigrid->aux_A;
  PASE_AUX_MATRIX *aux_B = solver->multigrid->aux_B;

  PASE_Aux_matrix_set_block_some_by_aux_matrix(aux_A[i], nlock, block_size-1, aux_A[j], aux_u_j);
  PASE_Aux_matrix_set_block_some_by_aux_matrix(aux_B[i], nlock, block_size-1, aux_B[j], aux_u_j);
  PASE_AUX_VECTOR Au = PASE_Aux_vector_create_by_aux_vector(aux_u_j[0]);
  for(idx_block = nlock; idx_block < block_size; ++idx_block) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A[j], aux_u_j[idx_block], Au);
    PASE_Mg_restrict(solver, j, Au->vec, i, aux_A[i]->vec[idx_block]);
    PASE_Aux_matrix_multiply_aux_vector(aux_B[j], aux_u_j[idx_block], Au);
    PASE_Mg_restrict(solver, j, Au->vec, i, aux_B[i]->vec[idx_block]);
  }
  PASE_Aux_vector_destroy(Au);

  return 0;
}

/**
 * @brief 设置第 i 层的辅助矩阵
 *
 * @param solver  输入/输出参数
 * @param i       输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_pase_aux_matrix_create(PASE_MG_SOLVER solver, PASE_INT i)
{
  PASE_INT         block_size = solver->block_size;
  PASE_INT         idx_block  = 0;
  PASE_MATRIX     *A     = solver->multigrid->A;
  PASE_MATRIX     *B     = solver->multigrid->B;
  PASE_AUX_MATRIX *aux_A = solver->multigrid->aux_A;
  PASE_AUX_MATRIX *aux_B = solver->multigrid->aux_B;
  if(NULL == aux_A[i]) {
    aux_A[i] = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
    aux_A[i]->mat = A[i];
    aux_A[i]->is_mat_owner = PASE_NO;
    aux_A[i]->block_size   = block_size;

    aux_A[i]->is_diag      = PASE_NO;
#if 1
    aux_A[i]->Tmatvec      = 0.0;
    aux_A[i]->Tvecvec      = 0.0;
    aux_A[i]->Tveccom      = 0.0;
    aux_A[i]->Tblockb      = 0.0;
    aux_A[i]->Ttotal       = 0.0;
    aux_A[i]->Tinnergeneral= 0.0;
#endif

    aux_A[i]->vec = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
    aux_A[i]->block = (PASE_SCALAR**)PASE_Malloc(block_size*sizeof(PASE_SCALAR*));
    for(idx_block = 0; idx_block < block_size; ++idx_block) {
      aux_A[i]->vec[idx_block] = PASE_Vector_create_by_matrix_and_vector_data_operator(A[i], solver->u[0]->ops);
      aux_A[i]->block[idx_block] = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    }
  }
  if(NULL == aux_B[i]) {
    aux_B[i] = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
    aux_B[i]->mat = B[i];
    aux_B[i]->is_mat_owner = PASE_NO;
    aux_B[i]->block_size   = block_size;

    aux_B[i]->is_diag      = PASE_NO;
#if 1
    aux_B[i]->Tmatvec      = 0.0;
    aux_B[i]->Tvecvec      = 0.0;
    aux_B[i]->Tveccom      = 0.0;
    aux_B[i]->Tblockb      = 0.0;
    aux_B[i]->Ttotal       = 0.0;
    aux_B[i]->Tinnergeneral= 0.0;
#endif
    aux_B[i]->vec = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
    aux_B[i]->block = (PASE_SCALAR**)PASE_Malloc(block_size*sizeof(PASE_SCALAR*));
    for(idx_block = 0; idx_block < block_size; ++idx_block) {
      aux_B[i]->vec[idx_block] = PASE_Vector_create_by_matrix_and_vector_data_operator(B[i], solver->u[0]->ops);
      aux_B[i]->block[idx_block] = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    }
  }
  return 0;
}

/**
 * @brief 设置第 cur_level 层的辅助向量并初始化
 *
 * @param solver     输入/输出参数
 * @param cur_level  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT cur_level)
{
  PASE_INT block_size = solver->block_size;
  PASE_INT idx_eigen  = 0;
  if(NULL == solver->aux_u[cur_level]) {
    solver->aux_u[cur_level] = (PASE_AUX_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_AUX_VECTOR));
    solver->aux_u[cur_level][0] = PASE_Aux_vector_create_by_aux_matrix(solver->multigrid->aux_A[cur_level]);
    for(idx_eigen = 1; idx_eigen < block_size; idx_eigen++) {
      solver->aux_u[cur_level][idx_eigen] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[cur_level][0]);
    }
  }

  /*多次迭代需要多次初始化初值，但空间不需要重新申请*/
  for(idx_eigen = solver->nlock; idx_eigen < block_size; idx_eigen++) {
    PASE_Vector_set_constant_value(solver->aux_u[cur_level][idx_eigen]->vec, 0.0);
    memset(solver->aux_u[cur_level][idx_eigen]->block, 0, block_size*sizeof(PASE_SCALAR));
    solver->aux_u[cur_level][idx_eigen]->block[idx_eigen] = 1.0;
  }
  return 0;
}

/**
 * @brief 将辅助向量投影至更细的 (辅助) 空间中
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_prolong_from_pase_aux_vector(PASE_MG_SOLVER solver)
{
  PASE_INT cur_level = solver->idx_cycle_level[solver->cur_cycle_level];
  PASE_INT next_level = solver->idx_cycle_level[solver->cur_cycle_level-1];
  clock_t start, end;
  start = clock();
  if(0 == next_level) {
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(solver, cur_level, solver->aux_u[cur_level], next_level, solver->u);
  } else {
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_aux_vector(solver, cur_level, solver->aux_u[cur_level], next_level, solver->aux_u[next_level]);
  }
  end = clock();
  solver->prolong_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 将第 i 层的辅助向量投影至第 j 层标准空间中
 *
 * @param solver   输入/输出参数
 * @param i        输入参数
 * @param aux_u_i  输入参数
 * @param j        输入参数
 * @param u_j      输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_AUX_VECTOR *aux_u_i, PASE_INT j, PASE_VECTOR *u_j)
{
  PASE_INT idx_block = 0;
  PASE_INT jdx_block = 0;
  PASE_INT block_size = solver->block_size;
  PASE_INT nconv = solver->nconv;
  PASE_VECTOR *u_h = (PASE_VECTOR*)PASE_Malloc((block_size-nconv)*sizeof(PASE_VECTOR));
  for(idx_block = nconv; idx_block < block_size; ++idx_block) {
    u_h[idx_block-nconv] = PASE_Vector_create_by_vector(u_j[0]);
    PASE_Mg_prolong(solver, i, aux_u_i[idx_block]->vec, j, u_h[idx_block-nconv]);
    for(jdx_block = 0; jdx_block < block_size; ++jdx_block) {
      PASE_Vector_axpy(aux_u_i[idx_block]->block[jdx_block], u_j[jdx_block], u_h[idx_block-nconv]);
    }
  }
  for(idx_block = nconv; idx_block < block_size; ++idx_block) {
    PASE_Vector_copy(u_h[idx_block-nconv], u_j[idx_block]);
    PASE_Vector_destroy(u_h[idx_block-nconv]);
  }
  PASE_Free(u_h);

  return 0;
}

/**
 * @brief 将第 i 层辅助向量投影至第 j 层辅助空间中
 *
 * @param solver   输入/输出参数
 * @param i        输入参数
 * @param aux_u_i  输入参数
 * @param j        输入参数
 * @param aux_u_j  输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_prolong_from_pase_aux_vector_to_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_AUX_VECTOR *aux_u_i, PASE_INT j, PASE_AUX_VECTOR *aux_u_j)
{
  PASE_INT k = 0;
  PASE_INT l = 0;
  PASE_INT block_size = solver->block_size;
  PASE_INT nconv = solver->nconv;

  /* u_new += u1*u_0->aux_h */
  /* u_new->b_H += P*u0->b_H */
  PASE_AUX_VECTOR *u_new = (PASE_AUX_VECTOR*)PASE_Malloc((block_size-nconv)*sizeof(PASE_AUX_VECTOR));
  for(k = nconv; k < block_size; ++k) {
    u_new[k-nconv] = PASE_Aux_vector_create_by_aux_vector(aux_u_j[0]);
    PASE_Mg_prolong(solver, i, aux_u_i[k]->vec, j, u_new[k-nconv]->vec);
    for(l = 0; l < block_size; ++l) {
      PASE_Aux_vector_axpy(aux_u_i[k]->block[l], aux_u_j[l], u_new[k-nconv]);
    }
  }
  for(k = nconv; k < block_size; ++k) {
    PASE_Aux_vector_copy(u_new[k-nconv], aux_u_j[k]);
    PASE_Aux_vector_destroy(u_new[k-nconv]);
  }
  PASE_Free(u_new);

  return 0;
}

/**
 * @brief 将第 i 层网格上向量 u_i 投影到第 j 层的网格上, 并存储在向量 u_j 中
 *
 * @param solver
 * @param i
 * @param u_i
 * @param j
 * @param u_j
 *
 * @return 
 */
PASE_INT 
PASE_Mg_prolong(PASE_MG_SOLVER solver,  PASE_INT i, PASE_VECTOR u_i, PASE_INT j, PASE_VECTOR u_j)
{
  PASE_INT    idx_level = 0;
  PASE_VECTOR u_h       = NULL; 
  PASE_VECTOR u_H       = u_i;
  for(idx_level = i-1; idx_level >= j+1; idx_level--) {
    u_h = PASE_Vector_create_by_matrix_and_vector_data_operator(solver->multigrid->A[idx_level], u_i->ops);
    PASE_Matrix_multiply_vector(solver->multigrid->P[idx_level], u_H, u_h); 
    if(idx_level < i-1) {
      PASE_Vector_destroy(u_H);
    }
    u_H = u_h;
    u_h = NULL;
  }
  PASE_Matrix_multiply_vector(solver->multigrid->P[j], u_H, u_j); 
  if(i > j+1) {
    PASE_Vector_destroy(u_H);
  }
  return 0;
}

/**
 * @brief 将第 i 层网格的向量 u_i 限制到第 j 层网格上, 并储存在向量 u_j 中
 *
 * @param solver  输入参数
 * @param i       输入参数
 * @param u_i     输入参数
 * @param j       输入参数
 * @param u_j     输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_restrict(PASE_MG_SOLVER solver, PASE_INT i, PASE_VECTOR u_i, PASE_INT j, PASE_VECTOR u_j)
{
  PASE_INT    idx_level = 0;
  PASE_VECTOR tmp_u_H   = NULL;
  PASE_VECTOR tmp_u_h   = u_i;
  PASE_MATRIX *A = solver->multigrid->A;
  PASE_MATRIX *R = solver->multigrid->R;
  for(idx_level = i; idx_level <= j-2; ++idx_level) {
    tmp_u_H = PASE_Vector_create_by_matrix_and_vector_data_operator(A[idx_level+1], u_i->ops);
    PASE_Matrix_multiply_vector(R[idx_level], tmp_u_h, tmp_u_H);
    if(idx_level > i) {
      PASE_Vector_destroy(tmp_u_h);
    }
    tmp_u_h = tmp_u_H;
  }
  PASE_Matrix_multiply_vector(R[j-1], tmp_u_h, u_j);
  if(j > i+1) {
    PASE_Vector_destroy(tmp_u_h);
  }
  return 0;
}

/**
 * @brief 完成一次 PASE_Mg_cycle 后, 需计算残差及已收敛特征对个数. 已收敛特征对在之后的迭代中，不再计算和更改. 
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_error_estimate(PASE_MG_SOLVER solver)
{
  PASE_INT         block_size	 = solver->block_size; 
  PASE_INT         nconv         = solver->nconv;
  PASE_REAL        atol          = solver->atol;
  PASE_REAL        rtol          = solver->rtol;
  PASE_VECTOR     *u0	         = solver->u;
  PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
  PASE_MATRIX      A0	         = solver->multigrid->A[0];
  PASE_MATRIX      B0	         = solver->multigrid->B[0];

  /* 计算最细层的残差：r = Au - kMu */
  PASE_INT         flag          = 0;
  PASE_REAL       *check_multi   = (PASE_REAL*)PASE_Malloc((block_size-1)*sizeof(PASE_REAL));
  PASE_INT         i		 = 0;
  PASE_REAL        r_norm        = 1e+5;
  PASE_REAL        u_Bnorm       = 0.0;
  PASE_VECTOR      r             = PASE_Vector_create_by_vector(u0[0]);
  solver->nlock                  = nconv;

  if(NULL == solver->r_norm) {
    solver->r_norm = (PASE_REAL*)PASE_Malloc(block_size*sizeof(PASE_REAL));
  }

  for(i = nconv; i < block_size; ++i) {
    PASE_Vector_inner_product_general(u0[i], u0[i], B0, &u_Bnorm);
    u_Bnorm = sqrt(u_Bnorm);
    //PASE_Printf(MPI_COMM_WORLD, "bnrm = %f\n", u_Bnorm);
    PASE_Matrix_multiply_vector(B0, u0[i], r);
    PASE_Matrix_multiply_vector_general(1.0, A0, u0[i], -eigenvalues[i], r); 
    PASE_Vector_inner_product(r, r, &r_norm);
    r_norm	      = sqrt(r_norm)/u_Bnorm;
    solver->r_norm[i] = r_norm;
    if(i+1 < block_size) {
      check_multi[i] = fabs((eigenvalues[i]-eigenvalues[i+1])/eigenvalues[i]);
    }
    if(r_norm < atol || (r_norm/eigenvalues[i]) < rtol) {
      if(0 == flag) {
        solver->nconv++;
      }
    } else {
      flag = 1;
    }
  }
  //检查第一个为收敛的特征值与最后一个刚收敛的特征值是否有可能是重特征值，为保证之后的排序问题，需让重特征值同时在收敛的集合或未收敛的集合.
  while(solver->nconv > nconv && solver->nconv < block_size && check_multi[solver->nconv-1] < 1e-8) {
    solver->nconv--;
  }
  PASE_Free(check_multi);
  PASE_Vector_destroy(r);

  if(solver->print_level > 0) {
    //PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);	
    PASE_Printf(MPI_COMM_WORLD, "cycle = %d, nconv = %d, ", solver->ncycl, solver->nconv);
    if(solver->nconv < solver->block_size) {
      PASE_Printf(MPI_COMM_WORLD, "the first unconverged eigenvalues (residual) = %.8e (%1.6e)\n", solver->eigenvalues[solver->nconv], solver->r_norm[solver->nconv]);
    } else {
      PASE_Printf(MPI_COMM_WORLD, "all the wanted eigenpairs have converged.\n");
    }
  }	

  return 0;
}


/**
 * @brief 获得初始向量, 用户可以通过函数 PASE_Mg_function_create 给定具体的实现函数.
 *
 * @param solver  输入/输出参数 
 *
 * @return 
 */
PASE_INT
PASE_Mg_get_initial_vector(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  start = clock();
  solver->function->get_initial_vector(solver);
  end = clock();
  solver->get_initvec_time += ((double)(end-start))/CLK_TCK;
  if(solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "\nInitial   \t");
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);
  return 0;
}

/**
 * @brief 前光滑函数
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_presmoothing(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  if(solver->max_pre_iter > 0) {
    start = clock();
    if(solver->cur_cycle_level == 0) {
      solver->function->presmoothing(solver);
    } else {
      solver->function->presmoothing_aux(solver);
    }
    end = clock();
    solver->smooth_time += ((double)(end-start))/CLK_TCK;
    if(solver->print_level > 1) {
      PASE_Printf(MPI_COMM_WORLD, "\nPresmoothing\t");
    }
    PASE_Mg_print_eigenvalue_of_current_level(solver);
  }

  return 0;
}

/**
 * @brief 后光滑函数
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_postsmoothing(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  if(solver->max_post_iter > 0) {
    start = clock();
    if(solver->cur_cycle_level == 0) {
      solver->function->postsmoothing(solver);
      if(1 == solver->cycle_type) {
        PASE_Mg_orthogonalize(solver);
      }
    } else {
      solver->function->postsmoothing_aux(solver);
    }
    end = clock();
    solver->smooth_time += ((double)(end-start))/CLK_TCK;
    if(solver->print_level > 1) {
      PASE_Printf(MPI_COMM_WORLD, "\nPostsmoothing\t");
    }
    PASE_Mg_print_eigenvalue_of_current_level(solver);
  }

  return 0;
}

/**
 * @brief 最粗层直接求解辅助空间的特征值问题
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_direct_solve(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  if(solver->max_direct_iter> 0) {
    start = clock();
    solver->function->direct_solve(solver);
    end   = clock();
    solver->direct_solve_time += ((double)(end-start))/CLK_TCK;
    if(solver->print_level > 1) {
      PASE_Printf(MPI_COMM_WORLD, "\nDirect solve\t");
    }
    PASE_Mg_print_eigenvalue_of_current_level(solver);
  }
  return 0;
}

/**
 * @brief 正交化函数
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_orthogonalize(PASE_MG_SOLVER solver)
{
  PASE_INT cur_level 	 = solver->idx_cycle_level[solver->cur_cycle_level];
  PASE_INT block_size	 = solver->block_size;
  PASE_INT nconv         = solver->nconv;
  PASE_INT idx_eigen     = 0;

  if(cur_level == 0) {
    for(idx_eigen = nconv; idx_eigen < block_size; idx_eigen++) {
      PASE_Vector_orthogonalize_general(solver->u, idx_eigen, 0, idx_eigen-1, solver->multigrid->B[0]);
      PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->A[0], &(solver->eigenvalues[idx_eigen]));
    }
  } else {
    for(idx_eigen = 0; idx_eigen < block_size; idx_eigen++) {
      PASE_Aux_vector_orthogonalize_general(solver->aux_u[cur_level], idx_eigen, 0, idx_eigen-1, solver->multigrid->aux_B[cur_level]);
      PASE_Aux_vector_inner_product_general(solver->aux_u[cur_level][idx_eigen], solver->aux_u[cur_level][idx_eigen], solver->multigrid->aux_A[cur_level], &(solver->eigenvalues[idx_eigen]));
    }
  }

  return 0;
}

/**
 * @brief 设置 PASE_MG_SOLVER 结构体里的成员 PASE_MULTIGRID 
 *
 * @param solver  输入/输出参数
 *
 * @return PASE_MG_SOLVER
 */
PASE_INT
PASE_Mg_solver_set_multigrid(PASE_MG_SOLVER solver, PASE_MULTIGRID multigrid)
{
  if(NULL != solver->multigrid) {
    PASE_Multigrid_destroy(solver->multigrid);
  }
  solver->multigrid          = multigrid;
  return 0;
}

/**
 * @brief 设置 MG 求解的类型, 当前可选: 0. 两层网格校正
 *                                      1. 多重网格校正
 *
 * @param solver      输入\输出参数
 * @param cycle_type  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_cycle_type(PASE_MG_SOLVER solver, PASE_INT cycle_type)
{
  solver->cycle_type = cycle_type;
  return 0;
}

/**
 * @brief 设置特征值的求解个数
 *
 * @param solver      输入/输出向量
 * @param block_size  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_block_size(PASE_MG_SOLVER solver, PASE_INT block_size)
{
  solver->block_size        = block_size;
  solver->actual_block_size = block_size;
  return 0;
}

/**
 * @brief 设置特征值的最大求解个数, 主要用于申请足够大的内存空间存放求解的特征向量组
 *
 * @param solver          输入/输出向量
 * @param max_block_size  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_block_size(PASE_MG_SOLVER solver, PASE_INT max_block_size)
{
  solver->max_block_size = max_block_size;
  return 0;
}

/**
 * @brief 设置 MG 方法的最大迭代步数
 *
 * @param solver     输入/输出向量
 * @param max_cycle  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_cycle(PASE_MG_SOLVER solver, PASE_INT max_cycle)
{
  solver->max_cycle = max_cycle;
  return 0;
}

/**
 * @brief 设置 MG 方法每个迭代步中, 前光滑的最大光滑步数
 *
 * @param solver        输入/输出向量
 * @param max_pre_iter  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_pre_iteration(PASE_MG_SOLVER solver, PASE_INT max_pre_iter)
{
  solver->max_pre_iter = max_pre_iter;
  return 0;
}

/**
 * @brief 设置 MG 方法每个迭代步中, 后光滑的最大光滑步数
 *
 * @param solver         输入/输出向量
 * @param max_post_iter  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_post_iteration(PASE_MG_SOLVER solver, PASE_INT max_post_iter)
{
  solver->max_post_iter = max_post_iter;
  return 0;
}

/**
 * @brief 设置 MG 方法每个迭代步中, 辅助粗空间求解特征值问题的最大迭代步数
 *
 * @param solver           输入/输出参数
 * @param max_direct_iter  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_max_direct_iteration(PASE_MG_SOLVER solver, PASE_INT max_direct_iter)
{
  solver->max_direct_iter = max_direct_iter;
  return 0;
}

/**
 * @brief 设置 MG 方法停止迭代需满足的绝对残差精度要求
 *
 * @param solver  输入/输出向量
 * @param atol    输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_atol(PASE_MG_SOLVER solver, PASE_REAL atol)
{
  solver->atol = atol;
  return 0;
}

/**
 * @brief 设置 MG 方法停止迭代需满足的相对残差精度要求
 *
 * @param solver  输入/输出向量
 * @param rtol    输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_rtol(PASE_MG_SOLVER solver, PASE_REAL rtol)
{
  solver->rtol = rtol;
  return 0;
}

/**
 * @brief 设置 MG 方法的信息打印级别
 *
 * @param solver       输入/输出向量
 * @param print_level  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_print_level(PASE_MG_SOLVER solver, PASE_INT print_level)
{
  solver->print_level = print_level;
  return 0;
}

/**
 * @brief 设置问题的精确特征值, 可用于算法测试
 *
 * @param solver             输入/输出向量
 * @param exact_eigenvalues  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_exact_eigenvalues(PASE_MG_SOLVER solver, PASE_SCALAR *exact_eigenvalues)
{
  solver->exact_eigenvalues = exact_eigenvalues;
  return 0;
}

/**
 * @brief 用 CG 方法做光滑
 *
 * @param mg_solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_smoothing_by_cg(void *mg_solver, char *PreOrPost)
{
  PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
  PASE_INT     block_size = solver->block_size;
  PASE_INT     max_iter   = 0;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter   = solver->max_pre_iter;
    if(NULL == solver->method_pre) {
      solver->method_pre = "cg";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter   = solver->max_post_iter;
    if(NULL == solver->method_post) {
      solver->method_post = "cg";
    }
  }
  PASE_SCALAR *eigenvalues = solver->eigenvalues;

  PASE_INT idx_eigen;
  PASE_REAL tol = 1e-10;
  PASE_REAL inner_A, inner_B;
  PASE_VECTOR b = PASE_Vector_create_by_vector(solver->u[0]);
  //clock_t start, end;
  for(idx_eigen = solver->nconv; idx_eigen < block_size; idx_eigen++) {
    //start = clock();
    PASE_Matrix_multiply_vector_general(eigenvalues[idx_eigen], solver->multigrid->B[0], solver->u[idx_eigen], 0.0, b);
    PASE_Linear_solve_by_cg(solver->multigrid->A[0], b, solver->u[idx_eigen], tol, max_iter);

    PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->A[0], &inner_A);
    PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->B[0], &inner_B);
    eigenvalues[idx_eigen] = inner_A / inner_B;
    //PASE_Printf(MPI_COMM_WORLD, "after Cg, eigenvalues[%d] = %.12e\n", j, eigenvalues[j]);
    //end = clock();
    //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  }
  PASE_Vector_destroy(b);

  return 0;
}

/**
 * @brief 用 CG 方法做前光滑
 *
 * @param mg_solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_presmoothing_by_cg(void *mg_solver)
{
  PASE_Mg_smoothing_by_cg(mg_solver, "pre");
  return 0;
}

/**
 * @brief 用 CG 方法做后光滑
 *
 * @param mg_solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_postsmoothing_by_cg(void *mg_solver)
{
  PASE_Mg_smoothing_by_cg(mg_solver, "post");
  return 0;
}

/**
 * @brief 用 CG 方法做辅助空间内的光滑
 *
 * @param mg_solver  输入/输出向量
 *
 * @return
 */
PASE_INT
PASE_Mg_smoothing_by_cg_aux(void *mg_solver, char *PreOrPost)
{
  PASE_MG_SOLVER solver     = (PASE_MG_SOLVER)mg_solver;
  PASE_INT       cur_level  = solver->idx_cycle_level[solver->cur_cycle_level];
  PASE_INT       block_size = solver->block_size;
  PASE_INT       max_iter   = 1;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter   = solver->max_pre_iter + cur_level * 0.5;
    if(NULL == solver->method_pre_aux) {
      solver->method_pre_aux = "cg";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter   = solver->max_post_iter + cur_level * 0.5;
    if(NULL == solver->method_post_aux) {
      solver->method_post_aux = "cg";
    }
  }

  PASE_SCALAR   *eigenvalues = solver->eigenvalues;

  PASE_INT idx_eigen;
  PASE_REAL tol = 1e-10;
  PASE_REAL inner_A, inner_B;
  PASE_AUX_VECTOR *aux_u = solver->aux_u[cur_level];
  PASE_AUX_VECTOR  aux_b = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
  //clock_t start, end;
  for(idx_eigen = solver->nconv; idx_eigen < block_size; idx_eigen++) {
    //start = clock();
    PASE_Aux_matrix_multiply_aux_vector_general(eigenvalues[idx_eigen], solver->multigrid->aux_B[cur_level], aux_u[idx_eigen], 0.0, aux_b);
    PASE_Linear_solve_by_cg_aux(solver->multigrid->aux_A[cur_level], aux_b, aux_u[idx_eigen], tol, max_iter);

    PASE_Aux_vector_inner_product_general(aux_u[idx_eigen], aux_u[idx_eigen], solver->multigrid->aux_A[cur_level], &inner_A);
    PASE_Aux_vector_inner_product_general(aux_u[idx_eigen], aux_u[idx_eigen], solver->multigrid->aux_B[cur_level], &inner_B);
    eigenvalues[idx_eigen] = inner_A / inner_B;
    //PASE_Printf(MPI_COMM_WORLD, "after Cg, eigenvalues[%d] = %.12e\n", j, eigenvalues[j]);
    //end = clock();
    //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  }
  PASE_Aux_vector_destroy(aux_b);

  return 0;
}

/**
 * @brief 用 CG 方法做辅助空间内的前光滑
 *
 * @param mg_solver  输入/输出向量
 *
 * @return
 */
PASE_INT
PASE_Mg_presmoothing_by_cg_aux(void *mg_solver)
{
  PASE_Mg_smoothing_by_cg_aux(mg_solver, "pre");
  return 0;
}

/**
 * @brief 用 CG 方法做辅助空间内的后光滑
 *
 * @param mg_solver  输入/输出向量
 *
 * @return
 */
PASE_INT
PASE_Mg_postsmoothing_by_cg_aux(void *mg_solver)
{
  PASE_Mg_smoothing_by_cg_aux(mg_solver, "post");
  return 0;
}

/**
 * @brief 用 CG 方法解线性方程组
 *
 * @param A         输入参数
 * @param b         输入参数
 * @param x         输入/输出参数
 * @param tol       输入参数
 * @param max_iter  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Linear_solve_by_cg(PASE_MATRIX A, PASE_VECTOR b, PASE_VECTOR x, PASE_REAL tol, PASE_INT max_iter)
{
  PASE_INT iter = 0;
  PASE_REAL bnorm = 1.0, rnorm = 1.0, rho = 1.0, rho_1 = 1.0, alpha = 1.0, beta = 1.0, tmp = 1.0;
  PASE_VECTOR r = PASE_Vector_create_by_vector(x);
  PASE_VECTOR p = PASE_Vector_create_by_vector(x);
  PASE_VECTOR q = PASE_Vector_create_by_vector(x);

  //start = clock();
  PASE_Vector_inner_product(b, b, &bnorm);
  bnorm = sqrt(bnorm);
  PASE_Vector_copy(b, r);
  PASE_Matrix_multiply_vector_general(-1.0, A, x, 1.0, r);
  PASE_Vector_inner_product(r, r, &rho);
  rnorm = sqrt(rho);
  if(rnorm/bnorm < tol) {
    return 0;
  }
  for(iter=0; iter<max_iter; iter++) {
    if(iter>0) {
      beta = rho / rho_1; 
      PASE_Vector_scale(beta, p);
      PASE_Vector_axpy(1.0, r, p);
    } else {
      PASE_Vector_copy(r, p);
    }
    PASE_Matrix_multiply_vector(A, p, q);
    PASE_Vector_inner_product(p, q, &tmp);
    alpha = rho / tmp;
    PASE_Vector_axpy(alpha, p, x);
    PASE_Vector_axpy(-1.0*alpha, q, r);

    rho_1 = rho;
    PASE_Vector_inner_product(r, r, &rho);
    rnorm = sqrt(rho);

    if(rnorm/bnorm < tol) {
      break;
    }
  }
  //end = clock();
  //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  //PASE_Printf(MPI_COMM_WORLD, "iter = %d, rnorm/bnorm = %e\n", iter, rnorm/bnorm);
  PASE_Vector_destroy(r);
  PASE_Vector_destroy(p);
  PASE_Vector_destroy(q);
  return 0;
}

/**
 * @brief 用 CG 方法解辅助空间内的线性方程组
 *
 * @param aux_A     输入参数
 * @param aux_b     输入参数
 * @param aux_x     输入/输出参数
 * @param tol       输入参数
 * @param max_iter  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Linear_solve_by_cg_aux(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_b, PASE_AUX_VECTOR aux_x, PASE_REAL tol, PASE_INT max_iter)
{
  PASE_INT iter = 0;
  PASE_REAL bnorm = 1.0, rnorm = 1.0, rho = 1.0, rho_1 = 1.0, alpha = 1.0, beta = 1.0, tmp = 1.0;
  PASE_AUX_VECTOR aux_r = PASE_Aux_vector_create_by_aux_vector(aux_x);
  PASE_AUX_VECTOR aux_p = PASE_Aux_vector_create_by_aux_vector(aux_x);
  PASE_AUX_VECTOR aux_q = PASE_Aux_vector_create_by_aux_vector(aux_x);

  //start = clock();
  PASE_Aux_vector_inner_product(aux_b, aux_b, &bnorm);
  bnorm = sqrt(bnorm);
  PASE_Aux_vector_copy(aux_b, aux_r);
  PASE_Aux_matrix_multiply_aux_vector_general(-1.0, aux_A, aux_x, 1.0, aux_r);
  PASE_Aux_vector_inner_product(aux_r, aux_r, &rho);
  rnorm = sqrt(rho);
  if(rnorm/bnorm < tol) {
    PASE_Aux_vector_destroy(aux_r);
    PASE_Aux_vector_destroy(aux_p);
    PASE_Aux_vector_destroy(aux_q);
    return 0;
  }
  for(iter = 0; iter < max_iter; iter++) {
    if(iter>0) {
      beta = rho / rho_1; 
      PASE_Aux_vector_scale(beta, aux_p);
      PASE_Aux_vector_axpy(1.0, aux_r, aux_p);
    } else {
      PASE_Aux_vector_copy(aux_r, aux_p);
    }
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_p, aux_q);
    PASE_Aux_vector_inner_product(aux_p, aux_q, &tmp);
    alpha = rho / tmp;
    PASE_Aux_vector_axpy(alpha, aux_p, aux_x);
    PASE_Aux_vector_axpy(-1.0*alpha, aux_q, aux_r);

    rho_1 = rho;
    PASE_Aux_vector_inner_product(aux_r, aux_r, &rho);
    rnorm = sqrt(rho);

    if(rnorm/bnorm < tol) {
      break;
    }
  }
  //end = clock();
  //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  PASE_Aux_vector_destroy(aux_r);
  PASE_Aux_vector_destroy(aux_p);
  PASE_Aux_vector_destroy(aux_q);
  return 0;
}

/**
 * @brief  特征值问题的 MG 求解
 *
 * @param A           输入参数
 * @param B           输入参数
 * @param eval        输入/输出参数
 * @param evec        输入/输出参数
 * @param block_size  输入参数
 * @param param       输入参数
 */
PASE_INT
PASE_EigenSolver(PASE_MATRIX A, PASE_MATRIX B, PASE_SCALAR *eval, PASE_VECTOR *evec, PASE_INT block_size, PASE_PARAMETER param)
{
  PASE_INT i              = 0;
  PASE_INT max_block_size = ((2*block_size)<(block_size+5))?(2*block_size):(block_size+5);
  PASE_MG_SOLVER solver   = PASE_Mg_solver_create(A, B, param);

  PASE_Mg_set_cycle_type(solver, param->cycle_type);
  PASE_Mg_set_block_size(solver, block_size);
  PASE_Mg_set_max_block_size(solver, max_block_size);
  PASE_Mg_set_max_cycle(solver, param->max_cycle);
  PASE_Mg_set_max_pre_iteration(solver, param->max_pre_iter);
  PASE_Mg_set_max_post_iteration(solver, param->max_post_iter);
  PASE_Mg_set_atol(solver, param->atol);
  PASE_Mg_set_rtol(solver, param->rtol);
  PASE_Mg_set_print_level(solver, param->print_level);

  solver->u = (PASE_VECTOR*)PASE_Malloc(max_block_size*sizeof(PASE_VECTOR));
  if(NULL != evec) {
    for(i = 0; i < block_size; ++i) {
      solver->u[i] = evec[i];
    }
    for(i = block_size; i < max_block_size; ++i) {
      solver->u[i] = PASE_Vector_create_by_vector(solver->u[0]);
    }
    solver->is_u_owner = PASE_NO;
  }
  PASE_Mg_set_up(solver, A, B, evec[0], param);
  PASE_Mg_solve(solver);
  PASE_Mg_solver_destroy(solver);
  return 0;
}

/**
 * @brief 用 GCG 方法求解辅助粗空间的特征值问题
 *
 * @param mg_solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_direct_solve_by_gcg(void *mg_solver)
{
  PASE_MG_SOLVER   solver      = (PASE_MG_SOLVER)mg_solver;
  PASE_INT         cur_level   = solver->idx_cycle_level[solver->max_cycle_level];
  PASE_INT         block_size  = solver->block_size;

  PASE_AUX_MATRIX  aux_A       = solver->multigrid->aux_A[cur_level];            
  PASE_AUX_MATRIX  aux_B       = solver->multigrid->aux_B[cur_level];            
  PASE_AUX_VECTOR *aux_u       = solver->aux_u[cur_level];
  PASE_SCALAR     *eigenvalues = solver->eigenvalues;

  PASE_INT         max_iter    = solver->max_direct_iter;
  PASE_REAL        gcg_tol     = solver->atol*1e-2;
  PASE_REAL        cg_tol      = solver->atol*1e-2;
  PASE_INT         cg_max_iter = 50;

#if DIAG_GCG
  PASE_Mg_precondition_for_gcg(mg_solver);
  solver->method_dire = "gcg with diag scheme";
#else
  solver->method_dire = "gcg";
#endif

#if 0
  PASE_INT        i      = 0;
  PASE_REAL       r_norm = 0;
  PASE_AUX_VECTOR tmp    = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
  for(i = 0; i<block_size; ++i) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_u[i], tmp);
    PASE_Aux_matrix_multiply_aux_vector_general(-eigenvalues[i], aux_B, aux_u[i], 1.0, tmp);
    PASE_Aux_vector_inner_product(tmp, tmp, &r_norm);
    r_norm	= sqrt(r_norm);
    PASE_Printf(MPI_COMM_WORLD, "eigenvalues[%d] = %.8e, residual[%d] = %.6e\n", i, eigenvalues[i], i, r_norm);
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");
#endif

  GCG_Eigen(aux_A, aux_B, 0, eigenvalues, aux_u, block_size, gcg_tol, cg_tol, max_iter, cg_max_iter, solver->nconv, &(solver->time_inner), &(solver->time_lapack), &(solver->time_other), &(solver->time_orth_gcg));

#if 0
  for(i = 0; i < block_size; ++i) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_u[i], tmp);
    PASE_Aux_matrix_multiply_aux_vector_general(-eigenvalues[i], aux_B, aux_u[i], 1.0, tmp);
    PASE_Aux_vector_inner_product(tmp, tmp, &r_norm);
    r_norm	= sqrt(r_norm);
    PASE_Printf(MPI_COMM_WORLD, "eigenvalues[%d] = %.8e, residual[%d] = %.6e\n", i, eigenvalues[i], i, r_norm);
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");
  PASE_Aux_vector_destroy(tmp);
#endif

#if DIAG_GCG
  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i = solver->nlock; i < block_size; ++i) {
    for(j = 0; j < block_size; ++j) {
      PASE_Vector_axpy(-aux_u[i]->block[j], aux_A->vec[j], aux_u[i]->vec);
    }
  }
#endif

  return 0;
}

/**
 * @brief 对矩阵 A 对角化预处理, 暂时只是针对 GCG 方法设计的
 *
 * @param mg_solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_precondition_for_gcg(void *mg_solver)
{
  clock_t start, end;
  clock_t start_total, end_total;
  start_total = clock();

  PASE_MG_SOLVER   solver      = (PASE_MG_SOLVER)mg_solver;
  PASE_INT         cur_level   = solver->idx_cycle_level[solver->max_cycle_level];
  PASE_INT         block_size  = solver->block_size;
  PASE_INT         nlock       = solver->nlock;

  PASE_AUX_MATRIX  aux_A       = solver->multigrid->aux_A[cur_level];            
  PASE_AUX_MATRIX  aux_B       = solver->multigrid->aux_B[cur_level];            
  PASE_AUX_VECTOR *aux_u       = solver->aux_u[cur_level];

  PASE_INT  i        = 0;
  PASE_INT  j        = 0;
  PASE_INT  max_iter = 10;
  PASE_REAL tol      = 1e-12;
  PASE_INT  size_tmp = (block_size-nlock) * block_size;
  PASE_SCALAR *tmp1_block = (PASE_SCALAR*)PASE_Malloc(size_tmp*sizeof(PASE_SCALAR));
  PASE_SCALAR *tmp2_block = (PASE_SCALAR*)PASE_Malloc(size_tmp*sizeof(PASE_SCALAR));
  MPI_Status status;
  MPI_Request *requests1 = (MPI_Request*)PASE_Malloc((block_size-nlock)*sizeof(MPI_Request));
  MPI_Request *requests2 = (MPI_Request*)PASE_Malloc((block_size-nlock)*sizeof(MPI_Request));
  if(NULL == solver->amg_data_coarsest) {
    start = clock();
    HYPRE_Solver amg_solver = NULL;
    HYPRE_BoomerAMGCreate(&amg_solver);
    HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0); /* print amg solution info */
    HYPRE_BoomerAMGSetOldDefault(amg_solver);    /* Falgout coarsening with modified classical interpolaiton */
    HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);  /* G-S/Jacobi hybrid relaxation */
    HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1); /* uses C/F relaxation */
    HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);  /* 2 sweeps of smoothing */
    HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
    HYPRE_BoomerAMGSetup(amg_solver, (HYPRE_ParCSRMatrix)aux_A->mat->matrix_data, (HYPRE_ParVector)aux_A->vec[0]->vector_data, (HYPRE_ParVector)aux_u[0]->vec->vector_data);
    solver->amg_data_coarsest = (void*) amg_solver;
    end = clock();
    solver->time_linear_diag += ((double)(end-start))/CLK_TCK;
  }
  //y = A^{-1} * b, tmp1_block = yT * b
  for(i = nlock; i < block_size; ++i) {
    for(j = 0; j < nlock; ++j) {
      tmp1_block[(i-nlock)*block_size + j] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_A->vec[i]->vector_data)),  hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_A->vec[j]->vector_data)));
    }
    //PASE_Linear_solve_by_cg(aux_A->mat, aux_A->vec[i], aux_u[i]->vec, tol, max_iter);
    start = clock();
    PASE_Linear_solve_by_amg_hypre(aux_A->mat, &(aux_A->vec[i]), &(aux_u[i]->vec), 1, tol, max_iter, solver->amg_data_coarsest);
    end = clock();
    solver->time_linear_diag += ((double)(end-start))/CLK_TCK;
    for(j = nlock; j <= i; ++j) {
      tmp1_block[(i-nlock)*block_size + j] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_u[i]->vec->vector_data)),  hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_A->vec[j]->vector_data)));
    }
    MPI_Iallreduce(MPI_IN_PLACE, &(tmp1_block[(i-nlock)*block_size]), i+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests1[i-nlock]));
  }

  //x = a - B * y, tmp2_block = yT * x + aT * y
  for(i = nlock; i < block_size; ++i) {
    for(j = 0; j < nlock; ++j) {
      tmp2_block[(i-nlock)*block_size + j] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_u[i]->vec->vector_data)),  hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_B->vec[j]->vector_data)));
      tmp2_block[(i-nlock)*block_size + j] += hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_B->vec[i]->vector_data)),  hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_A->vec[j]->vector_data)));
    }
    PASE_Matrix_multiply_vector_general(-1.0, aux_B->mat, aux_u[i]->vec, 0.0, aux_A->vec[i]);
    PASE_Vector_axpy(1.0, aux_B->vec[i], aux_A->vec[i]);
    for(j = nlock; j <= i; ++j) {
      tmp2_block[(i-nlock)*block_size + j] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_A->vec[i]->vector_data)),  hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_u[j]->vec->vector_data)));
      tmp2_block[(i-nlock)*block_size + j] += hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_B->vec[j]->vector_data)),  hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_u[i]->vec->vector_data)));
    }
    MPI_Iallreduce(MPI_IN_PLACE, &(tmp2_block[(i-nlock)*block_size]), i+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests2[i-nlock]));
  }
  //a = x
  PASE_VECTOR tmp = NULL;
  for(i = 0; i < nlock; ++i) {
    PASE_Vector_copy(aux_A->vec[i], aux_u[i]->vec);
  }
  for(i = nlock; i < block_size; ++i) {
    tmp = aux_B->vec[i];
    aux_B->vec[i] = aux_A->vec[i];
    aux_A->vec[i] = tmp;
    PASE_Vector_copy(aux_u[i]->vec, aux_A->vec[i]);
  }

  //beta = beta - tmp1_block, alpha = alpha - tmp2_block
  for(i = nlock ; i < block_size; ++i) {
    MPI_Wait(&(requests1[i-nlock]), &status);
    for(j = 0; j <= i; ++j) {
      aux_A->block[i][j] -= tmp1_block[(i-nlock)*block_size+j];
      if(j < i) {
        //aux_A->block[j][i] -= tmp1_block[(i-nlock)*block_size+j];
        aux_A->block[j][i] = aux_A->block[i][j];
      }
    }
  }
  for(i = nlock; i < block_size; ++i) {
    MPI_Wait(&(requests2[i-nlock]), &status);
    for(j = 0; j <= i; ++j) {
      aux_B->block[i][j] -= tmp2_block[(i-nlock)*block_size+j];
      if(j < i) {
        aux_B->block[j][i] = aux_B->block[i][j];
        //aux_B->block[j][i] -= tmp2_block[(i-nlock)*block_size+j];
      }
    }
  }
  aux_A->is_diag = PASE_YES;

  PASE_Free(tmp1_block);
  PASE_Free(tmp2_block);
  PASE_Free(requests1);
  PASE_Free(requests2);

  end_total = clock();
  solver->time_diag_pre += ((double)(end_total-start_total))/CLK_TCK;

  return 0;
}

/**
 * @brief 打印计算所得特征值，及其对应的残差.
 *
 * @param solver  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_print(PASE_MG_SOLVER solver)
{
  PASE_INT idx_eigen = 0;
  if(solver->print_level > 0) {
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    for(idx_eigen=0; idx_eigen<solver->block_size; idx_eigen++) {
      PASE_Printf(MPI_COMM_WORLD, "%d-th eig=%.8e, aresi = %.8e\n", idx_eigen, solver->eigenvalues[idx_eigen], solver->r_norm[idx_eigen]);
    }
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "set up time       = %f seconds\n", solver->set_up_time);
    PASE_Printf(MPI_COMM_WORLD, "get initvec time  = %f seconds\n", solver->get_initvec_time);
    PASE_Printf(MPI_COMM_WORLD, "smooth time       = %f seconds\n", solver->smooth_time);
    PASE_Printf(MPI_COMM_WORLD, "set aux time      = %f seconds\n", solver->set_aux_time);
    PASE_Printf(MPI_COMM_WORLD, "prolong time      = %f seconds\n", solver->prolong_time);
    PASE_Printf(MPI_COMM_WORLD, "direct solve time = %f seconds\n", solver->direct_solve_time);
    PASE_Printf(MPI_COMM_WORLD, "total solve time  = %f seconds\n", solver->total_solve_time);
    PASE_Printf(MPI_COMM_WORLD, "total time        = %f seconds\n", solver->total_time);
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "Direct solve time statistics\n");
    //PASE_Printf(MPI_COMM_WORLD, "Tmatvec     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tmatvec+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tmatvec);
    //PASE_Printf(MPI_COMM_WORLD, "Tveccom     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tveccom+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tveccom);
    //PASE_Printf(MPI_COMM_WORLD, "Tvecvec     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tvecvec+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tvecvec);
    //PASE_Printf(MPI_COMM_WORLD, "Tblockb     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tblockb+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tblockb);
    //PASE_Printf(MPI_COMM_WORLD, "TMatVec     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Ttotal+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Ttotal);
    //PASE_Printf(MPI_COMM_WORLD, "TVecVec     = %f seconds\n", solver->time_inner+solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tinnergeneral+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tinnergeneral);
    PASE_Printf(MPI_COMM_WORLD, "TLapack     = %f seconds\n", solver->time_lapack);
    PASE_Printf(MPI_COMM_WORLD, "Torth       = %f seconds\n", solver->time_orth_gcg);
    PASE_Printf(MPI_COMM_WORLD, "Tother      = %f seconds\n", solver->time_other);
    PASE_Printf(MPI_COMM_WORLD, "Tdiagpre    = %f seconds\n", solver->time_diag_pre);
    PASE_Printf(MPI_COMM_WORLD, "Tdiaglinear = %f seconds\n", solver->time_linear_diag);
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    if(NULL != solver->method_init) {
      PASE_Printf(MPI_COMM_WORLD, "Get initial vector:         %s\n", solver->method_init);
    }
    if(NULL != solver->method_pre) {
      PASE_Printf(MPI_COMM_WORLD, "Presmoothing:               %s\n", solver->method_pre);
    }
    if(NULL != solver->method_post) {
      PASE_Printf(MPI_COMM_WORLD, "Postsmoothing:              %s\n", solver->method_post);
    }
    if(NULL != solver->method_pre_aux) {
      PASE_Printf(MPI_COMM_WORLD, "Presmoothing in aux space:  %s\n", solver->method_pre_aux);
    }
    if(NULL != solver->method_post_aux) {
      PASE_Printf(MPI_COMM_WORLD, "Postsmoothing in aux space: %s\n", solver->method_post_aux);
    }
    if(NULL != solver->method_dire) {
      PASE_Printf(MPI_COMM_WORLD, "Direct solve:               %s\n", solver->method_dire);
    }
  }	
  return 0;
}

/**
 * @brief 打印当前的近似特征向量和当前所在的层数
 *
 * @param solver  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_print_eigenvalue_of_current_level(PASE_MG_SOLVER solver)
{
  PASE_INT i = 0;
  if(solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "%d-level\n", solver->idx_cycle_level[solver->cur_cycle_level]);
    for(i = 0; i < solver->block_size; ++i) {
      PASE_Printf(MPI_COMM_WORLD, "%d-th eig=%.8e\n", i, solver->eigenvalues[i]);
    }
  }
  return 0;
}

