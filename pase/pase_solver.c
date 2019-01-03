#include "pase_solver.h"

/**
 * @brief  特征值问题的 MG 求解
 *
 * @param A           输入参数
 * @param B           输入参数
 * @param eval        输入/输出参数,可为NULL
 * @param evec        输入/输出参数,可为NULL
 * @param nev         输入参数（表示求解特征值的个数）
 * @param param       输入参数
 * @param gcge_ops    输入参数
 */
//------------------------------------------------------------------------------------------------
PASE_INT
PASE_EigenSolver(void *A, void *B, PASE_SCALAR *eval, void **evec, 
        PASE_INT nev, PASE_PARAMETER param, GCGE_OPS *gcge_ops)
{
  //创建solver空间并赋初值NULL
  //pase的eigensolver，依赖于MG
  PASE_MG_SOLVER solver = PASE_Mg_solver_create(param);

  //进行AMG分层，分配工作空间
  PASE_Mg_set_up(solver, A, B, gcge_ops);

  //进行pase求解
  PASE_Mg_solve(solver);

  //将解返回给用户提供的特征对空间
  if(eval != NULL) {
    memcpy(eval, solver->eigenvalues, nev*sizeof(PASE_SCALAR));
  }
  if(evec != NULL) {
    PASE_INT mv_s[2] = { 0, 0};
    PASE_INT mv_e[2] = { nev, nev };
    gcge_ops->MultiVecAxpby(1.0, solver->sol[0], 0.0, evec, mv_s, mv_e, gcge_ops);
  }

  //释放空间
  PASE_Mg_solver_destroy(solver);

  return 0;
}


/**
 * @brief 创建 PASE_MG_SOLVER
 *
 * @param param  输入参数
 *
 * @return PASE_MG_SOLVER
 */
PASE_MG_SOLVER
PASE_Mg_solver_create(PASE_PARAMETER param)
{
  PASE_MG_SOLVER solver = (PASE_MG_SOLVER)PASE_Malloc(sizeof(PASE_MG_SOLVER_PRIVATE));

  //给网格层数即层号赋值
  PASE_INT i = 0;
  //num_levels表示多重网格矩阵一共有多少层
  solver->num_levels = param->num_levels;
  //initial_level表示V_h_1 所在的层号，如果用户提供的-1,那就默认取最粗层
  if(param->initial_level == -1) {
    solver->initial_level = solver->num_levels - 1;
  } else {
    solver->initial_level = param->initial_level;
  }
  if(param->coarest_level == -1) {
    solver->coarest_level = solver->num_levels - 1;
  } else {
    solver->coarest_level = param->coarest_level;
  }
  if(param->finest_level == -1) {
    solver->finest_level = 0;
  } else {
    solver->finest_level = param->finest_level;
  }

  //max_cycle_count_each_level表示每层最多做多少次二网格迭代
  solver->max_cycle_count_each_level  = param->max_cycle_count_each_level;
  //每层上最多前光滑的次数
  solver->max_pre_count_each_level    = param->max_pre_count_each_level;
  //每层上最多后光滑的次数
  solver->max_post_count_each_level   = param->max_post_count_each_level;
  //每层上最多直接求解时GCGE迭代的次数
  solver->max_direct_count_each_level = param->max_direct_count_each_level;
  //第一层直接求解时的最大迭代次数
  solver->max_initial_count           = param->max_initial_count;

  //实际要求的特征对个数
  solver->nev = param->nev;
  //最多算多少个特征对
  solver->max_nev = ((2* param->nev)<(param->nev+5))?(2* param->nev):(param->nev+5);
  //实际计算的特征值个数
  if(param->num_given_eigs >= param->nev) {
    solver->pase_nev = param->num_given_eigs;
  } else {
    solver->pase_nev = solver->max_nev;
  }
  solver->num_given_eigs = param->num_given_eigs;
  //BMG进行分批计算时，每次最多计算的线性方程组个数
  if(param->bmg_step_size = -1) {
    solver->bmg_step_size = solver->max_nev;
  } else {
    solver->bmg_step_size = param->bmg_step_size;
  }
  //相对残差收敛准则，||A*x-\lambda*B*x||_2/(\lambda*||x||_2) < rtol 则为收敛
  solver->rtol = param->rtol;
  //绝对残差收敛准则，||A*x-\lambda*B*x||_2/||x||_2 < atol 则为收敛
  solver->atol = param->atol;
  //以上两种收敛准则满足其一即可认定为收敛
  //打印level
  solver->print_level = param->print_level; 
  //abs_res_norm用于存储特征对的绝对残差
  solver->abs_res_norm = NULL;
  //已经收敛的特征对个数
  solver->nconv = 0;

  //特征值空间
  solver->eigenvalues = NULL;
  //解空间，多重网格迭代时，用于存储每一层的解
  solver->sol = NULL;
  //每一层的右端项空间, 分配空间为bmg_step_size
  //也在复合矩阵中用作aux_A->aux_Hh 
  //因此在可能会用作VH的层上，需要给rhs分配的空间为max_nev
  //细层的rhs还会用作P*u_H, 用于prolong 
  solver->rhs = NULL;
  //每一层BCG迭代时的P向量空间, 分配空间为bmg_step_size
  //也在复合矩阵中用作aux_B->aux_Hh 
  //因此在可能会用作VH的层上，需要给cg_p分配的空间为max_nev
  solver->cg_p = NULL;
  //每一层BCG迭代时的W向量空间
  solver->cg_w = NULL;
  //每一层BCG迭代时的残差向量空间
  solver->cg_res = NULL;
  //每一层BCG迭代时的double型临时空间
  solver->cg_double_tmp = NULL;
  //每一层BCG迭代时的int型临时空间
  solver->cg_int_tmp = NULL;
  //普通矩阵向量操作
  solver->gcge_ops = NULL;
  //pase aux 复合矩阵的矩阵向量操作
  solver->pase_ops = NULL;
  //pase aux 复合向量
  solver->aux_u = NULL;
  //pase aux 复合矩阵
  solver->aux_A = NULL;
  //pase aux 复合矩阵
  solver->aux_B = NULL;
  //多重网格结构
  solver->multigrid = NULL;

  //统计时间
  solver->set_up_time = 0.0;
  solver->smooth_time = 0.0;
  solver->set_aux_time = 0.0;
  solver->prolong_time = 0.0;
  solver->direct_solve_time = 0.0;
  solver->total_solve_time = 0.0;
  solver->total_time = 0.0;

  return solver;
}

/**
 * @brief PASE_MG_SOLVER 的准备阶段
 *
 * @param solver   输入/输出参数
 * @param A        输入参数
 * @param B        输入参数
 * @param gcge_ops 输入参数
 *
 * @return 
 */
//1. 针对矩阵A， B进行分层，得到多重网格结构 (分解的层数由前面的nlevel确定)， 但是还没有考虑nev的影响
//2. 给Pase的eigen Solver 分配相应的空间， 包括线性解法器所需要的空间
PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver, void *A, void *B, GCGE_OPS *gcge_ops)
{
  clock_t start, end;
  start = clock();

  //用gcge_ops创建pase_ops
  PASE_OPS_Create(&(solver->pase_ops), gcge_ops);
  solver->gcge_ops = gcge_ops;

  //以矩阵A，B作为最细层空间，创建多重网格结构
  //TODO 这里可能会修改max_levels?
  PASE_INT error = PASE_MULTIGRID_Create(&(solver->multigrid), 
	solver->num_levels, A, B, gcge_ops, solver->pase_ops);

  //--------------------------------------------------------------------
  //--特征值工作空间-----------------------------------------------------
  PASE_INT max_nev = solver->max_nev;
  solver->eigenvalues = (PASE_SCALAR*)calloc(max_nev, sizeof(PASE_SCALAR));
  //abs_res_norm存储每个特征对的绝对残差
  solver->abs_res_norm = (PASE_REAL*)calloc(max_nev, sizeof(PASE_REAL));

  //--普通向量组空间-----------------------------------------------------
  PASE_INT num_levels = solver->num_levels;
  PASE_INT *sol_size    = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *rhs_size    = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *cg_p_size   = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *cg_w_size   = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *cg_res_size = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  void ***sol           = (void***)malloc(num_levels * sizeof(void**));
  void ***rhs           = (void***)malloc(num_levels * sizeof(void**));
  void ***cg_p          = (void***)malloc(num_levels * sizeof(void**));
  void ***cg_w          = (void***)malloc(num_levels * sizeof(void**));
  void ***cg_res        = (void***)malloc(num_levels * sizeof(void**));
  //层指标
  PASE_INT idx_level = 0;
  //分批计算时的step_size
  PASE_INT step_size = solver->bmg_step_size;
  //先确定每个向量组要取多少个向量 
  //目前是针对bmg_step_size == max_nev, TODO 需要再更新以适应bmg_step_size != max_nev 的情况
  for(idx_level=0; idx_level< solver->num_levels; idx_level++) {
    sol_size[idx_level]    = max_nev;
    rhs_size[idx_level]    = step_size;
    cg_p_size[idx_level]   = step_size;
    cg_w_size[idx_level]   = step_size;
    cg_res_size[idx_level] = step_size;
  }
  //具体的在每一层上分配空间（节约内存开销）
  void **A_array = solver->multigrid->A_array;
  for(idx_level=0; idx_level< solver->num_levels; idx_level++) {
    A_array[idx_level];
    gcge_ops->MultiVecCreateByMat(&(sol[idx_level]),    sol_size[idx_level],    A_array[idx_level], gcge_ops);
    gcge_ops->MultiVecCreateByMat(&(rhs[idx_level]),    rhs_size[idx_level],    A_array[idx_level], gcge_ops);
    gcge_ops->MultiVecCreateByMat(&(cg_p[idx_level]),   cg_p_size[idx_level],   A_array[idx_level], gcge_ops);
    gcge_ops->MultiVecCreateByMat(&(cg_w[idx_level]),   cg_w_size[idx_level],   A_array[idx_level], gcge_ops);
    gcge_ops->MultiVecCreateByMat(&(cg_res[idx_level]), cg_res_size[idx_level], A_array[idx_level], gcge_ops);
  }

  solver->sol         = sol;
  solver->rhs         = rhs;
  solver->cg_p        = cg_p;
  solver->cg_w        = cg_w;
  solver->cg_res      = cg_res;
  solver->sol_size    = sol_size;
  solver->rhs_size    = rhs_size;
  solver->cg_p_size   = cg_p_size;
  solver->cg_w_size   = cg_w_size;
  solver->cg_res_size = cg_res_size;

  //--------------------------------------------------------------------
  //double_tmp在BCG中是6倍的空间, 用于存储BCG中的各种内积残差等
  solver->cg_double_tmp = (PASE_REAL*)malloc(6* step_size* sizeof(PASE_REAL));
  //int_tmp在BCG中是1倍的空间，用于存储unlock
  solver->cg_int_tmp    = (PASE_INT*)malloc(step_size * sizeof(PASE_INT));

  //给multigrid中需要的工作空间赋值
  solver->multigrid->sol           = solver->sol;
  solver->multigrid->rhs           = solver->rhs;
  solver->multigrid->cg_p          = solver->cg_p;
  solver->multigrid->cg_w          = solver->cg_w;
  solver->multigrid->cg_res        = solver->cg_res;
  solver->multigrid->cg_double_tmp = solver->cg_double_tmp;
  solver->multigrid->cg_int_tmp    = solver->cg_int_tmp;
  //--------------------------------------------------------------

  end = clock();
  solver->set_up_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 销毁 PASE_MG_SOLVER 并释放内存空间
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver)
{
  PASE_INT i;
  PASE_OPS_Free(&(solver->pase_ops)); 
  if(NULL != solver->eigenvalues) {
    free(solver->eigenvalues);
    solver->eigenvalues = NULL;
  }
  if(NULL != solver->abs_res_norm) {
    free(solver->abs_res_norm);
    solver->abs_res_norm = NULL;
  }
  //-----------------------------------------------------
  //释放向量工作空间
  for(i=0; i< solver->num_levels; i++) {
    solver->gcge_ops->MultiVecDestroy(&(solver->sol[i]),   
	  solver->sol_size[i], solver->gcge_ops);
    solver->gcge_ops->MultiVecDestroy(&(solver->rhs[i]), 
	  solver->rhs_size[i], solver->gcge_ops);
    solver->gcge_ops->MultiVecDestroy(&(solver->cg_p[i]), 
	  solver->cg_p_size[i], solver->gcge_ops);
    solver->gcge_ops->MultiVecDestroy(&(solver->cg_w[i]), 
	  solver->cg_w_size[i], solver->gcge_ops);
    solver->gcge_ops->MultiVecDestroy(&(solver->cg_res[i]), 
	  solver->cg_res_size[i], solver->gcge_ops);
  }
  free(solver->sol);     solver->sol    = NULL;
  free(solver->rhs);     solver->rhs    = NULL;
  free(solver->cg_p);    solver->cg_p   = NULL;
  free(solver->cg_w);    solver->cg_w   = NULL;
  free(solver->cg_res);  solver->cg_res = NULL;
  free(solver->sol_size);     solver->sol_size    = NULL;
  free(solver->rhs_size);     solver->rhs_size    = NULL;
  free(solver->cg_p_size);    solver->cg_p_size   = NULL;
  free(solver->cg_w_size);    solver->cg_w_size   = NULL;
  free(solver->cg_res_size);  solver->cg_res_size = NULL;

  //-----------------------------------------------------
  //释放 aux_u空间
  if(NULL != solver->aux_u) {
    PASE_MultiVector_destroy_sub(&(solver->aux_u));
  }
  //释放double,int型工作空间
  if(NULL != solver->cg_double_tmp) {
    free(solver->cg_double_tmp);
    solver->cg_double_tmp = NULL;
  }
  if(NULL != solver->cg_int_tmp) {
    free(solver->cg_int_tmp);
    solver->cg_int_tmp = NULL;
  }
  if(NULL != solver->aux_A) {
    PASE_Matrix_destroy_sub(&(solver->aux_A));
  }
  if(NULL != solver->aux_B) {
    PASE_Matrix_destroy_sub(&(solver->aux_B));
  }
  if(NULL != solver->multigrid) {
    PASE_MULTIGRID_Destroy(&(solver->multigrid));
  }
  free(solver);
  return 0;
}

/*
 * 1. 在 initial_level 层上GCGE直接求解获取初值
 * 2. for idx_level = initial_level-1 : 0
 *    1) prolong: sol[idx_level+1] --> sol[idx_level]
 *    2) for idx_cycle = 0 : max_cycle_count_each_level[idx_level]
 *       1) 如果是最细层，需要调整最粗层取哪个
 *       PASE_Mg_cycle
 *
 * PASE_Mg_cycle(solver, coarse_level, current_level):
 * 1. 前光滑 (solver, coarse_level, current_level
 * 2. 构造复合矩阵 (solver, coarse_level, current_level)
 * 3. 复合矩阵特征值求解
 * 4. prolong: (aux_u, coarse_level, current_level)
 * 5. 后光滑 (solver, coarse_level, current_level
 *
 */
PASE_INT
PASE_Mg_solve(PASE_MG_SOLVER solver)
{
  //一共要用到的层数
  PASE_INT num_levels = solver->num_levels;
  //V_h_1的层号
  PASE_INT initial_level = solver->initial_level;
  //V_H层号, 初始为实际最粗层，或由用户给出？?TODO
  PASE_INT coarest_level = solver->coarest_level;
  //细层层指标
  PASE_INT idx_level = 0;
  //要求解的特征值个数
  PASE_INT pase_nev = solver->pase_nev;
  //用户需要的特征值个数
  PASE_INT nev = solver->nev;
  //多重网格信息
  PASE_MULTIGRID multigrid = solver->multigrid;
  //二网格迭代次数指标
  PASE_INT idx_cycle = 0;
  PASE_INT coarse_level = 0;
  PASE_INT current_level = 0;
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  PASE_INT i = 0;

  if(solver->num_given_eigs < nev) {
    //V_h_1层直接求解
    PASE_Direct_solve(solver, initial_level);
  } 
  //GCGE_Printf("after initial direct:\n");
  //PASE_Mg_error_estimate(solver, initial_level);
  if(initial_level > 0)
  {
    coarse_level = initial_level;
    current_level = initial_level-1;
    mv_s[0] = 0;
    mv_e[0] = pase_nev;
    mv_s[1] = 0;
    mv_e[1] = pase_nev;
    PASE_MULTIGRID_FromItoJ(solver->multigrid, coarse_level, current_level, 
	  mv_s, mv_e, solver->sol[coarse_level], solver->sol[current_level]);
  }
  //GCGE_Printf("after initial prolong:\n");
  //PASE_Mg_error_estimate(solver, current_level);
  //各细层求解
  for(idx_level=initial_level-1; idx_level>-1; idx_level--) {
    //确定粗细层指标
    coarse_level = idx_level+1;
    current_level = idx_level;
    //对该层网格与VH层进行二网格迭代
    for(idx_cycle=0; idx_cycle < solver->max_cycle_count_each_level[current_level]; idx_cycle++)
    {
      //如果是最细层，需要调整最粗层取哪个
      if(current_level == 0) {
        //TODO 修改VH层, 自适应地确定
        coarest_level = num_levels-1;
      }//end for current_level==0
      PASE_Mg_cycle(solver, coarest_level, current_level);
      //最细层检查收敛性
      if(current_level == 0) {
        GCGE_Printf("idx_cycle: %d:\n", idx_cycle);
        PASE_Mg_error_estimate(solver, current_level);
        if(solver->nconv >= solver->nev) {
	  idx_cycle = solver->max_cycle_count_each_level[current_level] + 1;
	  idx_level = -1;
        }//end for if nconv>=nev
      }//end for if current_level==0
    }//end for idx_cycle
    if(current_level > 0)
    {
      //如果当前层不是最细层, 将解sol从上一粗层插值到当前细层
      mv_s[0] = 0;
      mv_e[0] = pase_nev;
      mv_s[1] = 0;
      mv_e[1] = pase_nev;
      PASE_MULTIGRID_FromItoJ(solver->multigrid, current_level, current_level-1, 
	  mv_s, mv_e, solver->sol[current_level], solver->sol[current_level-1]);
    }
  }//end for idx_level
  return 0;
}


/*
 * PASE_Mg_cycle(solver, coarse_level, current_level):
 * 1. 前光滑 (solver, coarse_level, current_level
 * 2. 构造复合矩阵 (solver, coarse_level, current_level)
 * 2. 构造复合向量 (solver, coarse_level)
 * 3. 复合矩阵特征值求解
 * 4. prolong: (aux_u, coarse_level, current_level)
 * 5. 后光滑 (solver, coarse_level, current_level
 *
 */
PASE_INT 
PASE_Mg_cycle(PASE_MG_SOLVER solver, PASE_INT coarse_level, PASE_INT current_level)
{
  //前光滑
  PASE_INT max_presmooth_iter = solver->max_pre_count_each_level[current_level];
  PASE_Mg_smoothing(solver, current_level, max_presmooth_iter);

  //GCGE_Printf("after presmoothing:\n");
  //PASE_Mg_error_estimate(solver, current_level);

  //构造复合矩阵 (solver, coarse_level, current_level)
  PASE_Mg_set_pase_aux_matrix(solver, coarse_level, current_level);
  //构造复合向量 
  PASE_Mg_set_pase_aux_vector(solver, coarse_level);
  //直接求解复合矩阵特征值问题
  PASE_Aux_direct_solve(solver, coarse_level);

  //把aux向量转换成细空间上的向量
  PASE_Mg_prolong_from_pase_aux_vector(solver, coarse_level, current_level);

  //后光滑
  //PASE_INT max_postsmooth_iter = solver->max_post_count_each_level[current_level];
  //PASE_Mg_smoothing(solver, current_level, max_postsmooth_iter);

  return 0;
}

PASE_INT
PASE_Mg_set_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT coarse_level, 
      PASE_INT current_level)
{
  //构造复合矩阵，细空间的维数, 从nconv:pase_nev-1
  PASE_INT pase_nev = solver->pase_nev;
  PASE_INT nconv = solver->nconv;

  //创建复合矩阵aux_A, aux_B, 结构
  PASE_Mg_pase_aux_matrix_create(solver, coarse_level);

  //给复合矩阵的aux_Hh部分与aux_hh部分赋值
  PASE_INT error = 0;
  //细网格矩阵A, B
  void *A = solver->multigrid->A_array[current_level];
  void *B = solver->multigrid->B_array[current_level];
  //复合矩阵 aux_A, aux_B
  PASE_Matrix aux_A = solver->aux_A;
  PASE_Matrix aux_B = solver->aux_B;
  //细网格当前的近似特征向量
  void **fine_sol = solver->sol[current_level];
  //给复合矩阵的aux部分赋值
  error = PASE_Aux_matrix_set_by_pase_matrix(aux_A, A, fine_sol, solver, 
	coarse_level, current_level);
  error = PASE_Aux_matrix_set_by_pase_matrix(aux_B, B, fine_sol, solver, 
	coarse_level, current_level);
  return 0;
}

//在某一层上(不是复合矩阵)调用GCGE直接求解
PASE_INT
PASE_Direct_solve(PASE_MG_SOLVER solver, PASE_INT idx_level)
{
  clock_t start, end;
  start = clock();

  void        *A            = solver->multigrid->A_array[idx_level];
  void        *B            = solver->multigrid->B_array[idx_level];
  PASE_SCALAR *eigenvalues  = solver->eigenvalues;
  void       **eigenvectors = solver->sol[idx_level];
  GCGE_SOLVER *gcge_solver = GCGE_SOLVER_CreateByOps(A, B,
          solver->pase_nev, eigenvalues, eigenvectors, solver->gcge_ops);
  gcge_solver->para->print_para = 0;
  gcge_solver->para->print_eval = 0;
  gcge_solver->para->print_conv = 0;
  gcge_solver->para->print_result = 0;
  gcge_solver->para->ev_tol = solver->rtol;
  //GCGE最大迭代次数，由solver->max_direct_count_each_level[idx_level]给出
  gcge_solver->para->ev_max_it = solver->max_initial_count;
  //求解
  GCGE_SOLVER_Solve(gcge_solver);  
  //释放空间, 不释放gcge_solver->ops
  GCGE_SOLVER_Free_Some(&gcge_solver);
  //每层上最多直接求解时GCGE迭代的次数
  end = clock();
  solver->set_aux_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

//创建辅助矩阵空间，但并不分配具体的大规模空间
PASE_INT 
PASE_Matrix_create_sub(PASE_Matrix *aux_A, PASE_INT n)
{
  *aux_A = (PASE_Matrix)malloc(sizeof(pase_Matrix));
  (*aux_A)->is_diag = 0;
  (*aux_A)->aux_hh = (PASE_SCALAR*)calloc(n*n, sizeof(PASE_SCALAR));
  (*aux_A)->num_aux_vec = n;
  return 0;
}

//释放辅助矩阵空间，但并不处理具体的大规模空间
PASE_INT 
PASE_Matrix_destroy_sub(PASE_Matrix *aux_A)
{
  free((*aux_A)->aux_hh);
  (*aux_A)->aux_hh = NULL;
  free(*aux_A);
  *aux_A = NULL;
  return 0;
}

//创建复合矩阵空间，大规模空间不申请，直接用solver中的工作空间
PASE_INT
PASE_Mg_pase_aux_matrix_create(PASE_MG_SOLVER solver, PASE_INT idx_level)
{
  if(solver->aux_A == NULL) {
    PASE_Matrix_create_sub(&(solver->aux_A), solver->pase_nev);
  }
  if(solver->aux_B == NULL) {
    PASE_Matrix_create_sub(&(solver->aux_B), solver->pase_nev);
  }
  PASE_Matrix aux_A = solver->aux_A;
  PASE_Matrix aux_B = solver->aux_B;
  aux_A->A_H        = solver->multigrid->A_array[idx_level];
  aux_A->aux_Hh     = solver->rhs[idx_level];
  aux_B->A_H        = solver->multigrid->B_array[idx_level];
  aux_B->aux_Hh     = solver->cg_p[idx_level];
  aux_A->num_aux_vec = solver->pase_nev - solver->nconv;
  aux_B->num_aux_vec = solver->pase_nev - solver->nconv;

  return 0;
}

PASE_INT 
PASE_Aux_matrix_set_by_pase_matrix(PASE_Matrix aux_A, void *A_h, void **sol, 
        PASE_MG_SOLVER solver, PASE_INT coarse_level, PASE_INT current_level)
{
  //matlab符号，包括前面，不包括后面
  //计算 aux_A->aux_Hh(:, nconv:pase_nev) 
  //   = I_h^H * A_h * u_h(:, nconv:pase_nev) 
  //计算 aux_A->aux_hh(:, nconv:pase_nev) 
  //   = u_h * A_h * u_h(:, nconv:pase_nev) 

  //计算 aux_A->aux_Hh(:, nconv:pase_nev) 
  //   = I_h^H * A_h * u_h(:, nconv:pase_nev) 
  GCGE_OPS *gcge_ops = solver->gcge_ops;
  PASE_OPS *pase_ops = solver->pase_ops;
  PASE_INT nconv = solver->nconv;
  PASE_INT pase_nev = solver->pase_nev;
  PASE_INT num_aux_vec = pase_nev-nconv;
  aux_A->num_aux_vec = num_aux_vec;
  //sol, rhs, cg_p已经用了，这里使用cg_w作为临时空间
  void     **Ah_u = solver->cg_w[current_level];
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  mv_s[0]  = nconv;
  mv_e[0]  = pase_nev;
  mv_s[1]  = 0;
  mv_e[1]  = num_aux_vec;
  // Ah_u(:,0:pase_nev-nconv) = A_h * sol(:, nconv:pase_nev) 
  gcge_ops->MatDotMultiVec(A_h, sol, Ah_u, mv_s, mv_e, gcge_ops);
  //从current_level限制到coarse_level
  mv_s[0] = 0;
  mv_e[0] = num_aux_vec;
  mv_s[1] = 0;
  mv_e[1] = num_aux_vec;
  PASE_INT error = PASE_MULTIGRID_FromItoJ(solver->multigrid, 
          current_level, coarse_level, mv_s, mv_e, Ah_u, aux_A->aux_Hh);
  //计算 aux_A->aux_hh = sol(:, nconv:pase_nev) * A_h * sol(:, nconv:pase_nev) 
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = 0;
  mv_e[1] = num_aux_vec;
  PASE_REAL *A_aux_hh = aux_A->aux_hh;
  gcge_ops->MultiVecInnerProd(sol, Ah_u, A_aux_hh, 
          "nonsym", mv_s, mv_e, num_aux_vec, 0, solver->gcge_ops);

  return 0;
}


PASE_INT
PASE_Mg_set_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT idx_level)
{
  clock_t start, end;
  start = clock();

  PASE_Mg_pase_aux_vector_create(solver, idx_level);

  //子空间复合矩阵特征值问题的初值[0,e_i]^T
  PASE_INT num_aux_vec = solver->aux_u->num_aux_vec;
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  mv_s[0] = 0;
  mv_e[0] = num_aux_vec;
  mv_s[1] = 0;
  mv_e[1] = num_aux_vec;
  //TODO 这里发现了bug, (void**), 不知道之前的问题是不是因为这里不对
  solver->pase_ops->MultiVecAxpby(0.0, (void**)(solver->aux_u),
          0.0, (void**)(solver->aux_u), mv_s, mv_e, solver->pase_ops);
  PASE_REAL *aux_h = solver->aux_u->aux_h;
  PASE_INT i = 0;
  for(i=0; i<num_aux_vec; i++) {
      aux_h[i*num_aux_vec+i] = 1.0;
  }
  end = clock();
  solver->set_aux_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

//创建复合向量空间，大规模空间不申请，直接用solver中的工作空间
PASE_INT 
PASE_Mg_pase_aux_vector_create(PASE_MG_SOLVER solver, PASE_INT idx_level)
{
  if(solver->aux_u == NULL) {
    PASE_MultiVector_create_sub(&(solver->aux_u), solver->max_nev);
  }
  PASE_INT num_aux_vec = solver->pase_nev - solver->nconv;
  solver->aux_u->num_aux_vec = num_aux_vec;
  solver->aux_u->num_vec = num_aux_vec;
  solver->aux_u->b_H = solver->sol[idx_level];
  return 0;
}

//创建辅助矩阵空间，但并不分配具体的大规模空间
PASE_INT
PASE_MultiVector_create_sub(PASE_MultiVector *aux_u, PASE_INT n)
{
  *aux_u = (PASE_MultiVector)malloc(sizeof(pase_MultiVector));
  (*aux_u)->num_aux_vec = n;
  (*aux_u)->num_vec = n;
  (*aux_u)->aux_h = (PASE_SCALAR*)calloc(n*n, sizeof(PASE_SCALAR));
  return 0;
}

//释放辅助矩阵空间，但并不处理具体的大规模空间
PASE_INT
PASE_MultiVector_destroy_sub(PASE_MultiVector *aux_u)
{
  free((*aux_u)->aux_h);
  (*aux_u)->aux_h = NULL;
  free(*aux_u);
  *aux_u = NULL;
  return 0;
}

PASE_INT
PASE_Aux_direct_solve(PASE_MG_SOLVER solver, PASE_INT coarse_level)
{
  clock_t start, end;
  start = clock();

  GCGE_SOLVER *gcge_pase_solver = GCGE_SOLVER_PASE_Create(
	solver->aux_A, solver->aux_B, solver->pase_nev, 
	solver->eigenvalues, solver->aux_u, solver->pase_ops);
  //gcg直接求解的精度比atol稍高一些
  gcge_pase_solver->para->ev_tol    = solver->atol * 1e-2;
  gcge_pase_solver->para->ev_max_it = solver->max_direct_count_each_level[coarse_level];
  //gcge_pase_solver->para->cg_max_it = 50;
  gcge_pase_solver->para->print_para = 0;
  gcge_pase_solver->para->print_conv = 0;
  gcge_pase_solver->para->print_eval = 0;
  gcge_pase_solver->para->print_result = 0;
  gcge_pase_solver->para->given_init_evec = 1;
  //gcge_pase_solver->para->orth_para->orth_zero_tol = 1e-13;
  //gcge_pase_solver->para->p_orth_type = "gs";
  //精度高时，multi正交化达不到要求
  gcge_pase_solver->para->w_orth_type = "gs";
  //gcge_pase_solver->para->num_unlock = pase_nev - solver->nconv;
  //求解
  GCGE_SOLVER_Solve(gcge_pase_solver);  
  //释放空间
  GCGE_SOLVER_Free(&gcge_pase_solver);

  end = clock();
  solver->set_aux_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

PASE_INT
PASE_Mg_prolong_from_pase_aux_vector(PASE_MG_SOLVER solver,
      PASE_INT coarse_level, PASE_INT current_level)
{
  clock_t start, end;
  start = clock();

  PASE_MultiVector aux_u = solver->aux_u;
  //rhs此时用来存储P*u_H
  void **P_uH = solver->rhs[current_level];
  void **sol = solver->sol[current_level];
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  PASE_INT nconv = solver->nconv;
  PASE_INT pase_nev = solver->pase_nev;
  PASE_INT num_aux_vec = pase_nev - nconv;
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = 0;
  mv_e[1] = num_aux_vec;
  //从aux_u_i->b_H延拓到u_j(先放在u_tmp)
  //同时aux_u_i->aux_h作为线性组合系数，对原u_j进行线性组合
  PASE_INT error = PASE_MULTIGRID_FromItoJ(solver->multigrid, 
        coarse_level, current_level, mv_s, mv_e, aux_u->b_H, P_uH);
  //线性组合时，alpha=beta=1.0
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = 0;
  mv_e[1] = num_aux_vec;
  solver->gcge_ops->MultiVecLinearComb(sol, P_uH, mv_s, mv_e,
        aux_u->aux_h+nconv*num_aux_vec, num_aux_vec, 0, 1.0, 1.0, solver->gcge_ops);
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = 0;
  mv_e[1] = num_aux_vec;
  solver->gcge_ops->MultiVecSwap(sol, P_uH, mv_s, mv_e, solver->gcge_ops);
  end = clock();
  solver->prolong_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

PASE_INT
PASE_Mg_smoothing(PASE_MG_SOLVER solver, PASE_INT current_level, PASE_INT max_iter)
{
  clock_t start, end;
  start = clock();

  void           *A             = solver->multigrid->A_array[current_level];
  void           *B             = solver->multigrid->B_array[current_level];
  void          **sol           = solver->sol[current_level];
  PASE_SCALAR    *eigenvalues   = solver->eigenvalues;
  void          **rhs           = solver->rhs[current_level];
  PASE_INT        pase_nev      = solver->pase_nev;
  PASE_INT        nconv         = solver->nconv;

  GCGE_OPS *gcge_ops = solver->gcge_ops;
  PASE_OPS *pase_ops = solver->pase_ops;
  PASE_INT i = 0;
 
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  //求解 A * u = eig * B * u
  //计算 rhs = eig * B * u
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = 0;
  mv_e[1] = pase_nev-nconv;
  gcge_ops->MatDotMultiVec(B, sol, rhs, mv_s, mv_e, gcge_ops);
  for(i=nconv; i<pase_nev; i++) {
      gcge_ops->MultiVecAxpbyColumn(0.0, rhs, i-nconv, eigenvalues[i], 
              rhs, i-nconv, gcge_ops);
  }

  PASE_REAL tol = solver->atol;
  PASE_REAL cg_rate = 1e-2;
  PASE_INT  max_coarest_nsmooth = 10*max_iter;
  mv_s[0] = 0;
  mv_e[0] = pase_nev-nconv;
  mv_s[1] = nconv;
  mv_e[1] = pase_nev;

  //TODO 分批求解
  PASE_BMG(solver->multigrid, current_level, rhs, sol, mv_s, mv_e, 
	tol, cg_rate, max_iter, max_coarest_nsmooth);
  end = clock();
  solver->smooth_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 完成一次 PASE_Mg_cycle 后, 需计算残差及已收敛特征对个数. 已收敛特征对在之后的迭代中，不再计算和更改. 
 *
 * @param solver    输入/输出参数
 * @param idx_level 输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_error_estimate(PASE_MG_SOLVER solver, PASE_INT idx_level)
{
  PASE_INT         pase_nev   	 = solver->pase_nev; 
  PASE_INT         nconv         = solver->nconv;
  PASE_REAL        atol          = solver->atol;
  PASE_REAL        rtol          = solver->rtol;
  void           **sol           = solver->sol[idx_level];
  PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
  void            *A	         = solver->multigrid->A_array[idx_level];
  void            *B	         = solver->multigrid->B_array[idx_level];

  /* 计算最细层的残差：r = Au - kMu */
  PASE_INT         flag          = 0;
  PASE_REAL       *check_multi   = (PASE_REAL*)PASE_Malloc((pase_nev-1)*sizeof(PASE_REAL));
  PASE_INT         i		 = 0;
  PASE_REAL        r_norm        = 1e+5;
  PASE_REAL        u_norm        = 0.0;
  PASE_REAL        xTAx          = 0.0;
  PASE_REAL        xTBx          = 0.0;
  void            *Au;
  void            *Bu;
  void            *ui;

  GCGE_OPS *gcge_ops = solver->gcge_ops;
  for(i = nconv; i < pase_nev; ++i) {
    gcge_ops->GetVecFromMultiVec(solver->rhs[idx_level], 0, &Bu, gcge_ops);
    gcge_ops->GetVecFromMultiVec(sol, i, &ui, gcge_ops);
    //Bu = B * sol[i]
    gcge_ops->MatDotVec(B, ui, Bu, gcge_ops);
    gcge_ops->VecInnerProd(ui, ui, &u_norm, gcge_ops);
    //u_norm = || sol[i] ||_2
    u_norm = sqrt(u_norm);
    gcge_ops->RestoreVecForMultiVec(solver->rhs[idx_level], 0, &Bu, gcge_ops);
    //计算 || Au - \lambda Bu ||_2
    gcge_ops->GetVecFromMultiVec(solver->rhs[idx_level], 1, &Au, gcge_ops);
    //Au = A * sol[i]
    gcge_ops->MatDotVec(A, ui, Au, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(sol, i, &ui, gcge_ops);
    gcge_ops->GetVecFromMultiVec(solver->rhs[idx_level], 0, &Bu, gcge_ops);
    //Bu = Au - eval[i] * Bu
    gcge_ops->VecAxpby(1.0, Au, -eigenvalues[i], Bu, gcge_ops);
    //r_norm = || Bu ||_2
    gcge_ops->VecInnerProd(Bu, Bu, &r_norm, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(solver->rhs[idx_level], 0, &Bu, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(solver->rhs[idx_level], 1, &Au, gcge_ops);
    //r_norm = || Au - \lambda Bu ||_2 / || ui ||_2
    r_norm = sqrt(r_norm)/u_norm;
    solver->abs_res_norm[i] = r_norm;
    GCGE_Printf("i: %d, eval: %18.15e, r_norm: %18.15e\n", i, eigenvalues[i], r_norm);

    if(i+1 < pase_nev) {
      check_multi[i] = fabs((eigenvalues[i]-eigenvalues[i+1])/eigenvalues[i]);
    }
    if(r_norm < atol || (r_norm/eigenvalues[i]) < rtol) {
      if(0 == flag) {
        //solver->nconv++;
	solver->nconv = i+1;
      }
    } else {
      /* break; */
      flag = 1;
    }
  }
  /*
  //检查第一个为收敛的特征值与最后一个刚收敛的特征值是否有可能是重特征值，为保证之后的排序问题，需让重特征值同时在收敛的集合或未收敛的集合.
  while(solver->nconv > nconv && solver->nconv < pase_nev && check_multi[solver->nconv-1] < 1e-8) {
    solver->nconv--;
  }
  */
  free(check_multi); check_multi = NULL;

  if(solver->print_level > 0) {
    //PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);	
    GCGE_Printf("nconv = %d, ", solver->nconv);
    if(solver->nconv < solver->pase_nev) {
      GCGE_Printf("the first unconverged eigenvalues (residual) = %.8e (%1.6e)\n", solver->eigenvalues[solver->nconv], solver->abs_res_norm[solver->nconv]);
    } else {
      GCGE_Printf("all the wanted eigenpairs have converged.\n");
    }
  }	

  return 0;
}
