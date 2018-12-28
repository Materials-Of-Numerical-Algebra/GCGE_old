#include "pase_solver.h"

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
//对二网格的测试，main中给定粗网格矩阵Ac,Bc, 限制与延拓矩阵P,R
PASE_INT
PASE_EigenSolver(void *A, void *B, PASE_SCALAR *eval, void **evec, 
        PASE_INT block_size, PASE_PARAMETER param, GCGE_OPS *gcge_ops,
        PASE_OPS *pase_ops)
{
  PASE_INT i              = 0;
  //目前并没有用到 TODO 看洪怎么用的
  PASE_INT max_block_size = ((2*block_size)<(block_size+5))?(2*block_size):(block_size+5);
  //创建solver空间并赋初值NULL
  PASE_MG_SOLVER solver   = PASE_Mg_solver_create(param);

  //设置参数
  //迭代中使用二网格or多重网格, cycle_type==0,二网格
  PASE_Mg_set_cycle_type(solver, param->cycle_type);
  //nev
  PASE_Mg_set_block_size(solver, block_size);
  //允许多算几个(最多多算5个)
  PASE_Mg_set_max_block_size(solver, max_block_size);
  //二网格最大迭代次数
  PASE_Mg_set_max_cycle(solver, param->max_cycle);
  //前光滑迭代次数
  PASE_Mg_set_max_pre_iteration(solver, param->max_pre_iter);
  //后光滑迭代次数
  PASE_Mg_set_max_post_iteration(solver, param->max_post_iter);
  //绝对收敛准则,相对收敛准则
  //两个收敛准则只需满足其一
  //TODO 目前测试阶段用的绝对收敛准则||Ax-\lambdaBx||_2/||x||_2
  PASE_Mg_set_atol(solver, param->atol);
  PASE_Mg_set_rtol(solver, param->rtol);
  //print_level==0时什么都不打印，1时打印收敛信息，2时打印运行信息
  PASE_Mg_set_print_level(solver, param->print_level);

  //用gcge_ops创建pase_ops, TODO 测试时为了使用main中提供的PrintMat，由main提供pase_ops
  //PASE_OPS *pase_ops;
  //PASE_OPS_Create(&pase_ops, gcge_ops);
  solver->pase_ops = pase_ops;
  solver->gcge_ops = gcge_ops;

  //进行AMG分层，分配工作空间
  PASE_Mg_set_up(solver, A, B, eval, evec, param);
  PASE_Mg_solve(solver);
  PASE_Mg_solver_destroy(solver);
  //PASE_OPS_Free(&pase_ops); 
  return 0;
}


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
PASE_Mg_solver_create(PASE_PARAMETER param)
{
  PASE_MG_SOLVER solver      = (PASE_MG_SOLVER)PASE_Malloc(sizeof(PASE_MG_SOLVER_PRIVATE));
  solver->function           = PASE_Mg_function_create(
                               PASE_Mg_get_initial_vector_by_full_multigrid_hypre, 
                               //NULL,
                               PASE_Mg_direct_solve_by_gcg, 
                               PASE_Mg_presmoothing_by_amg_hypre, 
                               PASE_Mg_postsmoothing_by_amg_hypre, 
                               NULL,
                               NULL);
                               //PASE_Mg_presmoothing_by_cg_aux,
                               //PASE_Mg_postsmoothing_by_cg_aux);
  solver->cycle_type                = 0;

  solver->idx_cycle_level           = NULL;
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

#if 0
  solver->time_inner                = 0.0;
  solver->time_lapack               = 0.0;
  solver->time_other                = 0.0;
  solver->time_diag_pre             = 0.0;
  solver->time_linear_diag          = 0.0;
  solver->time_orth_gcg             = 0.0;
#endif

  solver->exact_eigenvalues         = NULL;
  solver->eigenvalues               = NULL;
  solver->u                         = NULL;
  solver->is_u_owner                = PASE_YES;
  solver->aux_u                     = NULL;

  //---------------------------------------
  //临时空间
  solver->u_tmp                     = NULL;
  solver->u_tmp_1                   = NULL;
  solver->u_tmp_2                   = NULL;
  solver->u_tmp_3                   = NULL;
  //---------------------------------------

  solver->method_init               = NULL;
  solver->method_pre                = NULL;
  solver->method_post               = NULL;
  solver->method_pre_aux            = NULL;
  solver->method_post_aux           = NULL;
  solver->method_dire               = NULL;
  
  solver->gcge_ops                  = NULL;
  solver->pase_ops                  = NULL;
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
 * @brief PASE_MG_SOLVER 的准备阶段
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver, void *A, void *B, PASE_SCALAR *eval, void **x, PASE_PARAMETER param)
{
  clock_t start, end;
  start = clock();
  PASE_INT i = 0;
  /* TODO 进行AMG分层, 应该要返回已经分层好的max_level层矩阵, 目前只分配了空间，所以这里给各层矩阵赋值 */
  i = PASE_MULTIGRID_Create(&(solver->multigrid), param->max_level, A, B, solver->gcge_ops, solver->pase_ops);

  //把用户(main)中给的eval,evec赋给solver,Solve后再赋值返回给main
  solver->given_eval = eval;
  solver->given_evec = x;

  /* TODO multigrid中可能需要添加这样一个参数 */
  //solver->nleve = solver->multigrid->actual_level;
  //--------------------------------------------------------------------
  //--特征对工作空间-----------------------------------------------------
  if(NULL == solver->eigenvalues) {
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->max_block_size*sizeof(PASE_SCALAR));
    memcpy(solver->eigenvalues, eval, solver->block_size*sizeof(PASE_SCALAR));
  }

  PASE_INT mv_s[2];
  PASE_INT mv_e[2];

  //把初值x赋给solver->u
  solver->gcge_ops->MultiVecCreateByMat(&(solver->u), solver->max_block_size, A, solver->gcge_ops);
  mv_s[0] = 0;
  mv_e[0] = solver->block_size;
  mv_s[1] = 0;
  mv_e[1] = solver->block_size;
  solver->gcge_ops->MultiVecSwap(solver->u, x, mv_s, mv_e, solver->gcge_ops);
  //--------------------------------------------------------------------
  //--普通向量组空间-----------------------------------------------------
  /* TODO 给u_tmp也分配了max_block_size 大小的向量组空间, 是否需要这么大 */
  //u_tmp用作 rhs
  solver->u_tmp = (void***)malloc(param->max_level * sizeof(void**));
  //u_tmp_1用作 pase_u_h
  solver->u_tmp_1 = (void***)malloc(param->max_level * sizeof(void**));
  //u_tmp_2用作 BCG的临时空间
  solver->u_tmp_2 = (void***)malloc(param->max_level * sizeof(void**));
  //u_tmp_3用作 BCG的临时空间
  solver->u_tmp_3 = (void***)malloc(param->max_level * sizeof(void**));
  for(i=0; i< param->max_level; i++) {
      solver->gcge_ops->MultiVecCreateByMat(&(solver->u_tmp[i]),   solver->max_block_size, solver->multigrid->A_array[i], solver->gcge_ops);
      solver->gcge_ops->MultiVecCreateByMat(&(solver->u_tmp_1[i]), solver->max_block_size, solver->multigrid->A_array[i], solver->gcge_ops);
      solver->gcge_ops->MultiVecCreateByMat(&(solver->u_tmp_2[i]), solver->max_block_size, solver->multigrid->A_array[i], solver->gcge_ops);
      solver->gcge_ops->MultiVecCreateByMat(&(solver->u_tmp_3[i]), solver->max_block_size, solver->multigrid->A_array[i], solver->gcge_ops);
  }
  solver->multigrid->u_tmp = solver->u_tmp_3;
  //--------------------------------------------------------------------
  //--double,int型工作空间-------------------------------------------------
  solver->double_tmp = (PASE_REAL*)PASE_Malloc(6* solver->block_size * sizeof(PASE_REAL));
  solver->int_tmp = (PASE_INT*)PASE_Malloc(solver->block_size * sizeof(PASE_INT));
  //--------------------------------------------------------------

  if(0 == solver->cycle_type) {
    //二网格，0(最细层)与nleve-1(最粗层)
    //idx_cycle_level存储具体用第几层矩阵
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

  //复合多向量组空间
  //并没有分配具体的向量组空间,第一次用到的时候才会分配空间
  solver->aux_u = NULL;

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
  PASE_INT i, j;
  if(NULL != solver) {
    if(NULL != solver->idx_cycle_level) {
      PASE_Free(solver->idx_cycle_level);
    }
    //-----------------------------------------------------
    //释放 aux_u空间
    if(NULL != solver->aux_u) {
        solver->pase_ops->MultiVecDestroy((void***)(&(solver->aux_u)), solver->block_size, solver->pase_ops);
    }
    //-----------------------------------------------------
    //释放特征值空间
    if(NULL != solver->eigenvalues) {
      PASE_Free(solver->eigenvalues);
    }
    //-----------------------------------------------------
    //释放残差空间
    if(NULL != solver->r_norm) {
      PASE_Free(solver->r_norm);
    }
    //-----------------------------------------------------
    //释放临时特征向量空间
    if(NULL != solver->u) {
      if(PASE_YES == solver->is_u_owner) {
          solver->gcge_ops->MultiVecDestroy(&(solver->u), solver->max_block_size, solver->gcge_ops);
      }
    }
    //-----------------------------------------------------
    //释放向量工作空间
    for(i=0; i< solver->multigrid->num_levels; i++) {
      solver->gcge_ops->MultiVecDestroy(&(solver->u_tmp[i]),   solver->max_block_size, solver->gcge_ops);
      solver->gcge_ops->MultiVecDestroy(&(solver->u_tmp_1[i]), solver->max_block_size, solver->gcge_ops);
      solver->gcge_ops->MultiVecDestroy(&(solver->u_tmp_2[i]), solver->max_block_size, solver->gcge_ops);
      solver->gcge_ops->MultiVecDestroy(&(solver->u_tmp_3[i]), solver->max_block_size, solver->gcge_ops);
    }
    free(solver->u_tmp);    solver->u_tmp   = NULL;
    free(solver->u_tmp_1);  solver->u_tmp_1 = NULL;
    free(solver->u_tmp_2);  solver->u_tmp_2 = NULL;
    free(solver->u_tmp_3);  solver->u_tmp_3 = NULL;
    //-----------------------------------------------------
    //释放double,int型工作空间
    if(NULL != solver->double_tmp) {
      PASE_Free(solver->double_tmp);
    }
    if(NULL != solver->int_tmp) {
      PASE_Free(solver->int_tmp);
    }
    if(NULL != solver->function) {
      PASE_Free(solver->function);
    }
    if(NULL != solver->multigrid) {
      PASE_MULTIGRID_Destroy(&(solver->multigrid));
    }
    PASE_Free(solver);
  }
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
  /* TODO 暂时先不考虑多重网格获取初值 */
  PASE_Mg_get_initial_vector(solver);
  //初始残差
  GCGE_Printf("init residual: \n");
  PASE_Mg_error_estimate(solver);
  //PASE_Mg_print(solver);
  do {
    solver->ncycl++;
    //二网格迭代
    PASE_Mg_cycle(solver);
    //一次二网格迭代后计算残差
    PASE_Mg_error_estimate(solver);
  } while(solver->max_cycle > solver->ncycl && solver->nconv < solver->actual_block_size);
  end = clock();
  solver->total_solve_time += ((double)(end-start))/CLK_TCK;
  solver->total_time        = solver->total_solve_time + solver->set_up_time;
  PASE_Mg_print(solver);

  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  mv_s[0] = 0;
  mv_e[0] = solver->block_size;
  mv_s[1] = 0;
  mv_e[1] = solver->block_size;
  //把solver->u赋值给given_evec
  solver->gcge_ops->MultiVecSwap(solver->given_evec, solver->u, mv_s, mv_e, solver->gcge_ops);
  memcpy(solver->given_eval, solver->eigenvalues, solver->block_size * sizeof(PASE_SCALAR));

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
  void           **u0	         = solver->u;
  PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
  void            *A0	         = solver->multigrid->A_array[0];
  void            *B0	         = solver->multigrid->B_array[0];

  /* 计算最细层的残差：r = Au - kMu */
  PASE_INT         flag          = 0;
  PASE_REAL       *check_multi   = (PASE_REAL*)PASE_Malloc((block_size-1)*sizeof(PASE_REAL));
  PASE_INT         i		 = 0;
  PASE_REAL        r_norm        = 1e+5;
  PASE_REAL        u_Bnorm       = 0.0;
  PASE_REAL        xTAx          = 0.0;
  PASE_REAL        xTBx          = 0.0;
  void            *Au;
  void            *Bu;
  void            *ui;
  solver->nlock                  = nconv;

  if(NULL == solver->r_norm) {
    solver->r_norm = (PASE_REAL*)PASE_Malloc(block_size*sizeof(PASE_REAL));
  }

  //没用
  //solver->gcge_ops->SetDirichletBoundary(u0, block_size, A0, B0);

  nconv = 0;
  GCGE_OPS *gcge_ops = solver->gcge_ops;
  for(i = nconv; i < block_size; ++i) {

    //计算 u_Bnorm = || ui ||_B
    gcge_ops->GetVecFromMultiVec(solver->u_tmp[0], 0, &Bu, gcge_ops);
    gcge_ops->GetVecFromMultiVec(u0, i, &ui, gcge_ops);
    gcge_ops->MatDotVec(B0, ui, Bu, gcge_ops);
    gcge_ops->VecInnerProd(ui, Bu, &xTBx, gcge_ops);
    gcge_ops->VecInnerProd(ui, ui, &u_Bnorm, gcge_ops);
    //u_Bnorm = || ui ||_2
    u_Bnorm = sqrt(u_Bnorm);
    gcge_ops->RestoreVecForMultiVec(solver->u_tmp[0], 0, &Bu, gcge_ops);
    //计算 || Au - \lambda Bu ||_2
    gcge_ops->GetVecFromMultiVec(solver->u_tmp[0], 1, &Au, gcge_ops);
    gcge_ops->MatDotVec(A0, ui, Au, gcge_ops);
    gcge_ops->VecInnerProd(ui, Au, &xTAx, gcge_ops);
    //计算Rayleigh商
    eigenvalues[i] = xTAx/xTBx;
    gcge_ops->RestoreVecForMultiVec(u0, i, &ui, gcge_ops);
    gcge_ops->GetVecFromMultiVec(solver->u_tmp[0], 0, &Bu, gcge_ops);
    gcge_ops->VecAxpby(1.0, Au, -eigenvalues[i], Bu, gcge_ops);
    gcge_ops->VecInnerProd(Bu, Bu, &r_norm, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(solver->u_tmp[0], 0, &Bu, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(solver->u_tmp[0], 1, &Au, gcge_ops);
    //r_norm = || Au - \lambda Bu ||_2 / || ui ||_2
    r_norm = sqrt(r_norm)/u_Bnorm;
    solver->r_norm[i] = r_norm;
    GCGE_Printf("i: %d, eval: %18.15e, r_norm: %18.15e\n", i, eigenvalues[i], r_norm);

    if(i+1 < block_size) {
      check_multi[i] = fabs((eigenvalues[i]-eigenvalues[i+1])/eigenvalues[i]);
    }
    if(r_norm < atol || (r_norm/eigenvalues[i]) < rtol) {
      if(0 == flag) {
        //solver->nconv++;
	solver->nconv = i;
      }
    } else {
      /* TODO break; */
      flag = 1;
    }
  }
  //检查第一个为收敛的特征值与最后一个刚收敛的特征值是否有可能是重特征值，为保证之后的排序问题，需让重特征值同时在收敛的集合或未收敛的集合.
  while(solver->nconv > nconv && solver->nconv < block_size && check_multi[solver->nconv-1] < 1e-8) {
    solver->nconv--;
  }
  PASE_Free(check_multi);

  if(solver->print_level > 0) {
    //PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);	
    GCGE_Printf("cycle = %d, nconv = %d, ", solver->ncycl, solver->nconv);
    if(solver->nconv < solver->block_size) {
      GCGE_Printf("the first unconverged eigenvalues (residual) = %.8e (%1.6e)\n", solver->eigenvalues[solver->nconv], solver->r_norm[solver->nconv]);
    } else {
      GCGE_Printf("all the wanted eigenpairs have converged.\n");
    }
  }	

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

  //pase_mg_set_up时，将cur_cycle_level设为0，max_cycle_level设为1，
  //因此这里的递归只会调用一次
  if(cur_cycle_level < max_cycle_level)
  {
    //前光滑
    PASE_Mg_presmoothing(solver);
    //PASE_Mg_error_estimate(solver);

    //-------------------------------------
    //二网格中，涉及到aux的部分, 可以按照新结构重写
    solver->cur_cycle_level++;
    //设辅助空间，此时cur_cycle_level==1
    PASE_Mg_set_aux_space(solver);
    //直接求解
    PASE_Mg_cycle(solver);
    PASE_INT block_size = solver->block_size;
    //延拓
    PASE_Mg_prolong_from_pase_aux_vector(solver);
    //PASE_Mg_error_estimate(solver);
    solver->cur_cycle_level--;
    //二网格中，涉及到aux的部分
    //-------------------------------------

    //后光滑
    PASE_Mg_postsmoothing(solver);
    //PASE_Mg_error_estimate(solver);
  }
  else if( cur_cycle_level == max_cycle_level)
  {
    PASE_Mg_direct_solve(solver);
  }
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
      GCGE_Printf("\nPresmoothing\t");
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
#if 0
    if(solver->cur_cycle_level == 0) {
      solver->function->postsmoothing(solver);
      if(1 == solver->cycle_type) {
        PASE_Mg_orthogonalize(solver);
      }
    } else {
      solver->function->postsmoothing_aux(solver);
    }
#endif
    //目前只考虑二网格的情况
    solver->function->postsmoothing(solver);

    end = clock();
    solver->smooth_time += ((double)(end-start))/CLK_TCK;
    if(solver->print_level > 1) {
      GCGE_Printf("\nPostsmoothing\t");
    }
    PASE_Mg_print_eigenvalue_of_current_level(solver);
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
    GCGE_Printf("%d-level\n", solver->idx_cycle_level[solver->cur_cycle_level]);
    for(i = 0; i < solver->block_size; ++i) {
      GCGE_Printf("%d-th eig=%.8e\n", i, solver->eigenvalues[i]);
    }
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
      GCGE_Printf("\nDirect solve\t");
    }
    PASE_Mg_print_eigenvalue_of_current_level(solver);
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
  //cur_level ==  solver->nlevl-1
  PASE_INT cur_level = solver->idx_cycle_level[solver->cur_cycle_level];
  //last_level = 0
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
    //暂时只考虑二网格
    //cur_level ==  solver->nlevl-1,  last_level = 0
    PASE_Mg_set_pase_aux_matrix_by_pase_matrix(solver, cur_level, last_level, solver->u);
  return 0;
}

/**
 * @brief 根据细空间矩阵创建辅助粗空间的矩阵
 *
 * @param solver  输入/输出参数
 * @param i       输入参数     粗层
 * @param j       输入参数     细层
 * @param u_j     输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_pase_aux_matrix_by_pase_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, void **u_j)
{
  //i ==  solver->nlevl-1,  j = 0
  PASE_INT block_size = solver->block_size;
  PASE_INT nlock      = solver->nlock;
  PASE_INT idx_block  = 0;
  void **A = solver->multigrid->A_array;
  void **B = solver->multigrid->B_array;
  //创建复合矩阵，申请空间
  PASE_Mg_pase_aux_matrix_create(solver, i);
  PASE_Matrix aux_A = solver->multigrid->aux_A;
  PASE_Matrix aux_B = solver->multigrid->aux_B;
  PASE_INT error;

  //先对向量做边界处理再用其构造矩阵,-----没用
  //solver->gcge_ops->SetDirichletBoundary(u_j, block_size, A[j], B[j]);
  /* TODO 先创建这个矩阵 */
  //solver->gcge_ops->MultiVecPrint(u_j, 1, solver->gcge_ops);
  error = PASE_Aux_matrix_set_by_pase_matrix(aux_A, A[j], u_j, solver, i, j);
  error = PASE_Aux_matrix_set_by_pase_matrix(aux_B, B[j], u_j, solver, i, j);
  //对复合矩阵做边界处理后，特征值变小
  //solver->gcge_ops->SetDirichletBoundary(aux_A->aux_Hh, block_size, 
  //        aux_A->A_H, aux_B->A_H);
  //solver->gcge_ops->SetDirichletBoundary(aux_B->aux_Hh, block_size, 
  //        aux_A->A_H, aux_B->A_H);
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
  PASE_INT    block_size = solver->block_size;
  PASE_INT    idx_block  = 0;
  void        **A        = solver->multigrid->A_array;
  void        **B        = solver->multigrid->B_array;
  PASE_Matrix aux_A      = solver->multigrid->aux_A;
  PASE_Matrix aux_B      = solver->multigrid->aux_B;
  if(NULL == solver->multigrid->aux_A) {
      /* TODO 这里原本的创建，有很多参数赋值, 看是否需要 */
      PASE_MatrixCreate(&(solver->multigrid->aux_A), block_size, A[i], solver->pase_ops);
  }
  if(NULL == solver->multigrid->aux_B) {
      PASE_MatrixCreate(&(solver->multigrid->aux_B), block_size, B[i], solver->pase_ops);
  }
  return 0;
}

PASE_INT 
PASE_Aux_matrix_set_by_pase_matrix(PASE_Matrix aux_A, void *A_h, void **u_j, 
        PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j)
{
  //GCGE_Printf("line 757\n");
  //solver->gcge_ops->MultiVecPrint(u_j, 1, solver->gcge_ops);
  //matlab符号，包括前面，不包括后面
  //计算 aux_A->aux_Hh(:, nlock:block_size) 
  //   = I_h^H * A_h * u_h(:, nlock:block_size) 
  //计算 aux_A->aux_hh(:, nlock:block_size) 
  //   = u_h * A_h * u_h(:, nlock:block_size) 

  //计算 aux_A->aux_Hh(:, nlock:block_size) 
  //   = I_h^H * A_h * u_h(:, nlock:block_size) 
  GCGE_OPS *gcge_ops = solver->gcge_ops;
  PASE_OPS *pase_ops = solver->pase_ops;
  // TODO 应该用相应层的u_tmp[j]
  void     **u_tmp = solver->u_tmp[j];
  PASE_INT nlock = solver->nlock;
  nlock = 0;
  PASE_INT block_size = solver->block_size;
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  mv_s[0]  = nlock;
  mv_e[0]  = block_size;
  mv_s[1]  = 0;
  mv_e[1]  = block_size-nlock;
  // u_tmp(:,0:block_size-nlock) = A_h * u_h(:, nlock:block_size) 
  gcge_ops->MatDotMultiVec(A_h, u_j, u_tmp, mv_s, mv_e, gcge_ops);
  //i==1, j==0, 从j限制到1
  mv_s[0] = 0;
  mv_e[0] = block_size-nlock;
  mv_s[1] = nlock;
  mv_e[1] = block_size;
  /* TODO 简单测试的话这里该用什么？ */
  PASE_INT error = PASE_MULTIGRID_FromItoJ(solver->multigrid, 
          j, i, mv_s, mv_e, u_tmp, aux_A->aux_Hh);
  //计算 aux_A->aux_hh(:, nlock:block_size) 
  //   = u_h(:, 0:block_size) * A_h * u_h(:, nlock:block_size) 
  mv_s[0] = 0;
  mv_e[0] = block_size;
  mv_s[1] = 0;
  mv_e[1] = block_size-nlock;
  PASE_INT num_aux_vec = aux_A->num_aux_vec;
  PASE_REAL *A_aux_hh = aux_A->aux_hh;
  gcge_ops->MultiVecInnerProd(u_j, u_tmp, A_aux_hh+nlock*num_aux_vec,
          "nonsym", mv_s, mv_e, block_size, 0, solver->gcge_ops);
  //aux_hh对称化
  PASE_INT k = 0;//k表示列号
  PASE_INT l = 0;//l表示行号
  for(k=nlock; k<block_size; k++) {
      for(l=0; l<nlock; l++) {
          A_aux_hh[l*num_aux_vec+k] = A_aux_hh[k*num_aux_vec+l];
      }
  }
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

  /* TODO 这里的设置是否是对的 */
  PASE_INT block_size = solver->block_size;
  PASE_INT nlock = solver->nlock;
  PASE_INT idx_eigen  = 0;
  if(NULL == solver->aux_u) {
      solver->pase_ops->MultiVecCreateByMat((void***)(&(solver->aux_u)), block_size, 
              solver->multigrid->aux_A, solver->pase_ops);
#if 0
    solver->aux_u[cur_level] = (PASE_AUX_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_AUX_VECTOR));
    solver->aux_u[cur_level][0] = PASE_Aux_vector_create_by_aux_matrix(solver->multigrid->aux_A[cur_level]);
    for(idx_eigen = 1; idx_eigen < block_size; idx_eigen++) {
      solver->aux_u[cur_level][idx_eigen] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[cur_level][0]);
    }
#endif
  }

  /*多次迭代需要多次初始化初值，但空间不需要重新申请*/
  //子空间复合矩阵特征值问题的初值[0,e_i]^T
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  mv_s[0] = 0;
  mv_e[0] = block_size;
  mv_s[1] = 0;
  mv_e[1] = block_size;
  solver->pase_ops->MultiVecAxpby(0.0, (void**)(solver->aux_u),
          0.0, (void*)(solver->aux_u), mv_s, mv_e, solver->pase_ops);
  PASE_REAL *aux_h = solver->aux_u->aux_h;
  PASE_INT i = 0;
  PASE_INT j = 0;
  PASE_INT num_aux_vec = solver->aux_u->num_aux_vec;
  for(i=0; i<block_size; i++) {
      aux_h[i*num_aux_vec+i] = 1.0;
  }
#if 0
  for(idx_eigen = solver->nlock; idx_eigen < block_size; idx_eigen++) {
    PASE_Vector_set_constant_value(solver->aux_u[cur_level][idx_eigen]->vec, 0.0);
    memset(solver->aux_u[cur_level][idx_eigen]->block, 0, block_size*sizeof(PASE_SCALAR));
    solver->aux_u[cur_level][idx_eigen]->block[idx_eigen] = 1.0;
  }
#endif
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
  //next_level == 0
  PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(solver, cur_level, solver->aux_u, next_level, solver->u);
#if 0
  if(0 == next_level) {
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(solver, cur_level, solver->aux_u[cur_level], next_level, solver->u);
  } else {
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_aux_vector(solver, cur_level, solver->aux_u[cur_level], next_level, solver->aux_u[next_level]);
  }
#endif
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
PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_MultiVector aux_u_i, PASE_INT j, void **u_j)
{
    PASE_INT mv_s[2];
    PASE_INT mv_e[2];
    PASE_INT nconv = solver->nconv;
    PASE_INT block_size = solver->block_size;
    mv_s[0] = nconv;
    mv_e[0] = block_size;
    mv_s[1] = nconv;
    mv_e[1] = block_size;
    // TODO 应该用相应层的u_tmp[j]
    //从aux_u_i->b_H延拓到u_j(先放在u_tmp)
    //同时aux_u_i->aux_h作为线性组合系数，对原u_j进行线性组合
    PASE_INT error = PASE_MULTIGRID_FromItoJ(solver->multigrid, 
          i, j, mv_s, mv_e, aux_u_i->b_H, solver->u_tmp[j]);
    //线性组合时，alpha=beta=1.0
    mv_s[0] = nconv;
    mv_e[0] = block_size;
    mv_s[1] = nconv;
    mv_e[1] = block_size;
    solver->gcge_ops->MultiVecLinearComb(u_j, solver->u_tmp[j], mv_s, mv_e,
          aux_u_i->aux_h+nconv*block_size+nconv, aux_u_i->num_aux_vec, 0, 
          1.0, 1.0, solver->gcge_ops);
    mv_s[0] = nconv;
    mv_e[0] = block_size;
    mv_s[1] = nconv;
    mv_e[1] = block_size;
    solver->gcge_ops->MultiVecSwap(u_j, solver->u_tmp[j], mv_s, mv_e, solver->gcge_ops);
#if 0
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
#endif

  return 0;
}

PASE_INT
PASE_Mg_presmoothing_by_amg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_amg_hypre(mg_solver, "pre");
  return 0;
}


PASE_INT
PASE_Mg_postsmoothing_by_amg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_amg_hypre(mg_solver, "post");
  return 0;
}


PASE_INT 
PASE_Mg_smoothing_by_amg_hypre(void *mg_solver, char *PreOrPost)
{
  //PASE_SCALAR     inner_A, inner_B;
  PASE_MG_SOLVER  solver        = (PASE_MG_SOLVER)mg_solver;
  PASE_INT        block_size	= solver->block_size;
  PASE_INT        nconv         = solver->nconv; 
  nconv = 0;
  PASE_INT        i		= 0;
  PASE_INT        max_iter      = 1;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter = solver->max_pre_iter;
    if(NULL == solver->method_pre) {
      solver->method_pre = "amg";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter = solver->max_post_iter;
    if(NULL == solver->method_post) {
      solver->method_post = "amg";
    }
  }

  void           *A             = solver->multigrid->A_array[0];            
  void           *B             = solver->multigrid->B_array[0];            
  void          **u             = solver->u;
  PASE_SCALAR    *eigenvalues   = solver->eigenvalues;
  // TODO 这里也是相应层的u_tmp
  void          **rhs           = solver->u_tmp[0];

  GCGE_OPS *gcge_ops = solver->gcge_ops;
  PASE_OPS *pase_ops = solver->pase_ops;
 
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  //求解 A * u = eig * B * u
  //计算 rhs = eig * B * u
  mv_s[0] = nconv;
  mv_e[0] = block_size;
  mv_s[1] = 0;
  mv_e[1] = block_size-nconv;
  gcge_ops->MatDotMultiVec(B, u, rhs, mv_s, mv_e, gcge_ops);
  for(i=nconv; i<block_size; i++)
  {
      gcge_ops->MultiVecAxpbyColumn(0.0, rhs, i-nconv, eigenvalues[i], 
              rhs, i-nconv, gcge_ops);
  }

  //用GCGE_BCG求解
  PASE_REAL cg_rate = 1e-8;
  max_iter = 100;
  GCGE_BCG(A, rhs, u, nconv, block_size-nconv, max_iter, cg_rate,
      /* TODO break; */
          solver->gcge_ops, solver->u_tmp_1[0], solver->u_tmp_2[0], 
          solver->double_tmp, solver->int_tmp);
  //double_tmp: 6 * block_size
  //   int_tmp:     block_size
  //   u_tmp_1:     block_size
  //   u_tmp_2:     block_size


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

  PASE_Matrix      aux_A       = solver->multigrid->aux_A;            
  PASE_Matrix      aux_B       = solver->multigrid->aux_B;            
  PASE_MultiVector aux_u       = solver->aux_u;
  PASE_SCALAR     *eigenvalues = solver->eigenvalues;

  PASE_INT         max_iter    = solver->max_direct_iter;
  PASE_REAL        gcg_tol     = solver->atol*1e-2;
  PASE_REAL        cg_tol      = solver->atol*1e-2;
  PASE_INT         cg_max_iter = 50;

  /* TODO 这个DIAG_GCG是什么意思？ */
#if DIAG_GCG
  PASE_Mg_precondition_for_gcg(mg_solver);
  solver->method_dire = "gcg with diag scheme";
#else
  solver->method_dire = "gcg";
#endif
#if 0
  GCGE_Printf("aux_A\n");
  solver->pase_ops->PrintMat((void*)aux_A);
  GCGE_Printf("aux_B\n");
  solver->pase_ops->PrintMat((void*)aux_B);
#endif

  /* TODO PASE-GCGE 接口, GCGE不支持从中间开始算, 修改gcge_eigsol.c, 开始先检查一下收敛性 */
  //这里的PASE-GCGE接口需要gcge_app_pase.h, 但是gcge_app_pase.h需要pase.h, 因此需要将pase文件夹分成两部分
  //创建solver
  GCGE_SOLVER *gcge_pase_solver = GCGE_SOLVER_PASE_Create(aux_A, 
          aux_B, block_size, eigenvalues, aux_u, solver->pase_ops);
  //求解
  gcge_pase_solver->para->print_para = 0;
  gcge_pase_solver->para->print_conv = 0;
  gcge_pase_solver->para->print_eval = 0;
  gcge_pase_solver->para->print_result = 0;
  gcge_pase_solver->para->given_init_evec = 1;
  gcge_pase_solver->para->ev_tol = 1e-12;
  //gcge_pase_solver->para->orth_para->orth_zero_tol = 1e-13;
  //gcge_pase_solver->para->ev_max_it = 5;
  //gcge_pase_solver->para->p_orth_type = "gs";
  //精度高时，multi正交化达不到要求
  gcge_pase_solver->para->w_orth_type = "gs";
  //gcge_pase_solver->para->num_unlock = block_size - solver->nconv;
  GCGE_SOLVER_Solve(gcge_pase_solver);  
  //释放空间
  GCGE_SOLVER_Free(&gcge_pase_solver);

  //没用
  //solver->gcge_ops->SetDirichletBoundary(aux_u->b_H, block_size, 
  //        aux_A->A_H, aux_B->A_H);

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
    PASE_MULTIGRID_Destroy(&(solver->multigrid));
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
    GCGE_Printf("\n");
    GCGE_Printf("=============================================================\n");
    for(idx_eigen=0; idx_eigen<solver->block_size; idx_eigen++) {
      GCGE_Printf("%d-th eig=%.8e, aresi = %.8e\n", idx_eigen, solver->eigenvalues[idx_eigen], solver->r_norm[idx_eigen]);
    }
    GCGE_Printf("=============================================================\n");
    GCGE_Printf("set up time       = %f seconds\n", solver->set_up_time);
    GCGE_Printf("get initvec time  = %f seconds\n", solver->get_initvec_time);
    GCGE_Printf("smooth time       = %f seconds\n", solver->smooth_time);
    GCGE_Printf("set aux time      = %f seconds\n", solver->set_aux_time);
    GCGE_Printf("prolong time      = %f seconds\n", solver->prolong_time);
    GCGE_Printf("direct solve time = %f seconds\n", solver->direct_solve_time);
    GCGE_Printf("total solve time  = %f seconds\n", solver->total_solve_time);
    GCGE_Printf("total time        = %f seconds\n", solver->total_time);
    GCGE_Printf("=============================================================\n");
    GCGE_Printf("Direct solve time statistics\n");
    //GCGE_Printf("Tmatvec     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tmatvec+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tmatvec);
    //GCGE_Printf("Tveccom     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tveccom+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tveccom);
    //GCGE_Printf("Tvecvec     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tvecvec+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tvecvec);
    //GCGE_Printf("Tblockb     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tblockb+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tblockb);
    //GCGE_Printf("TMatVec     = %f seconds\n", solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Ttotal+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Ttotal);
    //GCGE_Printf("TVecVec     = %f seconds\n", solver->time_inner+solver->multigrid->aux_A[solver->idx_cycle_level[solver->max_cycle_level]]->Tinnergeneral+solver->multigrid->aux_B[solver->idx_cycle_level[solver->max_cycle_level]]->Tinnergeneral);
#if 0
    GCGE_Printf("TLapack     = %f seconds\n", solver->time_lapack);
    GCGE_Printf("Torth       = %f seconds\n", solver->time_orth_gcg);
    GCGE_Printf("Tother      = %f seconds\n", solver->time_other);
    GCGE_Printf("Tdiagpre    = %f seconds\n", solver->time_diag_pre);
    GCGE_Printf("Tdiaglinear = %f seconds\n", solver->time_linear_diag);
#endif
    GCGE_Printf("=============================================================\n");
    if(NULL != solver->method_init) {
      GCGE_Printf("Get initial vector:         %s\n", solver->method_init);
    }
    if(NULL != solver->method_pre) {
      GCGE_Printf("Presmoothing:               %s\n", solver->method_pre);
    }
    if(NULL != solver->method_post) {
      GCGE_Printf("Postsmoothing:              %s\n", solver->method_post);
    }
    if(NULL != solver->method_pre_aux) {
      GCGE_Printf("Presmoothing in aux space:  %s\n", solver->method_pre_aux);
    }
    if(NULL != solver->method_post_aux) {
      GCGE_Printf("Postsmoothing in aux space: %s\n", solver->method_post_aux);
    }
    if(NULL != solver->method_dire) {
      GCGE_Printf("Direct solve:               %s\n", solver->method_dire);
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
    GCGE_Printf("\nInitial   \t");
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);
  return 0;
}

PASE_INT
PASE_Mg_get_initial_vector_by_full_multigrid_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver       = (PASE_MG_SOLVER)mg_solver;
  solver->method_init         = "full multigrid";
  PASE_REAL      atol 	      = solver->atol;	/* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol         = 1e-50;
  GCGE_OPS      *gcge_ops     = solver->gcge_ops;

  clock_t     start, end;
  PASE_REAL   direct_time     = 0.0;
  PASE_REAL   aux_direct_time = 0.0;
  PASE_REAL   smooth_time     = 0.0;
  PASE_REAL   set_aux_time    = 0.0;
  PASE_REAL   prolong_time    = 0.0;

  PASE_INT     i              = 0;
  PASE_INT     cur_level      = solver->idx_cycle_level[solver->max_cycle_level];        
  PASE_INT     block_size     = solver->block_size;       
  PASE_INT     max_block_size = solver->max_block_size;
  void       **A              = solver->multigrid->A_array;
  void       **B              = solver->multigrid->B_array;
  PASE_SCALAR *eigenvalues    = solver->eigenvalues;
  void      ***pase_u         = solver->u_tmp_1;
  void      ***rhs            = solver->u_tmp;
  PASE_INT     mv_s[2];
  PASE_INT     mv_e[2];
  PASE_INT     idx_level      = 0;
  PASE_INT     max_iter       = 10;
  PASE_REAL    cg_rate        = 0.1;

  //------------------------------------------------------------
  //使用GCGE在最粗层(cur_level)空间获取初值
  GCGE_SOLVER *gcge_solver = GCGE_SOLVER_CreateByOps(A[cur_level], B[cur_level],
          block_size, eigenvalues, pase_u[cur_level], solver->gcge_ops);
  gcge_solver->para->print_para = 0;
  gcge_solver->para->print_eval = 0;
  gcge_solver->para->print_result = 1;
  //求解
  GCGE_SOLVER_Solve(gcge_solver);  
  //释放空间, 不释放gcge_solver->ops
  GCGE_SOLVER_Free_Some(&gcge_solver);
  //GCGE在最粗层空间获取初值
  //------------------------------------------------------------
  //粗层特征向量从cur_level层延拓到cur_level-1层
  mv_s[0] = 0;
  mv_e[0] = block_size;
  mv_s[1] = 0;
  mv_e[1] = block_size;
  //这里i只是需要一个返回值，没有具体意义
  i = PASE_MULTIGRID_FromItoJ(solver->multigrid, cur_level, cur_level-1, 
          mv_s, mv_e, pase_u[cur_level], pase_u[cur_level-1]);
  i = PASE_MULTIGRID_FromItoJ(solver->multigrid, cur_level-1, 0, 
          mv_s, mv_e, pase_u[cur_level-1], solver->u);
  GCGE_Printf("coarsest GCGE\n");
  PASE_Mg_error_estimate(solver);
  //粗层特征向量延拓到下一层
  //----------------------------------------------------
  //在每层上计算
  for(idx_level = cur_level-1; idx_level > 0; idx_level--) {
    //----------------------------------------------------
    //每个细层上解问题
    //----------------------------------------------------
    //首先计算当前层右端项 \lambda * B * u
    mv_s[0] = 0;
    mv_e[0] = block_size;
    mv_s[1] = 0;
    mv_e[1] = block_size;
    gcge_ops->MatDotMultiVec(B[idx_level], pase_u[idx_level], rhs[idx_level], mv_s, mv_e, gcge_ops);
    for(i=0; i<block_size; i++) {
        solver->gcge_ops->MultiVecAxpbyColumn(0.0, rhs[idx_level], i, eigenvalues[i], rhs[idx_level], i, solver->gcge_ops);
    }
    //用GCGE_BCG求解
    GCGE_BCG(A[idx_level], rhs[idx_level], pase_u[idx_level], 0, block_size, max_iter, cg_rate,
            solver->gcge_ops, solver->u_tmp_2[idx_level], solver->u_tmp_3[idx_level], 
            solver->double_tmp, solver->int_tmp);
    //testtesttest-------------------
    i = PASE_MULTIGRID_FromItoJ(solver->multigrid, idx_level, 0, 
          mv_s, mv_e, pase_u[idx_level], solver->u);
    GCGE_Printf("level: %d, BCG\n", idx_level);
    PASE_Mg_error_estimate(solver);
    //testtesttest-------------------
    //每个细层上解问题
    //----------------------------------------------------
    //用最粗层与目前细层构造复合辅助矩阵,cur_level表示最粗层，idx_level为目前细层
    PASE_Mg_set_pase_aux_matrix_by_pase_matrix(solver, cur_level, idx_level, pase_u[idx_level]);
    PASE_Mg_set_pase_aux_vector(solver, cur_level);
    //用最粗层与目前细层构造复合辅助矩阵
    //----------------------------------------------------
    //求解复合矩阵特征值问题
    PASE_Mg_direct_solve_by_gcg(solver);
    //复合矩阵特征值问题
    //----------------------------------------------------
    //先由复合向量投影到当前 idx_level层
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(solver, cur_level, solver->aux_u, idx_level, pase_u[idx_level]);
    //----------------------------------------------------
    //再从当前idx_level层投影到 idx_level-1 层
    mv_s[0] = 0;
    mv_e[0] = block_size;
    mv_s[1] = 0;
    mv_e[1] = block_size;
    i = PASE_MULTIGRID_FromItoJ(solver->multigrid, idx_level, idx_level-1, mv_s, mv_e, pase_u[idx_level], pase_u[idx_level-1]);
  }
  mv_s[0] = 0;
  mv_e[0] = block_size;
  mv_s[1] = 0;
  mv_e[1] = block_size;
  solver->gcge_ops->MultiVecSwap(solver->u, pase_u[0], mv_s, mv_e, solver->gcge_ops);
  GCGE_Printf("after init\n");
  PASE_Mg_error_estimate(solver);
  //----------------------------------------------------
  //最后再做一次AMG迭代(后光滑)
  PASE_Mg_smoothing_by_amg_hypre(solver, "NeitherPreNorPost");
  return 0;
}

