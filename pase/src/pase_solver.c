#include <time.h>
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
PASE_EigenSolver(void *A, void *B, PASE_REAL *eval, void **evec, 
        PASE_INT nev, PASE_PARAMETER param, GCGE_OPS *gcge_ops)
{
  clock_t start, end;
  start = clock();

  //创建solver空间并赋初值NULL
  //pase的eigensolver，依赖于MG
  PASE_MG_SOLVER solver = PASE_Mg_solver_create(param);

  //进行AMG分层，分配工作空间
  PASE_Mg_set_up(solver, A, B, gcge_ops);

  //打印参数信息
  PASE_Mg_print_param(solver);

  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  //用户给initial_level初值，可能是给最细层或者其他层
  if(solver->num_given_eigs > 0)
  {
    mv_s[0] = 0;
    mv_e[0] = nev;
    mv_s[1] = 0;
    mv_e[1] = nev;
    gcge_ops->MultiVecAxpby(1.0, evec, 0.0, solver->sol[solver->initial_level], mv_s, mv_e, gcge_ops);
  }

  //进行pase求解
  PASE_Mg_solve(solver);

  //将解返回给用户提供的特征对空间
  if(eval != NULL) {
    memcpy(eval, solver->eigenvalues, nev*sizeof(PASE_REAL));
  }
  if(evec != NULL) {
    if((solver->initial_level == solver->finest_level)||(solver->num_given_eigs == 0)) {
      //如果用户通过evec给了不是最细层的初值, 那就不做下面的拷贝
      mv_s[0] = 0;
      mv_e[0] = nev;
      mv_s[1] = 0;
      mv_e[1] = nev;
      gcge_ops->MultiVecAxpby(1.0, solver->sol[0], 0.0, evec, mv_s, mv_e, gcge_ops);
    }
  }

  end = clock();
  solver->total_time += ((double)(end-start))/CLK_TCK;
  //打印结果与时间信息
  PASE_Mg_print_result(solver);

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
  if(param->mg_coarsest_level == -1) {
    solver->mg_coarsest_level = solver->num_levels - 1;
  } else {
    solver->mg_coarsest_level = param->mg_coarsest_level;
  }
  if(param->finest_level == -1) {
    solver->finest_level = 0;
  } else {
    solver->finest_level = param->finest_level;
  }
  if(param->initial_aux_coarse_level == -1) {
    solver->initial_aux_coarse_level = solver->num_levels - 1;
  } else {
    solver->initial_aux_coarse_level = param->initial_aux_coarse_level;
  }
  solver->aux_coarse_level = solver->initial_aux_coarse_level;
  if(param->finest_aux_coarse_level == -1) {
    solver->finest_aux_coarse_level = 1;
  } else {
    solver->finest_aux_coarse_level = param->finest_aux_coarse_level;
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
  solver->max_initial_direct_count    = param->max_initial_direct_count;

  //实际要求的特征对个数
  solver->nev = param->nev;
  //最多算多少个特征对
  solver->max_nev = ((2* param->nev)<(param->nev+param->more_nev))?(2* param->nev):(param->nev+param->more_nev);
  //实际计算的特征值个数
  if(solver->initial_level == solver->finest_level) {
    //如果初始层是最细层, 那么要求用户给至少nev个初值
    if(param->num_given_eigs >= param->nev) {
      //如果给了初值, 那初始前光滑就对给定个数的向量操作
      //在求解辅助矩阵特征值问题时会将pase_nev重赋值max_nev
      solver->pase_nev = param->num_given_eigs;
    } else {
      //初始层是最细层，但又没给足够的初值，那就报错退出
      GCGE_Printf("ERROR! The initial level equals the finest level, "
                  "please give not less than nev initial vectors, "
                  "or modify the initial level!\n");
      exit(0);
    }
  } else {
    //如果初始层不是最细层, 那在初始层上会先进行GCGE直接求解, 让GCGE直接求解pase_nev个初始向量即可
    //初值个数会在PASE_Direct_solve中调用GCGE时赋值给GCGE_PARA
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
  //在初始网格层上求解初始特征对逼近时的收敛准则(残差准则)
  solver->initial_rtol = param->initial_rtol;
  //在每层网格上二网格迭代中求解辅助矩阵特征值问题时的收敛准则（残差准则）
  solver->aux_rtol = param->aux_rtol;
  //以上两种收敛准则满足其一即可认定为收敛
  //打印level
  solver->print_level = param->print_level; 
  //abs_res_norm用于存储特征对的绝对残差
  solver->abs_res_norm = NULL;
  //已经收敛的特征对个数
  solver->nlock_direct = 0;
  solver->nlock_smooth = 0;
  solver->nconv = 0;
  solver->nlock_auxmat_A = NULL;
  solver->nlock_auxmat_B = NULL;

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
  solver->aux_sol = NULL;
  //pase aux 复合矩阵
  solver->aux_A = NULL;
  //pase aux 复合矩阵
  solver->aux_B = NULL;
  //pase aux 复合矩阵的特征值
  solver->aux_eigenvalues = NULL;
  //多重网格结构
  solver->multigrid = NULL;

  //设置自动调节aux_coarse_level的参数
  solver->conv_efficiency = NULL; 
  solver->check_efficiency_flag = param->check_efficiency_flag;

  //统计时间
  solver->initialize_convert_time = 0.0;
  solver->initialize_amg_time = 0.0;
  solver->smooth_time = 0.0;
  solver->build_aux_time = 0.0;
  solver->prolong_time = 0.0;
  solver->aux_direct_solve_time = 0.0;
  solver->total_solve_time = 0.0;
  solver->total_time = 0.0;
  solver->get_initvec_time = 0.0;

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
  //层指标
  PASE_INT idx_level = 0;

  //用gcge_ops创建pase_ops
  PASE_OPS_Create(&(solver->pase_ops), gcge_ops);
  solver->gcge_ops = gcge_ops;

  //--------------------------------------------------------------------
  //--特征值工作空间-----------------------------------------------------
  PASE_INT max_nev = solver->max_nev;
  solver->eigenvalues = (PASE_REAL*)calloc(max_nev, sizeof(PASE_REAL));
  solver->aux_eigenvalues = (PASE_REAL*)calloc(max_nev, sizeof(PASE_REAL));
  //abs_res_norm存储每个特征对的绝对残差
  solver->abs_res_norm = (PASE_REAL*)calloc(max_nev, sizeof(PASE_REAL));
  //辅助矩阵的b_H中lock住的列数，并初始化为0
  solver->nlock_auxmat_A = (PASE_INT*)calloc(solver->num_levels, sizeof(PASE_INT));
  for(idx_level=0; idx_level<solver->num_levels; idx_level++) {
    solver->nlock_auxmat_A[idx_level] = 0;
  }
  solver->nlock_auxmat_B = (PASE_INT*)calloc(solver->num_levels, sizeof(PASE_INT));
  for(idx_level=0; idx_level<solver->num_levels; idx_level++) {
    solver->nlock_auxmat_B[idx_level] = 0;
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!之前出问题，rhs的内存受影响，
  //竟然是因为这里分配空间的时候给PASE_REAL分配了PASE_INT的空间!!!!!!!!!!!!!!!!!!
  solver->conv_efficiency = (PASE_REAL*)calloc(solver->num_levels, sizeof(PASE_REAL));

  //--普通向量组空间-----------------------------------------------------
  PASE_INT num_levels = solver->num_levels;
  PASE_INT *sol_size    = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *rhs_size    = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *cg_p_size   = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *cg_w_size   = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  PASE_INT *cg_res_size = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
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
  int **size = (int**)malloc(5*sizeof(int*));
  size[0] = sol_size;
  size[1] = rhs_size;
  size[2] = cg_p_size;
  size[3] = cg_w_size;
  size[4] = cg_res_size;
  int size_dtmp = 6*step_size;
  int size_itmp = step_size;

  //以矩阵A，B作为最细层空间，创建多重网格结构
  //TODO 这里可能会修改max_levels?
  PASE_MULTIGRID_Create(&(solver->multigrid), 
        solver->num_levels, solver->mg_coarsest_level, 
        size, size_dtmp, size_itmp,
        A, B, gcge_ops,
	&(solver->initialize_convert_time), 
	&(solver->initialize_amg_time));
  free(size); size = NULL;
  solver->multigrid->coarsest_level = solver->mg_coarsest_level;


  solver->sol         = solver->multigrid->sol;
  solver->rhs         = solver->multigrid->rhs;
  solver->cg_p        = solver->multigrid->cg_p;
  solver->cg_w        = solver->multigrid->cg_w;
  solver->cg_res      = solver->multigrid->cg_res;
  solver->sol_size    = sol_size;
  solver->rhs_size    = rhs_size;
  solver->cg_p_size   = cg_p_size;
  solver->cg_w_size   = cg_w_size;
  solver->cg_res_size = cg_res_size;

  //--------------------------------------------------------------------
  //double_tmp在BCG中是6倍的空间, 用于存储BCG中的各种内积残差等
  solver->cg_double_tmp = solver->multigrid->cg_double_tmp;
  //int_tmp在BCG中是1倍的空间，用于存储unlock
  solver->cg_int_tmp    = solver->multigrid->cg_int_tmp;

  //--------------------------------------------------------------
  //检查一些参数的不合理设置
  //如果辅助矩阵的粗层比获取初值的网格层要细, 或者是同一层
  if(solver->aux_coarse_level <= solver->initial_level) {
    //将solver的获取初值的网格层设为辅助矩阵的粗层
    //solver->initial_level = solver->aux_coarse_level;
    //初始层上可以算的比较精确
    if(solver->initial_rtol > solver->rtol) {
      solver->initial_rtol = solver->rtol;
    }
    //且在初始层到辅助矩阵粗层上不必再做二网格迭代
    for(idx_level=solver->initial_level; idx_level>=solver->aux_coarse_level; idx_level--) {
      solver->max_cycle_count_each_level[idx_level] = 0;
    }
  } else {
    //如果辅助矩阵的粗层比获取初值的网格层要粗, 
    //那么如果初始层上的精度要求比较高，就不必再在初始层做二网格迭代
    if(solver->initial_rtol <= solver->rtol) {
       solver->max_cycle_count_each_level[solver->initial_level] = 0;
    }
  }

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
  if(NULL != solver->aux_eigenvalues) {
    free(solver->aux_eigenvalues);
    solver->aux_eigenvalues = NULL;
  }
  if(NULL != solver->abs_res_norm) {
    free(solver->abs_res_norm);
    solver->abs_res_norm = NULL;
  }
  if(NULL != solver->nlock_auxmat_A) {
    free(solver->nlock_auxmat_A);
    solver->nlock_auxmat_A = NULL;
  }
  if(NULL != solver->nlock_auxmat_B) {
    free(solver->nlock_auxmat_B);
    solver->nlock_auxmat_B = NULL;
  }
  if(NULL != solver->conv_efficiency) {
    free(solver->conv_efficiency);
    solver->conv_efficiency = NULL;
  }
  //-----------------------------------------------------
  //释放向量工作空间
  int **size = (int**)malloc(5*sizeof(int*));
  size[0] = solver->sol_size;
  size[1] = solver->rhs_size;
  size[2] = solver->cg_p_size;
  size[3] = solver->cg_w_size;
  size[4] = solver->cg_res_size;
  if(NULL != solver->multigrid) {
    PASE_MULTIGRID_Destroy(&(solver->multigrid), size);
  }
  free(size); size = NULL;
  free(solver->sol_size);     solver->sol_size    = NULL;
  free(solver->rhs_size);     solver->rhs_size    = NULL;
  free(solver->cg_p_size);    solver->cg_p_size   = NULL;
  free(solver->cg_w_size);    solver->cg_w_size   = NULL;
  free(solver->cg_res_size);  solver->cg_res_size = NULL;

  //-----------------------------------------------------
  //释放 aux_sol空间
  if(NULL != solver->aux_sol) {
    PASE_MultiVector_destroy_sub(&(solver->aux_sol));
  }
  if(NULL != solver->aux_A) {
    PASE_Matrix_destroy_sub(&(solver->aux_A));
  }
  if(NULL != solver->aux_B) {
    PASE_Matrix_destroy_sub(&(solver->aux_B));
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
 * 4. prolong: (aux_sol, coarse_level, current_level)
 * 5. 后光滑 (solver, coarse_level, current_level
 *
 */
PASE_INT
PASE_Mg_solve(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  clock_t start_1, end_1;
  start = clock();

  //一共要用到的层数
  PASE_INT num_levels = solver->num_levels;
  //V_h_1的层号
  PASE_INT initial_level = solver->initial_level;
  //V_H层号
  PASE_INT aux_coarse_level = solver->aux_coarse_level;
  //V_hn层号
  PASE_INT finest_level = solver->finest_level;
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
  PASE_INT current_level = 0;
  //next_level表示下一个细层
  PASE_INT next_level = 0;
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  PASE_INT i = 0;
  PASE_REAL cycle_time = 0.0;

  //如果初始层不是最细层，先直接求解获取初值
  //那么，如果初始层是最细层，那就要求用户一定提供了至少nev个初值
  if(initial_level > finest_level) {
    PASE_Direct_solve(solver, initial_level);
  } 
  //各细层求解
  for(idx_level=initial_level; idx_level>finest_level-1; idx_level--) {
    //确定粗细层指标
    current_level = idx_level;
    //对该层网格与aux粗层进行二网格迭代
    for(idx_cycle=0; idx_cycle < solver->max_cycle_count_each_level[current_level]; idx_cycle++)
    {
      //进行二网格迭代
      start_1 = clock();
      PASE_Mg_cycle(solver, solver->aux_coarse_level, current_level);
      end_1 = clock();
      cycle_time = ((double)(end_1-start_1))/CLK_TCK;
      //检查收敛性并确定下次迭代的粗层
      PASE_Mg_get_new_aux_coarse_level(solver, current_level, &idx_cycle, cycle_time);
    }//end for idx_cycle
    if(current_level > 0)
    {
      //next_level表示下一个细层
      next_level = current_level-1;
      start_1 = clock();
      //如果当前层不是最细层, 将解sol从上一粗层插值到当前细层
      mv_s[0] = 0;
      mv_e[0] = pase_nev;
      mv_s[1] = 0;
      mv_e[1] = pase_nev;
      PASE_MULTIGRID_FromItoJ(solver->multigrid, current_level, next_level, 
          mv_s, mv_e, solver->sol[current_level], solver->sol[next_level]);
      end_1 = clock();
      solver->prolong_time += ((double)(end_1-start_1))/CLK_TCK;
    }
  }//end for idx_level

  end = clock();
  solver->total_solve_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/* 
 * 选取aux_coarse_level, 
 * 初始check_efficiency_flag=1
 * 第二次二网格迭代开始，如果check_efficiency_flag==1, 做下面的判断 
 *   如果当前效率更高，
 *       如果还没到第二细层，那就将aux_coarse_level放细一层
 *       否则已经是第二细层了, 那就使用当前aux_coarse_level，
 *                             且赋值check_efficiency_flag = 0
 *   否则，那么当前效率不如上一次迭代
 *       aux_coarse_level放粗一层，退回到上一次
 *       且赋值check_efficiency_flag = 0
 *       
 * check_efficiency_flag表示是否要做效率的对比检查，
 * 如果已经检测到了最优粗层, 那就不需要再做这样的检查
 *
 */
PASE_INT
PASE_Mg_get_new_aux_coarse_level(PASE_MG_SOLVER solver, PASE_INT current_level, 
      PASE_INT *idx_cycle, PASE_REAL cycle_time)
{
  //如果不是最细层，直接返回
  if(current_level > solver->finest_level) {
    GCGE_Printf("current_level: %d, idx_cycle = %d, aux_coarse_level: %d, cycle_time: %f\n", 
	  current_level, *idx_cycle+1, solver->aux_coarse_level, cycle_time);
    return 0;
  }
  //如果是最细层，先检查收敛性
  PASE_Mg_error_estimate(solver, current_level, *idx_cycle, cycle_time);
  if(solver->check_efficiency_flag == 1) {
    //如果还在检查效率，且还没到最细的辅助粗层, 就继续放细一层
    if(solver->aux_coarse_level > solver->finest_aux_coarse_level) {
      solver->aux_coarse_level -= 1;
    } else {
      //否则，那么目前就在最细的辅助粗层, 从初始aux_coarse_level到当前最细
      //对比选出最优的那一层(conv_efficiency越小, 效率越高)
      solver->aux_coarse_level = PASE_Get_min_double(solver->conv_efficiency, 
	    solver->finest_aux_coarse_level, solver->initial_aux_coarse_level);
      PASE_INT i = 0;
      for(i=solver->finest_aux_coarse_level; i<=solver->initial_aux_coarse_level; i++) {
        GCGE_Printf("conv_efficiency[%d] = %e\n", i, solver->conv_efficiency[i]);
      }
      GCGE_Printf("The best aux_coarse_level is level %d \n", solver->aux_coarse_level);
      //之后不再检查效率
      solver->check_efficiency_flag = 0;
    }
  }
  //如果已经全部收敛(只会在最细层修改nconv)
  if(solver->nconv >= solver->nev) {
    GCGE_Printf("total cycle count     = %d\n", *idx_cycle + 1);
    *idx_cycle = solver->max_cycle_count_each_level[current_level] + 1;
  }//end for if nconv>=nev
  return 0;
}

PASE_INT 
PASE_Get_min_double(PASE_REAL *a, PASE_INT start, PASE_INT end)
{
  PASE_INT  i = 0;
  PASE_REAL min_double = a[start];
  PASE_INT  min_idx = start;
  for(i=start+1; i<=end; i++) {
    if(a[i] < min_double) {
      min_double = a[i];
      min_idx = i;
    }
  }
  return min_idx;
}
/*
 * PASE_Mg_cycle(solver, coarse_level, current_level):
 * 1. 前光滑 (solver, coarse_level, current_level
 * 2. 构造复合矩阵 (solver, coarse_level, current_level)
 * 2. 构造复合向量 (solver, coarse_level)
 * 3. 复合矩阵特征值求解
 * 4. prolong: (aux_sol, coarse_level, current_level)
 * 5. 后光滑 (solver, coarse_level, current_level
 *
 */
PASE_INT 
PASE_Mg_cycle(PASE_MG_SOLVER solver, PASE_INT coarse_level, PASE_INT current_level)
{
  //前光滑
  PASE_INT max_presmooth_iter = solver->max_pre_count_each_level[current_level];
  //GCGE_Printf("before presmoothing:, current_level: %d\n", current_level);
  PASE_Mg_smoothing(solver, current_level, max_presmooth_iter);

  //GCGE_Printf("after presmoothing:\n");
  //PASE_Mg_error_estimate(solver, current_level);

  //构造复合矩阵 (solver, coarse_level, current_level)
  PASE_Mg_set_pase_aux_matrix(solver, coarse_level, current_level);
  //GCGE_Printf("after set_aux_matrix:\n");
  //构造复合向量 
  PASE_Mg_set_pase_aux_vector(solver, coarse_level, current_level);
  //GCGE_Printf("after set_aux_vector:\n");
  //直接求解复合矩阵特征值问题
  PASE_Aux_direct_solve(solver, coarse_level);
  //GCGE_Printf("after aux_direct:\n");

  //把aux向量转换成细空间上的向量
  PASE_Mg_prolong_from_pase_aux_vector(solver, coarse_level, current_level);
  //GCGE_Printf("after prolong:\n");

  //后光滑
  PASE_INT max_postsmooth_iter = solver->max_post_count_each_level[current_level];
  PASE_Mg_smoothing(solver, current_level, max_postsmooth_iter);
  //GCGE_Printf("after postsmoothing:\n");

  return 0;
}

PASE_INT
PASE_Mg_set_pase_aux_matrix(PASE_MG_SOLVER solver, PASE_INT coarse_level, 
      PASE_INT current_level)
{
  clock_t start, end;
  start = clock();

  //构造复合矩阵，细空间的维数, 从nlock_direct:pase_nev-1
  PASE_INT pase_nev = solver->pase_nev;

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
  void **current_sol = solver->sol[current_level];
  //给复合矩阵的aux部分赋值
  error = PASE_Aux_matrix_set_by_pase_matrix(aux_A, solver->rhs, 
	A, current_sol, solver, coarse_level, current_level, 
	solver->nlock_auxmat_A);
  error = PASE_Aux_matrix_set_by_pase_matrix(aux_B, solver->cg_p, 
	B, current_sol, solver, coarse_level, current_level, 
	solver->nlock_auxmat_B);

  end = clock();
  solver->build_aux_time += ((double)(end-start))/CLK_TCK;
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
  PASE_REAL   *eigenvalues  = solver->eigenvalues;
  void       **eigenvectors = solver->sol[idx_level];
  GCGE_SOLVER *gcge_solver  = GCGE_SOLVER_CreateByOps(A, B,
          solver->pase_nev, eigenvalues, eigenvectors, solver->gcge_ops);
  //设置参数
  gcge_solver->para->print_para    = 0;
  gcge_solver->para->print_eval    = 0;
  gcge_solver->para->print_conv    = 0;
  gcge_solver->para->print_result  = 0;
  gcge_solver->para->ev_tol        = solver->initial_rtol;
  gcge_solver->para->ev_max_it     = solver->max_initial_direct_count;
  //设定下面这个num_init_evec参数后，就可以给定任意多个向量作为初值
  gcge_solver->para->num_init_evec = solver->num_given_eigs;
  //求解
  GCGE_SOLVER_Solve(gcge_solver);  
  //释放空间, 不释放gcge_solver->ops
  GCGE_SOLVER_Free_Some(&gcge_solver);
  //每层上最多直接求解时GCGE迭代的次数
  end = clock();
  solver->get_initvec_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

//创建辅助矩阵空间，但并不分配具体的大规模空间
PASE_INT 
PASE_Matrix_create_sub(PASE_Matrix *aux_A, PASE_INT n)
{
  *aux_A = (PASE_Matrix)malloc(sizeof(pase_Matrix));
  (*aux_A)->is_diag = 0;
  (*aux_A)->aux_hh = (PASE_REAL*)calloc(n*n, sizeof(PASE_REAL));
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
  aux_A->num_aux_vec = solver->pase_nev - solver->nlock_direct;
  aux_B->num_aux_vec = solver->pase_nev - solver->nlock_direct;

  return 0;
}

PASE_INT 
PASE_Aux_matrix_set_by_pase_matrix(PASE_Matrix aux_A, void ***aux_bH, 
        void *A_h, void **sol, PASE_MG_SOLVER solver, 
	PASE_INT coarse_level, PASE_INT current_level, 
	PASE_INT *nlock_auxmat)
{
  //matlab符号，包括前面，不包括后面
  //计算 aux_A->aux_Hh(:, nlock_direct:pase_nev) 
  //   = I_h^H * A_h * u_h(:, nlock_direct:pase_nev) 
  //计算 aux_A->aux_hh(:, nlock_direct:pase_nev) 
  //   = u_h * A_h * u_h(:, nlock_direct:pase_nev) 

  //计算 aux_A->aux_Hh(:, nlock_direct:pase_nev) 
  //   = I_h^H * A_h * u_h(:, nlock_direct:pase_nev) 
  PASE_INT i = 0;
  PASE_INT j = 0;
  GCGE_OPS *gcge_ops = solver->gcge_ops;
  PASE_OPS *pase_ops = solver->pase_ops;
  PASE_INT pase_nev = solver->pase_nev;
  PASE_INT num_aux_vec = pase_nev;
  PASE_INT coarse_nlock_auxmat = nlock_auxmat[coarse_level];
  PASE_INT restrict_from_level = 0;
  PASE_INT restrict_to_level   = 0;
  //先从nlock_auxmat开始计算，再更新nlock_auxmat
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  mv_s[0]  = nlock_auxmat[current_level];
  mv_e[0]  = pase_nev;
  mv_s[1]  = nlock_auxmat[current_level];
  mv_e[1]  = pase_nev;
  // Ah_u(:,nlock_auxmat:pase_nev) = A_h * sol(:, nlock_auxmat:pase_nev) 
  gcge_ops->MatDotMultiVec(A_h, sol, aux_bH[current_level], mv_s, mv_e, gcge_ops);
  //计算完后更新当前层的nlock_auxmat
  nlock_auxmat[current_level] = solver->nconv;
  //从current_level限制到coarse_level
  for(i=current_level; i<coarse_level; i++) {
    //从i层限制到粗一层i+1层, 计算的向量从粗一层的nlock_auxmat开始
    restrict_from_level = i;
    restrict_to_level   = i+1;
    mv_s[0] = nlock_auxmat[restrict_to_level];
    mv_e[0] = pase_nev;
    mv_s[1] = nlock_auxmat[restrict_to_level];
    mv_e[1] = pase_nev;
    PASE_INT error = PASE_MULTIGRID_FromItoJ(solver->multigrid, 
        restrict_from_level, restrict_to_level, mv_s, mv_e, 
	aux_bH[restrict_from_level], aux_bH[restrict_to_level]);
    //计算完后更新粗一层的nlock_auxmat
    nlock_auxmat[restrict_to_level] = solver->nconv;
  }
  //计算 aux_A->aux_hh = sol(:, 0:pase_nev) * A_h * sol(:, coarse_nlock_auxmat_b:pase_nev) 
  mv_s[0] = 0;
  mv_e[0] = pase_nev;
  mv_s[1] = coarse_nlock_auxmat;
  mv_e[1] = pase_nev;
  PASE_REAL *A_aux_hh = aux_A->aux_hh;
  //从aux_A->aux_hh的第coarse_nlock_auxmat_b列开始赋值
  gcge_ops->MultiVecInnerProd(sol, aux_bH[current_level], 
	A_aux_hh+coarse_nlock_auxmat*num_aux_vec, 
	"nonsym", mv_s, mv_e, num_aux_vec, 0, solver->gcge_ops);
  //对称化处理
  for(i=coarse_nlock_auxmat; i<num_aux_vec; i++) {
    for(j=0; j<coarse_nlock_auxmat; j++) {
      A_aux_hh[j*num_aux_vec+i] = A_aux_hh[i*num_aux_vec+j];
    }
  }

  return 0;
}


PASE_INT
PASE_Mg_set_pase_aux_vector(PASE_MG_SOLVER solver, PASE_INT coarse_level, 
      PASE_INT current_level)
{
  clock_t start, end;
  start = clock();

  PASE_Mg_pase_aux_vector_create(solver, coarse_level);

  //子空间复合矩阵特征值问题的初值[0,e_i]^T
  PASE_INT  pase_nev = solver->pase_nev;
  PASE_INT  nlock_direct = solver->nlock_direct;
  PASE_INT  num_aux_vec = solver->aux_sol->num_aux_vec;
  PASE_REAL *aux_h = solver->aux_sol->aux_h;
  PASE_INT  i = 0;
  PASE_INT  mv_s[2];
  PASE_INT  mv_e[2];
  //把aux_sol(:, nlock_direct:pase_nev-1)赋值为[0,e_i]^T
  mv_s[0] = nlock_direct;
  mv_e[0] = pase_nev;
  mv_s[1] = nlock_direct;
  mv_e[1] = pase_nev;
  solver->pase_ops->MultiVecAxpby(0.0, (void**)(solver->aux_sol),
          0.0, (void**)(solver->aux_sol), mv_s, mv_e, solver->pase_ops);
  for(i=0; i<num_aux_vec; i++) {
      aux_h[(i+nlock_direct)*num_aux_vec+i] = 1.0;
  }
  if(nlock_direct > 0)
  {
    //aux_sol(:, 0:nlock_direct-1) = R * sol(:, 0:nlock_direct-1)
    memset(solver->aux_sol->aux_h, 0.0, nlock_direct*num_aux_vec*sizeof(PASE_REAL));
    mv_s[0] = 0;
    mv_e[0] = nlock_direct;
    mv_s[1] = 0;
    mv_e[1] = nlock_direct;
    PASE_MULTIGRID_FromItoJ(solver->multigrid, current_level, coarse_level, 
        mv_s, mv_e, solver->sol[current_level], solver->aux_sol->b_H);
  }
  end = clock();
  solver->build_aux_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

//创建复合向量空间，大规模空间不申请，直接用solver中的工作空间
PASE_INT 
PASE_Mg_pase_aux_vector_create(PASE_MG_SOLVER solver, PASE_INT idx_level)
{
  if(solver->aux_sol == NULL) {
    PASE_MultiVector_create_sub(&(solver->aux_sol), solver->max_nev);
  }
  PASE_INT num_aux_vec = solver->pase_nev - solver->nlock_direct;
  solver->aux_sol->num_aux_vec = num_aux_vec;
  solver->aux_sol->num_vec = num_aux_vec;
  solver->aux_sol->b_H = solver->sol[idx_level];
  return 0;
}

//创建辅助矩阵空间，但并不分配具体的大规模空间
PASE_INT
PASE_MultiVector_create_sub(PASE_MultiVector *aux_sol, PASE_INT n)
{
  *aux_sol = (PASE_MultiVector)malloc(sizeof(pase_MultiVector));
  (*aux_sol)->num_aux_vec = n;
  (*aux_sol)->num_vec = n;
  (*aux_sol)->aux_h = (PASE_REAL*)calloc(n*n, sizeof(PASE_REAL));
  return 0;
}

//释放辅助矩阵空间，但并不处理具体的大规模空间
PASE_INT
PASE_MultiVector_destroy_sub(PASE_MultiVector *aux_sol)
{
  free((*aux_sol)->aux_h);
  (*aux_sol)->aux_h = NULL;
  free(*aux_sol);
  *aux_sol = NULL;
  return 0;
}

void 
PASE_PrintDenseMat(PASE_REAL *aux_hh, PASE_INT n)
{
  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      GCGE_Printf("aux_h(%d, %d) = %18.15e\n", i, j, aux_hh[i*n+j]);
    }
  }
}

PASE_INT
PASE_Aux_direct_solve(PASE_MG_SOLVER solver, PASE_INT coarse_level)
{
  clock_t start, end;
  start = clock();

  //我们设置的pase_nev等于max_nev，如果用户只给了nev个初值，那要在第一次Aux_direct求解时，
  //将pase_nev重赋值为max_nev
  solver->pase_nev = solver->max_nev;
  PASE_INT num_unlock = solver->pase_nev - solver->nconv;

  GCGE_SOLVER *gcge_pase_solver = GCGE_SOLVER_PASE_Create(
        solver->aux_A, solver->aux_B, solver->pase_nev, 
        solver->aux_eigenvalues, solver->aux_sol, solver->pase_ops, num_unlock);
  //gcg直接求解的精度比atol稍高一些
  gcge_pase_solver->para->print_para        = 0;
  gcge_pase_solver->para->print_conv        = 0;
  gcge_pase_solver->para->print_eval        = 0;
  gcge_pase_solver->para->print_result      = 0;
  gcge_pase_solver->para->ev_tol            = solver->aux_rtol;
  gcge_pase_solver->para->ev_max_it         = solver->max_direct_count_each_level[coarse_level];
  gcge_pase_solver->para->num_init_evec     = solver->pase_nev;
  gcge_pase_solver->para->opt_rr_eig_partly = 1;
  //gcge_pase_solver->para->cg_max_it = 50;
  //gcge_pase_solver->para->orth_para->orth_zero_tol = 1e-13;
  //gcge_pase_solver->para->p_orth_type = "gs";
  gcge_pase_solver->para->x_orth_type = "multi";
  //求解
  GCGE_SOLVER_Solve(gcge_pase_solver);  
  //解完后选择每个细空间向量方向值最大的向量，重新排序
  //出现内存错误，主要是multigrid->double_tmp的大小不够
#if 0 
  if(solver->nconv > 0) {
    PASE_Aux_sol_sort(solver, coarse_level);
  } 
#endif
  memcpy(solver->eigenvalues+solver->nconv, solver->aux_eigenvalues+solver->nconv, 
        num_unlock*sizeof(PASE_REAL));
  //释放空间
  GCGE_SOLVER_Free(&gcge_pase_solver);

  end = clock();
  solver->aux_direct_solve_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

PASE_INT 
PASE_Aux_sol_sort(PASE_MG_SOLVER solver, PASE_INT coarse_level)
{
  PASE_INT  nconv       = solver->nconv;
  PASE_INT  pase_nev    = solver->pase_nev;
  PASE_REAL *aux_h      = solver->aux_sol->aux_h;
  PASE_INT  num_aux_vec = solver->aux_sol->num_aux_vec;
  void      **vecs_tmp  = solver->cg_res[coarse_level];
  PASE_INT  i           = 0;
  PASE_INT  j           = 0;
  GCGE_OPS  *gcge_ops   = solver->gcge_ops;
  PASE_INT  position    = nconv;
  PASE_REAL last_eval_nconv = solver->eigenvalues[nconv-1];
  while((solver->aux_eigenvalues[position]>last_eval_nconv)&&(position>0)) {  
    GCGE_Printf("aux_eigenvalues[%d] = %18.15e, last_eval_nconv: %18.15e\n", 
	  position, solver->aux_eigenvalues[position], last_eval_nconv);
    position--;
  }
  PASE_INT  unlock_start = position;
  PASE_INT  mv_s[2];
  PASE_INT  mv_e[2];
  PASE_MultiVector aux_tmp = (PASE_MultiVector)malloc(sizeof(pase_MultiVector));
  aux_tmp->num_aux_vec = num_aux_vec;
  aux_tmp->num_vec     = num_aux_vec;
  aux_tmp->b_H         = solver->aux_B->aux_Hh;
  aux_tmp->aux_h       = solver->aux_B->aux_hh;
  //取aux_sol的unlock_start:pase_nev-1
  //u_h的unlock_start:pase_nev-1
  mv_s[0] = unlock_start;
  mv_e[0] = pase_nev;
  mv_s[1] = unlock_start;
  mv_e[1] = num_aux_vec;
  PASE_REAL *inner_prod = solver->cg_double_tmp;
  PASE_INT  ldi = mv_e[0] - mv_s[0];
  solver->pase_ops->MultiVecInnerProd((void**)(solver->aux_sol), (void**)aux_tmp, 
	inner_prod, "nonsym", mv_s, mv_e, ldi, 0, solver->pase_ops);

#if 1
  //inner_prod是一个方阵，行列+unlock_start后对应真正的特征对
  PASE_INT  *max_idx = solver->cg_int_tmp; 
  for(i=0; i<mv_e[1]-mv_s[1]; i++) {
    for(j=0; j<i; j++) {
      inner_prod[i*ldi+max_idx[j]] = 0.0;
    }
    PASE_Find_max_in_vector(max_idx+i, inner_prod+i*ldi, 0, ldi);
  }
  PASE_Sort_int(max_idx, 0, mv_e[1]-mv_s[1]);
  position = pase_nev-1;
  void *from_vec;
  void *to_vec;
  PASE_REAL *eigenvalues     = solver->eigenvalues;
  PASE_REAL *aux_eigenvalues = solver->aux_eigenvalues;
  PASE_INT   current_idx = 0;
  GCGE_Printf("num_aux_vec: %d, pase_nev: %d, position: %d\n", 
	num_aux_vec, pase_nev, position);
  for(i=num_aux_vec-1; i>=nconv; i--) {
    current_idx = max_idx[i-unlock_start]+unlock_start;
    GCGE_Printf("unlock_start: %d, max_idx[%d] = %d, current_idx: %d, position: %d\n", 
	  unlock_start, i, max_idx[i], current_idx, position);
    if(current_idx != position) {
      solver->pase_ops->GetVecFromMultiVec((void**)solver->aux_sol, current_idx, &from_vec, solver->pase_ops);
      solver->pase_ops->GetVecFromMultiVec((void**)solver->aux_sol, position, &to_vec, solver->pase_ops);
      solver->pase_ops->VecAxpby(1.0, from_vec, 0.0, to_vec, solver->pase_ops);
      solver->pase_ops->RestoreVecForMultiVec((void**)solver->aux_sol, current_idx, &from_vec, solver->pase_ops);
      solver->pase_ops->RestoreVecForMultiVec((void**)solver->aux_sol, position, &to_vec, solver->pase_ops);
      aux_eigenvalues[position] = aux_eigenvalues[current_idx];
      GCGE_Printf("sort, nconv: %d, unlock_start: %d, current_idx: %d, eigenvalues[%d] = %18.15lf\n", 
	    nconv, unlock_start, current_idx, position, eigenvalues[position]);
    }
    position--;
  }
#endif

  free(aux_tmp); aux_tmp = NULL;
}

PASE_INT 
PASE_Sort_int(PASE_INT *a, PASE_INT left, PASE_INT right)
{
    PASE_INT i = 0; 
    PASE_INT j = 0;
    PASE_INT temp = 0;
    i = left;
    j = right;
    temp = a[left];
    if(left > right)
        return 0;
    while(i != j)
    {
        while((a[j] >= temp)&&(j > i))
            j--;
        if(j > i)
            a[i++] = a[j];
        while((a[i] <= temp)&&(j > i))
            i++;
        if(j > i)
            a[j--] = a[i];
    }
    a[i] = temp;
    PASE_Sort_int(a, left, i-1);
    PASE_Sort_int(a, i+1, right);
    return 0;
}

PASE_INT 
PASE_Find_max_in_vector(PASE_INT *max_idx, PASE_REAL *vector, PASE_INT start, PASE_INT end)
{
  PASE_REAL max_value = fabs(vector[start]);
  PASE_INT  max_index = start;
  PASE_REAL tmp_value = 0.0;
  PASE_INT  i = start;
  //对每一列进行循环
  for(i=start+1; i<end; i++) {
    tmp_value = fabs(vector[i]);
    //如果该列的第一行比最大值大，重新确定最大值及位置
    if(tmp_value > max_value) {
      max_index = i;
      max_value = tmp_value;
    }
  }
  *max_idx = max_index;
}

PASE_INT
PASE_Mg_prolong_from_pase_aux_vector(PASE_MG_SOLVER solver,
      PASE_INT coarse_level, PASE_INT current_level)
{
  clock_t start, end;
  start = clock();

  PASE_MultiVector aux_sol = solver->aux_sol;
  //rhs此时用来存储P*u_H
  void **P_uH = solver->rhs[current_level];
  void **sol = solver->sol[current_level];
  PASE_INT mv_s[2];
  PASE_INT mv_e[2];
  PASE_INT nlock_direct = solver->nlock_direct;
  PASE_INT pase_nev = solver->pase_nev;
  PASE_INT num_aux_vec = aux_sol->num_aux_vec;
  PASE_INT nconv = solver->nconv;
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = nconv;
  mv_e[1] = pase_nev;
  //从aux_sol_i->b_H延拓到u_j(先放在u_tmp)
  //同时aux_sol_i->aux_h作为线性组合系数，对原u_j进行线性组合
  PASE_INT error = PASE_MULTIGRID_FromItoJ(solver->multigrid, 
        coarse_level, current_level, mv_s, mv_e, aux_sol->b_H, P_uH);
  //线性组合时，alpha=beta=1.0
  //线性组合的基底是所有参与构造复合矩阵的细空间向量
  mv_s[0] = nlock_direct;
  mv_e[0] = pase_nev;
  mv_s[1] = nconv;
  mv_e[1] = pase_nev;
  solver->gcge_ops->MultiVecLinearComb(sol, P_uH, mv_s, mv_e,
        aux_sol->aux_h+nconv*num_aux_vec, num_aux_vec, 0, 1.0, 1.0, solver->gcge_ops);
  mv_s[0] = nconv;
  mv_e[0] = pase_nev;
  mv_s[1] = nconv;
  mv_e[1] = pase_nev;
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
  PASE_REAL      *eigenvalues   = solver->eigenvalues;
  void          **rhs           = solver->rhs[current_level];
  PASE_INT        pase_nev      = solver->pase_nev;
  PASE_INT        nlock_smooth  = solver->nlock_smooth;

  GCGE_OPS *gcge_ops = solver->gcge_ops;
  PASE_OPS *pase_ops = solver->pase_ops;
  PASE_INT i = 0;
 
  //GCGE_Printf("nlock_smooth: %d, pase_nev: %d\n", nlock_smooth, pase_nev);
  PASE_INT mv_s[5];
  PASE_INT mv_e[5];
  //求解 A * u = eig * B * u
  //计算 rhs = eig * B * u
  mv_s[0] = nlock_smooth;
  mv_e[0] = pase_nev;
  mv_s[1] = nlock_smooth;
  mv_e[1] = pase_nev;
#if 0
  GCGE_Printf("line 1040, sol: %p, rhs: %p\n", sol, rhs);
  void *tmp;
  gcge_ops->GetVecFromMultiVec(solver->rhs[0], 0, &tmp, gcge_ops);
  GCGE_Printf("line 1042\n");
  gcge_ops->RestoreVecForMultiVec(solver->rhs[0], 0, &tmp, gcge_ops);
  gcge_ops->GetVecFromMultiVec(rhs, 0, &tmp, gcge_ops);
  gcge_ops->RestoreVecForMultiVec(rhs, 0, &tmp, gcge_ops);
  GCGE_Printf("line 1044\n");
#endif
  gcge_ops->MatDotMultiVec(B, sol, rhs, mv_s, mv_e, gcge_ops);
  for(i=nlock_smooth; i<pase_nev; i++) {
      gcge_ops->MultiVecAxpbyColumn(0.0, rhs, i, eigenvalues[i], 
              rhs, i, gcge_ops);
  }
  //GCGE_Printf("line 1045\n");

  PASE_REAL tol = solver->atol;
  PASE_REAL cg_rate = 1e-2;
  PASE_INT  max_coarest_nsmooth = 10*max_iter;
  mv_s[0] = nlock_smooth;
  mv_e[0] = pase_nev;
  mv_s[1] = nlock_smooth;
  mv_e[1] = pase_nev;
  mv_s[2] = nlock_smooth;
  mv_e[2] = pase_nev;
  mv_s[3] = nlock_smooth;
  mv_e[3] = pase_nev;
  mv_s[4] = nlock_smooth;
  mv_e[4] = pase_nev;

  //GCGE_Printf("line 1061\n");
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
PASE_Mg_error_estimate(PASE_MG_SOLVER solver, PASE_INT idx_level, 
      PASE_INT idx_cycle, PASE_REAL cycle_time)
{
  PASE_INT         pase_nev    = solver->pase_nev; 
  PASE_REAL        atol        = solver->atol;
  PASE_REAL        rtol        = solver->rtol;
  void           **sol         = solver->sol[idx_level];
  void           **rhs         = solver->rhs[idx_level];
  PASE_REAL       *eigenvalues = solver->eigenvalues;
  void            *A           = solver->multigrid->A_array[idx_level];
  void            *B           = solver->multigrid->B_array[idx_level];

  /* 计算最细层的残差：r = Au - kMu */
  PASE_INT         flag        = 0;
  PASE_REAL       *check_multi = (PASE_REAL*)PASE_Malloc((pase_nev-1)*sizeof(PASE_REAL));
  PASE_INT         i           = 0;
  PASE_REAL        r_norm      = 1e+5;
  PASE_REAL        u_norm      = 0.0;
  PASE_REAL        xTAx        = 0.0;
  PASE_REAL        xTBx        = 0.0;
  void            *Au;
  void            *Bu;
  void            *ui;
  //nconv表示这次判断收敛性前已收敛的特征值个数
  PASE_INT         nconv         = solver->nconv;
  PASE_REAL        residual_for_old_nonconv = 0.0;
  PASE_REAL        new_nonconv_residual     = 0.0;

  GCGE_OPS *gcge_ops = solver->gcge_ops;
  for(i = nconv; i < pase_nev; ++i) {
    //GCGE_Printf("line 947, nconv: %d, pase_nev: %d, i: %d\n", nconv, pase_nev, i);
    gcge_ops->GetVecFromMultiVec(rhs, nconv, &Bu, gcge_ops);
    gcge_ops->GetVecFromMultiVec(sol, i, &ui, gcge_ops);
    //Bu = B * sol[i]
    gcge_ops->MatDotVec(B, ui, Bu, gcge_ops);
    gcge_ops->VecInnerProd(Bu, ui, &u_norm, gcge_ops);
    //u_norm = || sol[i] ||_B
    u_norm = sqrt(u_norm);
    gcge_ops->RestoreVecForMultiVec(rhs, nconv, &Bu, gcge_ops);
    //计算 || Au - \lambda Bu ||_2
    gcge_ops->GetVecFromMultiVec(rhs, nconv+1, &Au, gcge_ops);
    //Au = A * sol[i]
    gcge_ops->MatDotVec(A, ui, Au, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(sol, i, &ui, gcge_ops);
    gcge_ops->GetVecFromMultiVec(rhs, nconv, &Bu, gcge_ops);
    //Bu = Au - eval[i] * Bu
    gcge_ops->VecAxpby(1.0, Au, -eigenvalues[i], Bu, gcge_ops);
    //r_norm = || Bu ||_2
    gcge_ops->VecInnerProd(Bu, Bu, &r_norm, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(rhs, nconv, &Bu, gcge_ops);
    gcge_ops->RestoreVecForMultiVec(rhs, nconv+1, &Au, gcge_ops);
    //r_norm = || Au - \lambda Bu ||_2 / || ui ||_B
    r_norm = sqrt(r_norm)/u_norm;
    solver->abs_res_norm[i] = r_norm;
    if(solver->print_level > 2) {
      GCGE_Printf("i: %d, eval: %18.15e, r_norm: %18.15e\n", i, eigenvalues[i], r_norm);
    }

    if(i+1 < pase_nev) {
      check_multi[i] = fabs((eigenvalues[i]-eigenvalues[i+1])/eigenvalues[i]);
    }
    if(r_norm < atol || (r_norm/eigenvalues[i]) < rtol) {
      if(0 == flag) {
        //计算(新的)未收敛的残差和
        //solver->nconv++;
        solver->nconv = i+1;
      }
    } else {
      /* break; */
      flag = 1;
    }
  }
  if ( (solver->nconv-1)>=0 && (solver->nconv-1)<(pase_nev-1) )
  {
     while(check_multi[solver->nconv-1] < 1e-5 && solver->nconv > nconv) {
	solver->nconv--;
	if ( (solver->nconv-1)<0 && (solver->nconv-1)>=(pase_nev-1) )
	{
	   break;
	}
     }
  }
  //如果需要检查效率, 计算残差和
  if(solver->check_efficiency_flag == 1) {
    PASE_REAL sum_residual = 0.0;
    //计算(旧的)未收敛的残差和
    for(i = nconv; i < pase_nev; ++i) {
      sum_residual += solver->abs_res_norm[i];
    }
    solver->conv_efficiency[solver->aux_coarse_level] = -cycle_time/log(sum_residual);
    //GCGE_Printf("conv_efficiency[%d] = %e\n", solver->aux_coarse_level, 
    //	  solver->conv_efficiency[solver->aux_coarse_level]);
  }
  /*
  //检查第一个为收敛的特征值与最后一个刚收敛的特征值是否有可能是重特征值，为保证之后的排序问题，需让重特征值同时在收敛的集合或未收敛的集合.
  while(solver->nconv > nconv && solver->nconv < pase_nev && check_multi[solver->nconv-1] < 1e-8) {
    solver->nconv--;
  }
  */
  free(check_multi); check_multi = NULL;
  solver->nlock_smooth = solver->nconv;
  solver->nlock_direct = 0;


  if(solver->print_level > 0) {
    //PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);    
    GCGE_Printf("idx_cycle = %d, aux_coarse_level: %d, nconv = %d, cycle_time: %f\n", idx_cycle+1, solver->aux_coarse_level, solver->nconv, cycle_time);
    if(solver->nconv < solver->pase_nev) {
      //GCGE_Printf("the first unconverged eigenvalues (residual) = %.8e (%1.6e)\n", 
      //      solver->eigenvalues[solver->nconv], solver->abs_res_norm[solver->nconv]);
    } else {
      GCGE_Printf("all the wanted eigenpairs have converged.\n");
    }
  }    
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
PASE_Mg_print_result(PASE_MG_SOLVER solver)
{
  PASE_INT idx_eigen = 0;
  if(solver->print_level > 1) {
    GCGE_Printf("\n");
    GCGE_Printf("=============================================================\n");
    for(idx_eigen=0; idx_eigen<solver->pase_nev; idx_eigen++) {
      GCGE_Printf("%3d-th eig=%.8e, abs_res = %.8e\n", idx_eigen, solver->eigenvalues[idx_eigen], solver->abs_res_norm[idx_eigen]);
    }
  }
  if(solver->print_level > 0) {
    GCGE_Printf("=============================================================\n");
    GCGE_Printf("initialize convert time = %f seconds\n", solver->initialize_convert_time);
    GCGE_Printf("initialize amg time     = %f seconds\n", solver->initialize_amg_time);
    GCGE_Printf("get initvec time        = %f seconds\n", solver->get_initvec_time);
    GCGE_Printf("smooth time             = %f seconds\n", solver->smooth_time);
    GCGE_Printf("build aux time          = %f seconds\n", solver->build_aux_time);
    GCGE_Printf("prolong time            = %f seconds\n", solver->prolong_time);
    GCGE_Printf("aux direct solve time   = %f seconds\n", solver->aux_direct_solve_time);
    GCGE_Printf("total solve time        = %f seconds\n", solver->total_solve_time);
    GCGE_Printf("total time              = %f seconds\n", solver->total_time);
    GCGE_Printf("=============================================================\n");
  }    
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
PASE_Mg_print_param(PASE_MG_SOLVER solver)
{
  if(solver->print_level > 0) {
    PASE_INT i = 0;
    PASE_INT num_levels = solver->num_levels;
    GCGE_Printf("\n");
    GCGE_Printf("===PARAMETERS===============================================\n");
    GCGE_Printf("nev                      = %d\n", solver->nev);
    GCGE_Printf("pase_nev                 = %d\n", solver->pase_nev);
    GCGE_Printf("max_nev                  = %d\n", solver->max_nev);
    GCGE_Printf("rtol                     = %e\n", solver->rtol);
    GCGE_Printf("atol                     = %e\n", solver->atol);
    GCGE_Printf("aux_rtol                 = %e\n", solver->aux_rtol);
    GCGE_Printf("initial_rtol             = %e\n", solver->initial_rtol);
    GCGE_Printf("num_levels               = %d\n", solver->num_levels);
    GCGE_Printf("initial_level            = %d\n", solver->initial_level);
    GCGE_Printf("num_given_eigs           = %d\n", solver->num_given_eigs);
    GCGE_Printf("mg_coarsest_level        = %d\n", solver->mg_coarsest_level);
    GCGE_Printf("initial_aux_coarse_level = %d\n", solver->initial_aux_coarse_level);
    GCGE_Printf("finest_aux_coarse_level  = %d\n", solver->finest_aux_coarse_level);
    GCGE_Printf("finest_level             = %d\n", solver->finest_level);
    GCGE_Printf("max_initial_direct_count = %d\n", solver->max_initial_direct_count);
    GCGE_Printf("from finest to coarest:\n");
    GCGE_Printf("max_cycle_count_each_level  = ");
    for(i=1; i<num_levels; i++) {
      GCGE_Printf("%3d, ", solver->max_cycle_count_each_level[i]);
    }
    GCGE_Printf("\n");
    GCGE_Printf("max_pre_count_each_level    = ");
    for(i=1; i<num_levels; i++) {
      GCGE_Printf("%3d, ", solver->max_pre_count_each_level[i]);
    }
    GCGE_Printf("\n");
    GCGE_Printf("max_post_count_each_level   = ");
    for(i=1; i<num_levels; i++) {
      GCGE_Printf("%3d, ", solver->max_post_count_each_level[i]);
    }
    GCGE_Printf("\n");
    GCGE_Printf("max_direct_count_each_level = ");
    for(i=1; i<num_levels; i++) {
      GCGE_Printf("%3d, ", solver->max_direct_count_each_level[i]);
    }
    GCGE_Printf("\n");
    GCGE_Printf("=============================================================\n");
  }    
  return 0;
}

