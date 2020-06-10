#include <time.h>
#include "pase_mg.h"
/* ------------------------liyu 20200529 修改部分 ---------------------------- */
void RedistributeDataOfMultiGridMatrixOnEachProcess(
     Mat * petsc_A_array, Mat *petsc_B_array, Mat *petsc_P_array, 
     PetscInt num_levels, PetscReal *proc_rate, PetscInt unit);
/* ------------------------liyu 20200529 修改部分 ---------------------------- */

void GetMultigridMatFromHypreToPetsc(Mat **A_array, Mat **P_array, 
      HYPRE_Int *num_levels, HYPRE_ParCSRMatrix hypre_parcsr_mat, 
      PASE_REAL *convert_time, PASE_REAL *amg_time);
/**
 * @brief 创建 PASE_MULTIGRID
 *
 * @param A      输入参数
 * @param B      输入参数
 * @param param  输入参数, 包含 AMG 分层的各个参数
 * @param ops    输入参数, 多重网格操作集合
 *
 * @return PASE_MULTIGRID
 */
PASE_INT 
PASE_MULTIGRID_Create(PASE_MULTIGRID* multi_grid, 
        PASE_INT max_levels, PASE_INT mg_coarsest_level, 
        PASE_INT **size, PASE_INT size_dtmp, PASE_INT size_itmp,
        void *A, void *B, GCGE_OPS *gcge_ops,
	PASE_REAL *convert_time, PASE_REAL *amg_time)
{
    /* P 是行多列少, P*v是从粗到细 */
    *multi_grid = (PASE_MULTIGRID)PASE_Malloc(sizeof(pase_MultiGrid));
    (*multi_grid)->num_levels = max_levels;
    (*multi_grid)->coarsest_level = mg_coarsest_level;
    (*multi_grid)->gcge_ops = gcge_ops;
    (*multi_grid)->A_array = NULL;
    (*multi_grid)->B_array = NULL;
    (*multi_grid)->P_array = NULL;

#if 1
    /* --------------------------------------- JUST for PETSC using its amg --------------------------------------- */
    PetscErrorCode ierr;
    PetscInt m, n, level;
    Mat petsc_A, petsc_B;
    Mat *A_array = NULL, *B_array = NULL, *P_array = NULL;
    petsc_A = (Mat)A; petsc_B = (Mat)B;

    *convert_time += 0;

    PC    pc;
    Mat   *Aarr=NULL, *Parr=NULL;

    ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRQ(ierr);
    ierr = PCSetOperators(pc,petsc_A,petsc_A);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCGAMG);CHKERRQ(ierr);
    //ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
    //ierr = PCMGSetLevels(pc, (*multi_grid)->num_levels, NULL);CHKERRQ(ierr);
    ierr = PCGAMGSetType(pc, PCGAMGCLASSICAL);CHKERRQ(ierr);
    //ierr = PCGAMGSetType(pc, PCGAMGGEO);CHKERRQ(ierr);
    //ierr = PCGAMGSetType(pc, PCGAMGAGG);CHKERRQ(ierr);
    //PCGAMGAGG, PCGAMGGEO, PCGAMGCLASSICAL
    PetscReal th[16] = {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 
                        0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
    ierr = PCGAMGSetNlevels(pc, (*multi_grid)->num_levels);CHKERRQ(ierr);
    ierr = PCGAMGSetThreshold(pc, th, 16);CHKERRQ(ierr);
    /* 最粗矩阵不限制在0进程 */
    ierr = PCGAMGSetUseParallelCoarseGridSolve(pc,PETSC_TRUE);CHKERRQ(ierr);
    /* this will generally improve the loading balancing of the work on each level */
    //ierr = PCGAMGSetRepartition(pc, PETSC_TRUE);CHKERRQ(ierr);
    /* GAMG will reduce the number of MPI processes used directly on the coarse grids 
     * so that there are around <limit> equations on each process that has degrees of freedom */
    //ierr = PCGAMGSetProcEqLim(pc, 1000);CHKERRQ(ierr);
    ierr = PCSetUp(pc);CHKERRQ(ierr);
    /* the size of Aarr is num_levels-1, Aarr is the coarsest matrix */
    ierr = PCGetCoarseOperators(pc, &((*multi_grid)->num_levels), &Aarr);
    /* the size of Parr is num_levels-1 */
    ierr = PCGetInterpolations(pc, &((*multi_grid)->num_levels), &Parr);
    //如果coarsest_level不能超过实际最粗层
    if((*multi_grid)->coarsest_level > (*multi_grid)->num_levels-1) {
       (*multi_grid)->coarsest_level = (*multi_grid)->num_levels-1;
    }

    /* we should make that zero is the refinest level */
    /* when num_levels == 5, 1 2 3 4 of A_array == 3 2 1 0 of Aarr */

    A_array = malloc(sizeof(Mat)*((*multi_grid)->num_levels));
    P_array = malloc(sizeof(Mat)*((*multi_grid)->num_levels-1));

    for (level = 1; level < (*multi_grid)->num_levels; ++level)
    {
       A_array[level] = Aarr[(*multi_grid)->num_levels-level-1];
       MatGetSize(A_array[level], &m, &n);
       PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", level, m, n );

       P_array[level-1] = Parr[(*multi_grid)->num_levels-level-1];
       MatGetSize(P_array[level-1], &m, &n);
       PetscPrintf(PETSC_COMM_WORLD, "P_array[%d], m = %d, n = %d\n", level-1, m, n );
    }
    ierr = PetscFree(Aarr);CHKERRQ(ierr);
    ierr = PetscFree(Parr);CHKERRQ(ierr);
    ierr = PCDestroy(&pc);CHKERRQ(ierr);

    //将原来的最细层A矩阵指针给A_array
    A_array[0] = petsc_A;
    B_array = malloc((*multi_grid)->num_levels*sizeof(Mat));
    B_array[0] = petsc_B;
    MatGetSize(B_array[0], &m, &n);
    PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", 0, m, n );
    /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
    for ( level = 1; level < (*multi_grid)->num_levels; ++level )
    {
       MatPtAP(B_array[level-1], P_array[level-1], 
	     MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(B_array[level]));
       MatGetSize(B_array[level], &m, &n);
       PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", level, m, n );
    }
    /* --------------------------------------- JUST for PETSC using its amg --------------------------------------- */
#else
    /* --------------------------------------- JUST for PETSC using HYPRE --------------------------------------- */
    Mat petsc_A, petsc_B;
    Mat *A_array, *B_array, *P_array;
    HYPRE_IJMatrix     hypre_ij_mat;
    HYPRE_ParCSRMatrix hypre_parcsr_mat;
    petsc_A = (Mat)A; petsc_B = (Mat)B;

    clock_t start, end;
    start = clock();
    MatrixConvertPETSC2HYPRE(&hypre_ij_mat, petsc_A);
    end = clock();
    *convert_time += ((double)(end-start))/CLK_TCK;

    HYPRE_IJMatrixGetObject(hypre_ij_mat, (void**) &hypre_parcsr_mat);
    /* Will malloc for A and P */
    //对输入的hypre格式的A矩阵进行AMG分层, 将得到的各层粗网格矩阵及prolong矩阵转化为petsc矩阵
    GetMultigridMatFromHypreToPetsc(&A_array, &P_array, &((*multi_grid)->num_levels), hypre_parcsr_mat, convert_time, amg_time);
    //将原来的最细层A矩阵指针给A_array
    A_array[0] = petsc_A;
    HYPRE_IJMatrixDestroy(hypre_ij_mat);
    B_array = malloc((*multi_grid)->num_levels*sizeof(Mat));
    int level; 
    B_array[0] = petsc_B;
    //MatDuplicate(petsc_B, MAT_COPY_VALUES, &B_array[0]);
    /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
    for ( level = 1; level < (*multi_grid)->num_levels; ++level )
    {
        MatPtAP(B_array[level-1], P_array[level-1], 
                MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(B_array[level]));
    }
    /* --------------------------------------- JUST for PETSC using HYPRE --------------------------------------- */
#endif

/* ------------------------liyu 20200529 修改部分 ---------------------------- */
    /* 重分配矩阵数据在进程中，默认只改变最粗层的分布
     * 即proc_rate[num_levels-1] = 0.5 使用一半的进程数 */
    /* 对整个进程, 重新分配各层矩阵数据 
     * 保证每层实际拥有数据的进程数nbigranks是unit的倍数(比如LSSC4 unit=36)
     * nbigranks[level] = size*proc_rate[level]
     * 进程号从0开始到nbigranks-1
     * proc_rate[0]是无效参数，即不改变最细层的矩阵分配 
     * 若proc_rate设为(0,1)之外，则不进行数据重分配 表示不会改变idx-1层的P和idx层的A */
    /* TODO
     * 这个函数形式参数max_levels就只能理解为最大层数，实际层数
     * (*multi_grid)->num_levels有可能比它小
     * 这会使得在param中的num_levels与真实的num_levels不等
     * 但是在创建pase_solver中，一些参数如initial_level会与param中的num_levels相关
     * 这是错误的设计*/
    GCGE_INT    unit = 1;
    GCGE_DOUBLE *proc_rate = malloc((*multi_grid)->num_levels*sizeof(GCGE_DOUBLE));
    for (level = 0; level < (*multi_grid)->num_levels; ++level)
    {
       proc_rate[level] = -1.0;
    }
//    proc_rate[(*multi_grid)->num_levels-1] = 0.5;
    //proc_rate[(*multi_grid)->num_levels-1] = 1e-8;
    //proc_rate[(*multi_grid)->num_levels-1] = 1.0;
    RedistributeDataOfMultiGridMatrixOnEachProcess(
	  A_array, B_array, P_array, 
	  (*multi_grid)->num_levels, proc_rate, unit);
    free(proc_rate);
/* ------------------------liyu 20200529 修改部分 ---------------------------- */

    (*multi_grid)->A_array = (void**)A_array;
    (*multi_grid)->B_array = (void**)B_array;
    (*multi_grid)->P_array = (void**)P_array;

    BV *u, *rhs, *u_tmp, *u_tmp_1, *u_tmp_2;
    u       = (BV*)malloc((*multi_grid)->num_levels*sizeof(BV));
    rhs     = (BV*)malloc((*multi_grid)->num_levels*sizeof(BV));
    u_tmp   = (BV*)malloc((*multi_grid)->num_levels*sizeof(BV));
    u_tmp_1 = (BV*)malloc((*multi_grid)->num_levels*sizeof(BV));
    u_tmp_2 = (BV*)malloc((*multi_grid)->num_levels*sizeof(BV));
    int i = 0;
    for(i=0; i<(*multi_grid)->num_levels; i++)
    {
       if (size[0][i]) 
	  gcge_ops->MultiVecCreateByMat((void***)(&(u[i])), size[0][i], 
		(*multi_grid)->A_array[i], gcge_ops);
       if (size[1][i]) 
	  gcge_ops->MultiVecCreateByMat((void***)(&(rhs[i])), size[1][i], 
		(*multi_grid)->A_array[i], gcge_ops);
       if (size[2][i]) 
	  gcge_ops->MultiVecCreateByMat((void***)(&(u_tmp[i])), size[2][i], 
		(*multi_grid)->A_array[i], gcge_ops);
       if (size[3][i]) 
	  gcge_ops->MultiVecCreateByMat((void***)(&(u_tmp_1[i])), size[3][i], 
		(*multi_grid)->A_array[i], gcge_ops);
       if (size[4][i]) 
	  gcge_ops->MultiVecCreateByMat((void***)(&(u_tmp_2[i])), size[4][i], 
		(*multi_grid)->A_array[i], gcge_ops);
    }
    double *double_tmp = (double*)calloc(size_dtmp, sizeof(double));
    int    *int_tmp    = (int*)calloc(size_itmp, sizeof(int));
    (*multi_grid)->sol        = (void***)u;
    (*multi_grid)->rhs        = (void***)rhs;
    (*multi_grid)->cg_p       = (void***)u_tmp;
    (*multi_grid)->cg_w       = (void***)u_tmp_1;
    (*multi_grid)->cg_res     = (void***)u_tmp_2;
    (*multi_grid)->cg_double_tmp = double_tmp;
    (*multi_grid)->cg_int_tmp    = int_tmp;

    PetscPrintf(PETSC_COMM_WORLD, "End PASE_MULTIGRID_Create\n", 0, m, n );
    return 0;
}

PASE_INT 
PASE_MULTIGRID_Destroy(PASE_MULTIGRID* multi_grid, PASE_INT **size)
{
    PetscErrorCode ierr;
    /* --------------------------------------- JUST for PETSC using HYPRE --------------------------------------- */
    Mat *A_array, *B_array, *P_array;
    A_array = (Mat *)(*((*multi_grid)->A_array));
    B_array = (Mat *)(*((*multi_grid)->B_array));
    P_array = (Mat *)(*((*multi_grid)->P_array));
    int level; 
    for ( level = 1; level < (*multi_grid)->num_levels; ++level )
    {
        ierr = MatDestroy((Mat*)(&((*multi_grid)->A_array[level])));
        ierr = MatDestroy((Mat*)(&((*multi_grid)->B_array[level])));
    }
    for ( level = 0; level < (*multi_grid)->num_levels - 1; ++level )
    {
        ierr = MatDestroy((Mat*)(&((*multi_grid)->P_array[level])));
    }
    /* --------------------------------------- JUST for PETSC using HYPRE --------------------------------------- */
    int i = 0;
    //释放空间
    void ***u, ***rhs, ***u_tmp, ***u_tmp_1, ***u_tmp_2;
    double *double_tmp;
    int *int_tmp;
    u          = (*multi_grid)->sol;
    rhs        = (*multi_grid)->rhs;
    u_tmp      = (*multi_grid)->cg_p;
    u_tmp_1    = (*multi_grid)->cg_w;
    u_tmp_2    = (*multi_grid)->cg_res;
    double_tmp = (*multi_grid)->cg_double_tmp;
    int_tmp    = (*multi_grid)->cg_int_tmp;    
    for(i=0; i<(*multi_grid)->num_levels; i++)
    {
        if (size[0][i])
            (*multi_grid)->gcge_ops->MultiVecDestroy((void***)(&(u[i])), size[0][i], (*multi_grid)->gcge_ops);
        if (size[1][i]) 
            (*multi_grid)->gcge_ops->MultiVecDestroy((void***)(&(rhs[i])), size[1][i], (*multi_grid)->gcge_ops);
        if (size[2][i])
            (*multi_grid)->gcge_ops->MultiVecDestroy((void***)(&(u_tmp[i])), size[2][i], (*multi_grid)->gcge_ops);
        if (size[3][i])
            (*multi_grid)->gcge_ops->MultiVecDestroy((void***)(&(u_tmp_1[i])), size[3][i], (*multi_grid)->gcge_ops);
        if (size[4][i])
            (*multi_grid)->gcge_ops->MultiVecDestroy((void***)(&(u_tmp_2[i])), size[4][i], (*multi_grid)->gcge_ops);
    }
    free(u); u = NULL;
    free(rhs); rhs = NULL;
    free(u_tmp); u_tmp = NULL;
    free(u_tmp_1); u_tmp_1 = NULL;
    free(u_tmp_2); u_tmp_2 = NULL;

    free(double_tmp); double_tmp = NULL;
    free(int_tmp);    int_tmp = NULL;

    free((*multi_grid)->A_array);
    free((*multi_grid)->B_array);
    free((*multi_grid)->P_array);
    (*multi_grid)->A_array = NULL;
    (*multi_grid)->B_array = NULL;
    (*multi_grid)->P_array = NULL;

    /* 释放各层矩阵的通讯域 */
    for(level=0; level<(*multi_grid)->num_levels; level++)
    {
       MPI_Comm_free(&PASE_MG_COMM[level][0]);
       if (PASE_MG_COMM[level][1] != MPI_COMM_NULL)
       {
	  MPI_Comm_free(&PASE_MG_COMM[level][1]);
       }
       if (PASE_MG_INTERCOMM[level] != MPI_COMM_NULL)
       {
	  MPI_Comm_free(&PASE_MG_INTERCOMM[level]);
       }
    }

    free(*multi_grid); *multi_grid = NULL;
}

/**
 * TODO
 * @brief 将多向量从第level_i层 prolong/restrict 到level_j层
 *
 * @param multigrid 多重网格结构
 * @param level_i   起始层
 * @param level_j   目标层
 * @param mv_s      多向量pvx_i与pvx_j的起始位置
 * @param mv_e      多向量pvx_i与pvx_j的终止位置
 * @param pvx_i     起始多向量
 * @param pvx_j     目标多向量
 *
 * @return 
 */
PASE_INT 
PASE_MULTIGRID_FromItoJ(PASE_MULTIGRID multi_grid, 
        PASE_INT level_i, PASE_INT level_j, 
        PASE_INT *mv_s, PASE_INT *mv_e, 
        void **pvx_i, void** pvx_j)
{
#if 0
    /* P 是行多列少, P*v是从粗到细 */
    if (level_i == level_j + 1) /* level_i > level_j : 从粗层到细层 */
    {
        multi_grid->gcge_ops->MatDotMultiVec(multi_grid->P_array[level_j], pvx_i, pvx_j, mv_s, mv_e, multi_grid->gcge_ops);
    }
    else if (level_i == level_j - 1) /* level_i < level_j : 从细层到粗层 */
    {
        /* OPS 中需要加入 矩阵转置乘以向量 */
        multi_grid->gcge_ops->MatTransposeDotMultiVec(multi_grid->P_array[level_i], pvx_i, pvx_j, mv_s, mv_e, multi_grid->gcge_ops);
    }
#endif
    void **from_vecs;
    void **to_vecs;
    PASE_INT k = 0;
    PASE_INT start[2];
    PASE_INT end[2];
    if(level_i > level_j)
    {
        //从粗层到细层，延拓，用P矩阵直接乘
        for(k=level_i; k>level_j; k--) {
            if(k == level_i) {
                from_vecs = pvx_i;
                start[0] = mv_s[0];
                end[0]   = mv_e[0];
            } else {
                from_vecs = multi_grid->cg_res[k];
                start[0] = 0;
                end[0]   = mv_e[0]-mv_s[0];
            }
            if(k == level_j+1) {
                to_vecs = pvx_j;
                start[1] = mv_s[1];
                end[1]   = mv_e[1];
            } else {
                to_vecs = multi_grid->cg_res[k-1];
                start[1] = 0;
                end[1]   = mv_e[0]-mv_s[0];
            }
            multi_grid->gcge_ops->MatDotMultiVec(multi_grid->P_array[k-1], 
                    from_vecs, to_vecs, start, end, multi_grid->gcge_ops);
        }
    }
    else if(level_i < level_j)
    {
        //从细层到粗层，限制，用P^T矩阵乘
        for(k=level_i; k<level_j; k++) {
            if(k == level_i) {
                from_vecs = pvx_i;
                start[0] = mv_s[0];
                end[0]   = mv_e[0];
            } else {
                from_vecs = multi_grid->cg_res[k];
                start[0] = 0;
                end[0]   = mv_e[0]-mv_s[0];
            }
            if(k == level_j-1) {
                to_vecs = pvx_j;
                start[1] = mv_s[1];
                end[1]   = mv_e[1];
            } else {
                to_vecs = multi_grid->cg_res[k+1];
                start[1] = 0;
                end[1]   = mv_e[0]-mv_s[0];
            }
            multi_grid->gcge_ops->MatTransposeDotMultiVec(multi_grid->P_array[k], 
                    from_vecs, to_vecs, start, end, multi_grid->gcge_ops);
        }
    }
    else if (level_i == level_j)
    {
        multi_grid->gcge_ops->MultiVecAxpby(1.0, pvx_i, 0.0, pvx_j, mv_s, mv_e, multi_grid->gcge_ops);
    }
    else /* 其它情况需要借助辅助向量，这个比较希望在 PASE_MULTIGRID 结构中在每一层中生成一组多向量 */
    {
        printf ( "NO SUPPORT\n" );
        return 1;
    }
    return 0;
}



/* -------------------------- 利用AMG生成各个层的矩阵------------------ */
//对输入的hypre矩阵进行AMG分层, 将得到的各层粗网格矩阵及prolong矩阵转化为petsc矩阵
//不产生A0最细层petsc矩阵
void GetMultigridMatFromHypreToPetsc(Mat **A_array, Mat **P_array, 
      HYPRE_Int *num_levels, HYPRE_ParCSRMatrix hypre_parcsr_mat, 
      PASE_REAL *convert_time, PASE_REAL *amg_time)
{
    clock_t start, end;
    start = clock();
    /* Create solver */
    HYPRE_Solver      amg;
    HYPRE_BoomerAMGCreate(&amg);
    /* Set some parameters (See Reference Manual for more parameters) */
    HYPRE_BoomerAMGSetPrintLevel (amg, 0);         /* print solve info + parameters */
    HYPRE_BoomerAMGSetInterpType (amg, 0);
    HYPRE_BoomerAMGSetPMaxElmts  (amg, 0);
    HYPRE_BoomerAMGSetCoarsenType(amg, 6);
    HYPRE_BoomerAMGSetMaxLevels  (amg, *num_levels);  /* maximum number of levels */
    /* Create a HYPRE_ParVector by HYPRE_ParCSRMatrix */
    MPI_Comm           comm         = hypre_ParCSRMatrixComm(hypre_parcsr_mat);
    HYPRE_Int          global_size  = hypre_ParCSRMatrixGlobalNumRows(hypre_parcsr_mat);
    HYPRE_Int         *partitioning = NULL;
#ifdef HYPRE_NO_GLOBAL_PARTITION
    partitioning = hypre_CTAlloc(HYPRE_Int,  2, HYPRE_MEMORY_HOST);
    hypre_ParCSRMatrixGetLocalRange(hypre_parcsr_mat, partitioning, partitioning+1, partitioning, partitioning+1);
    partitioning[1] += 1;
#else
    HYPRE_ParCSRMatrixGetRowPartitioning(hypre_parcsr_mat, &partitioning);
#endif
    HYPRE_ParVector    hypre_par_vec = hypre_ParVectorCreate(comm, global_size, partitioning);
    hypre_ParVectorInitialize(hypre_par_vec);
    hypre_ParVectorSetPartitioningOwner(hypre_par_vec, 1);
    /* Now setup AMG */
    HYPRE_BoomerAMGSetup(amg, hypre_parcsr_mat, hypre_par_vec, hypre_par_vec);
    hypre_ParAMGData* amg_data = (hypre_ParAMGData*) amg;
    *num_levels = hypre_ParAMGDataNumLevels(amg_data);
    /* Create PETSC Mat */
    *A_array = malloc(sizeof(Mat)*(*num_levels));
    *P_array = malloc(sizeof(Mat)*(*num_levels-1));
    hypre_ParCSRMatrix **hypre_mat;
    HYPRE_Int idx;
    hypre_mat = hypre_ParAMGDataAArray(amg_data);
    GCGE_Printf("idx_level    global_num_rows\n");
    for (idx = 1; idx < *num_levels; ++idx)
    {
        GCGE_Printf("  %3d       %10d\n", idx, hypre_mat[idx]->global_num_rows);
        MatrixConvertHYPRE2PETSC((*A_array)+idx, hypre_mat[idx]);
    }
    hypre_mat = hypre_ParAMGDataPArray(amg_data);
    end = clock();
    *amg_time += ((double)(end-start))/CLK_TCK;
    start = clock();
    for (idx = 0; idx < *num_levels-1; ++idx)
    {
        MatrixConvertHYPRE2PETSC((*P_array)+idx, hypre_mat[idx]);
    }
    HYPRE_BoomerAMGDestroy(amg);	
    hypre_ParVectorDestroy(hypre_par_vec);
    end = clock();
    *convert_time += ((double)(end-start))/CLK_TCK;
}




/* ------------------------liyu 20200529 修改部分 ---------------------------- */
/**
 * @brief 
 *    nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
 *    if (nbigranks < unit) nbigranks = unit<size?unit:size;
 *
 * @param petsc_A_array
 * @param petsc_B_array
 * @param petsc_P_array
 * @param num_levels
 * @param proc_rate
 * @param unit           保证每层nbigranks是unit的倍数
 */
void RedistributeDataOfMultiGridMatrixOnEachProcess(
     Mat * petsc_A_array, Mat *petsc_B_array, Mat *petsc_P_array, 
     PetscInt num_levels, PetscReal *proc_rate, PetscInt unit)
{
   PetscMPIInt   rank, size;
//   PetscViewer   viewer;

   PetscInt      level, row;
   Mat           new_P_H;
   PetscMPIInt   nbigranks;
   PetscInt      global_nrows, global_ncols; 
   PetscInt      local_nrows,  local_ncols;
   PetscInt      new_local_ncols;
   /* 保证每层nbigranks是unit的倍数 */
   PetscInt      rstart, rend, ncols;
   const PetscInt              *cols; 
   const PetscScalar           *vals;

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_size(PETSC_COMM_WORLD, &size);

   if (proc_rate[0]<=1.0 && proc_rate[0]>0.0)
   {
      PetscPrintf(PETSC_COMM_WORLD, "Warning the refinest matrix cannot be redistributed\n");
   }

   /* 不改变最细层的进程分布 */
   MPI_Comm_dup( PETSC_COMM_WORLD, &PASE_MG_COMM[0][0]);
   PASE_MG_COMM[0][1]    = MPI_COMM_NULL;
   PASE_MG_INTERCOMM[0]  = MPI_COMM_NULL;
   PASE_MG_COMM_COLOR[0] = 0;
   for (level = 1; level < num_levels; ++level)
   {
      MatGetSize(petsc_P_array[level-1], &global_nrows, &global_ncols);
      /* 在设定new_P_H的局部行时已经不能用以前P的局部行，因为当前层的A可能已经改变 */
      MatGetLocalSize(petsc_A_array[level-1], &local_nrows, &local_ncols);
      /* 应该通过ncols_P，即最粗层矩阵大小和进程总数size确定nbigranks */
      nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
      if (nbigranks < unit) nbigranks = unit<size?unit:size;

      /* 若proc_rate设为(0,1)之外，则不进行数据重分配 */
      if (proc_rate[level]>1.0 || proc_rate[level]<=0.0 || nbigranks >= size || nbigranks <= 0)
      {
	 PetscPrintf(PETSC_COMM_WORLD, "Retain data distribution of %D level\n", level);
	 /* 创建分层矩阵的通信域 */
	 PASE_MG_COMM_COLOR[level] = 0;
	 MPI_Comm_dup(PETSC_COMM_WORLD, &PASE_MG_COMM[level][0]);
	 PASE_MG_COMM[level][1]   = MPI_COMM_NULL;
	 PASE_MG_INTERCOMM[level] = MPI_COMM_NULL;
	 continue; /* 直接到下一次循环 */
      }
      else
      {
	 PetscPrintf(PETSC_COMM_WORLD, "Redistribute data of %D level\n", level);
	 PetscPrintf(PETSC_COMM_WORLD, "nbigranks[%D] = %D\n", level, nbigranks);
      }
      /* 上面的判断保证 0 < nbigranks < size */

      /* 创建分层矩阵的通信域 */
      int comm_color, local_leader, remote_leader;
      /* 对0到nbigranks-1进程平均分配global_ncols */
      new_local_ncols = 0;
      if (rank < nbigranks)
      {
	 new_local_ncols = global_ncols/nbigranks;
	 if (rank < global_ncols%nbigranks)
	 {
	    ++new_local_ncols;
	 }
	 comm_color    = 0;
	 local_leader  = 0;
	 remote_leader = nbigranks;
      }
      else {
	 comm_color    = 1;
	 local_leader  = 0; /* 它的全局进程号是nbigranks */
	 remote_leader = 0;
      }
      /* 在不同进程中PASE_MG_COMM_COLOR[level]是不一样的值，它表征该进程属于哪个通讯域 */
      PASE_MG_COMM_COLOR[level] = comm_color;
      /* 分成两个子通讯域, PASE_MG_COMM[level][0]从0~(nbigranks-1)
       * PASE_MG_COMM[level][0]从nbigranks~(size-1) */
      MPI_Comm_split(PETSC_COMM_WORLD, comm_color, rank, &PASE_MG_COMM[level][comm_color]);
      MPI_Intercomm_create(PASE_MG_COMM[level][comm_color], local_leader, 
	    MPI_COMM_WORLD, remote_leader, level, &PASE_MG_INTERCOMM[level]);

      int aux_size = -1, aux_rank = -1;
      MPI_Comm_rank(PASE_MG_COMM[level][comm_color], &aux_rank);
      MPI_Comm_size(PASE_MG_COMM[level][comm_color], &aux_size);
      printf("aux %d/%d, global %d/%d\n", aux_rank, aux_size, rank, size);  

      /* 创建新的延拓矩阵, 并用原始的P为之赋值 */
      MatCreate(PETSC_COMM_WORLD, &new_P_H);
      MatSetSizes(new_P_H, local_nrows, new_local_ncols, global_nrows, global_ncols);
      //MatSetFromOptions(new_P_H);
      /* can be improved */
      //       MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
      //       MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
      MatSetUp(new_P_H);
      MatGetOwnershipRange(petsc_P_array[level-1], &rstart, &rend);
      for(row = rstart; row < rend; ++row) 
      {
	 MatGetRow(petsc_P_array[level-1], row, &ncols, &cols, &vals);
	 MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
	 MatRestoreRow(petsc_P_array[level-1], row, &ncols, &cols, &vals);
      }
      MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);

      MatGetLocalSize(petsc_P_array[level-1], &local_nrows, &local_ncols);
      PetscPrintf(PETSC_COMM_SELF, "[%D] original P_H[%D] local size %D * %D\n", 
	    rank, level, local_nrows, local_ncols);
      MatGetLocalSize(new_P_H, &local_nrows, &local_ncols);
      PetscPrintf(PETSC_COMM_SELF, "[%D] new P_H[%D] local size %D * %D\n", 
	    rank, level, local_nrows, local_ncols);
      //       MatView(petsc_P_array[level-1], viewer);
      //       MatView(new_P_H, viewer);

      /* 销毁之前的P_H A_H B_H */
      MatDestroy(&(petsc_P_array[level-1]));
      MatDestroy(&(petsc_A_array[level]));
      if (petsc_B_array!=NULL)
      {
	 MatDestroy(&(petsc_B_array[level]));
      }

      petsc_P_array[level-1] = new_P_H;
      MatPtAP(petsc_A_array[level-1], petsc_P_array[level-1],
	    MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_A_array[level]));
      if (petsc_B_array!=NULL)
      {
	 MatPtAP(petsc_B_array[level-1], petsc_P_array[level-1],
	       MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
      }
      //       MatView(petsc_A_array[num_levels-1], viewer);
      //       MatView(petsc_B_array[num_levels-1], viewer);

      /* 这里需要修改petsc_P_array[level], 原因是
       * petsc_A_array[level]修改后，
       * 它利用原来的petsc_P_array[level]插值上来的向量已经与petsc_A_array[level]不匹配
       * 所以在不修改level+1层的分布结构的情况下，需要对petsc_P_array[level]进行修改 */
      /* 如果当前层不是最粗层，并且，下一层也不进行数据重分配 */
      if (level+1<num_levels && (proc_rate[level+1]>1.0 || proc_rate[level+1]<=0.0) )
      {
	 MatGetSize(petsc_P_array[level], &global_nrows, &global_ncols);
	 /*需要当前层A的列 作为P的行 */
	 MatGetLocalSize(petsc_A_array[level],   &new_local_ncols, &local_ncols);
	 /*需要下一层A的行 作为P的列 */
	 MatGetLocalSize(petsc_A_array[level+1], &local_nrows, &new_local_ncols);
	 /* 创建新的延拓矩阵, 并用原始的P为之赋值 */
	 MatCreate(PETSC_COMM_WORLD, &new_P_H);
	 MatSetSizes(new_P_H, local_ncols, local_nrows, global_nrows, global_ncols);
	 //MatSetFromOptions(new_P_H);
	 /* can be improved */
	 //       MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
	 //       MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
	 MatSetUp(new_P_H);
	 MatGetOwnershipRange(petsc_P_array[level], &rstart, &rend);
	 for(row = rstart; row < rend; ++row) 
	 {
	    MatGetRow(petsc_P_array[level], row, &ncols, &cols, &vals);
	    MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
	    MatRestoreRow(petsc_P_array[level], row, &ncols, &cols, &vals);
	 }
	 MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
	 MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);

	 MatDestroy(&(petsc_P_array[level]));
	 petsc_P_array[level] = new_P_H;
      }
   }
}
/* ------------------------liyu 20200529 修改部分 ---------------------------- */
