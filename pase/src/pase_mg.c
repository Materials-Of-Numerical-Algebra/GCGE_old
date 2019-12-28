#include <time.h>
#include "pase_mg.h"

void GetMultigridMatFromHypreToPetsc(Mat **A_array, Mat **P_array, 
      HYPRE_Int *num_levels, HYPRE_ParCSRMatrix hypre_parcsr_mat, 
      GCGE_DOUBLE *convert_time, GCGE_DOUBLE *amg_time);
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
GCGE_INT 
PASE_MULTIGRID_Create(PASE_MULTIGRID* multi_grid, 
        GCGE_INT max_levels, GCGE_INT mg_coarsest_level, 
        void *A, void *B, GCGE_OPS *gcge_ops,
	GCGE_DOUBLE *convert_time, GCGE_DOUBLE *amg_time)
{
    /* P 是行多列少, P*v是从粗到细 */
    *multi_grid = (PASE_MULTIGRID)PASE_Malloc(sizeof(pase_MultiGrid));
    (*multi_grid)->num_levels = max_levels;
    (*multi_grid)->coarsest_level = mg_coarsest_level;
    (*multi_grid)->gcge_ops = gcge_ops;
    (*multi_grid)->A_array = NULL;
    (*multi_grid)->B_array = NULL;
    (*multi_grid)->P_array = NULL;

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

    (*multi_grid)->A_array = (void**)A_array;
    (*multi_grid)->B_array = (void**)B_array;
    (*multi_grid)->P_array = (void**)P_array;
    return 0;
}

GCGE_INT 
PASE_MULTIGRID_Destroy(PASE_MULTIGRID* multi_grid)
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
    free((*multi_grid)->A_array);
    free((*multi_grid)->B_array);
    free((*multi_grid)->P_array);
    (*multi_grid)->A_array = NULL;
    (*multi_grid)->B_array = NULL;
    (*multi_grid)->P_array = NULL;

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
GCGE_INT 
PASE_MULTIGRID_FromItoJ(PASE_MULTIGRID multi_grid, 
        GCGE_INT level_i, GCGE_INT level_j, 
        GCGE_INT *mv_s, GCGE_INT *mv_e, 
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
    GCGE_INT k = 0;
    GCGE_INT start[2];
    GCGE_INT end[2];
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
      GCGE_DOUBLE *convert_time, GCGE_DOUBLE *amg_time)
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
    partitioning = hypre_CTAlloc(HYPRE_Int,  2);
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
