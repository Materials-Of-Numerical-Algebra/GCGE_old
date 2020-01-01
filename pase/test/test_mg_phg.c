/*
 * =====================================================================================
 *
 *       Filename:  test_mg.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年12月26日 15时50分13秒
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include "gcge.h"
#include "pase.h"
#include "gcge_app_slepc.h"

/* 这两个函数在get_mat_phg.c中 */
int CreateMatrixPHG (void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);
int DestroyMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);

static char help[] = "Test MultiGrid.\n";
void PETSCPrintMat(Mat A, char *name);
void PETSCPrintVec(Vec x);
void PETSCPrintBV (BV  x, char *name);
/* 
 *  Description:  测试PASE_MULTIGRID
 */
int
main ( int argc, char *argv[] )
{
    /* PetscInitialize */
    SlepcInitialize(&argc,&argv,(char*)0,help);
    PetscErrorCode ierr;

    PetscMPIInt    rank;
    PetscViewer    viewer;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    /* 测试矩阵声明 */
    Mat      petsc_mat_A, petsc_mat_B;

    /* 得到一个PHG矩阵, 并将之转换为PETSC矩阵 */
    void *phg_mat_A, *phg_mat_B, *phg_dof_U, *phg_map_M, *phg_grid_G;
    CreateMatrixPHG (&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
    MatrixConvertPHG2PETSC((void **)(&petsc_mat_A), &phg_mat_A);
    MatrixConvertPHG2PETSC((void **)(&petsc_mat_B), &phg_mat_B);
    DestroyMatrixPHG(&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);

    //MatView(petsc_mat_A, viewer);
    //MatView(petsc_mat_B, viewer);
    HYPRE_Int idx, num_levels = 3, mg_coarsest_level = 2;

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_SLEPC_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);
    //用gcge_ops创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);

    int num_vecs = 3;
    PASE_MULTIGRID multi_grid;
    PASE_REAL convert_time = 0.0;
    PASE_REAL amg_time = 0.0;
    int **size = (int**)malloc(5*sizeof(int*));
    int i = 0;
    int j = 0;
    for (i=0; i<5; i++) {
        size[i] = (int*)calloc(num_levels, sizeof(int));
        for (j=0; j<num_levels; j++) {
            size[i][j] = num_vecs;
        }
    }
    int size_dtmp = num_vecs*num_vecs;
    int size_itmp = num_vecs*num_vecs;
    PASE_MULTIGRID_Create(&multi_grid, num_levels, mg_coarsest_level, 
	  size, size_dtmp, size_itmp,
	  (void *)petsc_mat_A, (void *)petsc_mat_B, 
	  gcge_ops, &convert_time, &amg_time);

#if 0
    //先测试P与PT乘以单向量是否有问题
    Vec x;
    Vec y;
    gcge_ops->VecCreateByMat((void**)&x, multi_grid->A_array[0], gcge_ops);
    gcge_ops->VecCreateByMat((void**)&y, multi_grid->A_array[1], gcge_ops);
    gcge_ops->VecSetRandomValue((void*)x, gcge_ops);
    gcge_ops->VecSetRandomValue((void*)y, gcge_ops);
    gcge_ops->MatDotVec(multi_grid->P_array[0], (void*)y, (void*)x, gcge_ops);
    gcge_ops->MatTransposeDotVec(multi_grid->P_array[0], (void*)x, (void*)y, gcge_ops);
    gcge_ops->VecDestroy((void**)&x, gcge_ops);
    gcge_ops->VecDestroy((void**)&y, gcge_ops);

    PETSCPrintMat((Mat)(multi_grid->P_array[0]), "P0");
    PETSCPrintMat((Mat)(multi_grid->P_array[1]), "P1");
    PETSCPrintMat((Mat)(multi_grid->A_array[0]), "A0");
    PETSCPrintMat((Mat)(multi_grid->A_array[1]), "A1");
    PETSCPrintMat((Mat)(multi_grid->B_array[0]), "B0");
    PETSCPrintMat((Mat)(multi_grid->B_array[1]), "B1");
    //LSSC4上View类的函数都有点问题，包括MatView, BVView, KSPView
    //MatView((Mat)(multi_grid->P_array[0]), viewer);
    //MatView((Mat)(multi_grid->A_array[1]), viewer);
#else
    /* 打印各层A, B, P矩阵*/
    //连续打会打不出来，不知道为什么？？？？只打一个可以
    PetscPrintf(PETSC_COMM_WORLD, "A_array\n");
    for (idx = 0; idx < num_levels; ++idx)
    {
        PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
        MatView((Mat)(multi_grid->A_array[idx]), viewer);
    }
    PetscPrintf(PETSC_COMM_WORLD, "B_array\n");
    for (idx = 0; idx < num_levels; ++idx)
    {
        PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
        MatView((Mat)(multi_grid->B_array[idx]), viewer);
    }
    PetscPrintf(PETSC_COMM_WORLD, "P_array\n");
    for (idx = 0; idx < num_levels-1; ++idx)
    {
        PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
        MatView((Mat)(multi_grid->P_array[idx]), viewer);
    }
#endif

#if 1
    /* TODO: 测试BV结构的向量进行投影插值 */
    int level_i = 0;
    int level_j = 2;
    BV vecs_i;
    BV vecs_j;
    gcge_ops->MultiVecCreateByMat((void***)(&vecs_i), num_vecs, multi_grid->A_array[level_i], gcge_ops);
    gcge_ops->MultiVecCreateByMat((void***)(&vecs_j), num_vecs, multi_grid->A_array[level_j], gcge_ops);
    gcge_ops->MultiVecSetRandomValue((void**)vecs_i, 0, num_vecs, gcge_ops);
    gcge_ops->MultiVecSetRandomValue((void**)vecs_j, 0, num_vecs, gcge_ops);

    int mv_s[2];
    int mv_e[2];
    mv_s[0] = 0;
    mv_e[0] = num_vecs;
    mv_s[1] = 0;
    mv_e[1] = num_vecs;
    //gcge_ops->MatDotMultiVec(multi_grid->P_array[0], (void**)vecs_j, 
    //	  (void**)vecs_i, mv_s, mv_e, gcge_ops);
    //gcge_ops->MatTransposeDotMultiVec(multi_grid->P_array[0], (void**)vecs_i, 
    //	  (void**)vecs_j, mv_s, mv_e, gcge_ops);
    PASE_MULTIGRID_FromItoJ(multi_grid, level_j, level_i, mv_s, mv_e, 
            (void**)vecs_j, (void**)vecs_i);
    PETSCPrintBV(vecs_i, "Pto");
    PETSCPrintBV(vecs_j, "Pfrom");
    PASE_MULTIGRID_FromItoJ(multi_grid, level_i, level_j, mv_s, mv_e, 
            (void**)vecs_i, (void**)vecs_j);
    PETSCPrintBV(vecs_i, "Rfrom");
    PETSCPrintBV(vecs_j, "Rto");

    //GCGE_Printf("vecs_i\n");
    //BVView(vecs_i, viewer);
    //GCGE_Printf("vecs_j\n");
    //BVView(vecs_j, viewer);

    gcge_ops->MultiVecDestroy((void***)(&vecs_i), num_vecs, gcge_ops);
    gcge_ops->MultiVecDestroy((void***)(&vecs_j), num_vecs, gcge_ops);
#endif
    //注释掉Destroy不报错memory access out of range, 说明是Destroy时用错
  //  gcge_ops->MultiVecDestroy(&(multi_grid->cg_p[1]), 3, gcge_ops);
    PASE_MULTIGRID_Destroy(&multi_grid, size);
    for (i=0; i<5; i++){
        free(size[i]); size[i] = NULL;
    }
    free(size); size = NULL;

    PASE_OPS_Free(&pase_ops); 
    GCGE_OPS_Free(&gcge_ops);

    ierr = MatDestroy(&petsc_mat_A);
    ierr = MatDestroy(&petsc_mat_B);

    /* PetscFinalize */
    ierr = SlepcFinalize();
    return 0;
}

void PETSCPrintMat(Mat A, char *name)
{
    PetscErrorCode ierr;
    int nrows = 0, ncols = 0;
    ierr = MatGetSize(A, &nrows, &ncols);
    int row = 0, col = 0;
    const PetscInt *cols;
    const PetscScalar *vals;
    for(row=0; row<nrows; row++) {
        ierr = MatGetRow(A, row, &ncols, &cols, &vals);
        for(col=0; col<ncols; col++) {
            GCGE_Printf("%s(%d, %d) = %18.15e;\n", name, row+1, cols[col]+1, vals[col]);
        }
        ierr = MatRestoreRow(A, row, &ncols, &cols, &vals);
    }
}

void PETSCPrintVec(Vec x)
{
    PetscErrorCode ierr;
    int size = 0;
    int i = 0;
    ierr = VecGetSize(x, &size);
    const PetscScalar *array;
    ierr = VecGetArrayRead(x, &array);
    for(i=0; i<size; i++)
    {
        GCGE_Printf("%18.15e\t", array[i]);
    }
    ierr = VecRestoreArrayRead(x, &array);
    GCGE_Printf("\n");
}

void PETSCPrintBV(BV x, char *name)
{
    PetscErrorCode ierr;
    int n = 0, i = 0;
    ierr = BVGetSizes(x, NULL, NULL, &n);
    Vec xs;
    GCGE_Printf("%s = [\n", name);
    for(i=0; i<n; i++)
    {
        ierr = BVGetColumn(x, i, &xs);
	PETSCPrintVec(xs);
        ierr = BVRestoreColumn(x, i, &xs);
    }
    GCGE_Printf("];\n");
}
