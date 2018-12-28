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
#include "pase_mg.h"
#include "gcge.h"
#include "pase.h"
#include "gcge_app_slepc.h"


static char help[] = "Test MultiGrid.\n";
void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
void PETSCPrintMat(Mat A, char *name);
void PETSCPrintVec(Vec x);
void PETSCPrintBV(BV x, char *name);
/* 
 *  Description:  测试PASE_MULTIGRID
 */
int
main ( int argc, char *argv[] )
{
    /* SlepcInitialize */
    SlepcInitialize(&argc,&argv,(char*)0,help);
    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscViewer    viewer;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    /* 得到细网格矩阵 */
    Mat      A, B;
    PetscInt n = 7, m = 3;
    GetPetscMat(&A, &B, n, m);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_CreateSLEPC(&gcge_ops);
    //用gcge_ops创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);

    //创建特征值与特征向量空间
    int nev = 5;
    BV evec;
    gcge_ops->MultiVecCreateByMat((void***)(&evec), nev, (void*)A, gcge_ops);
    double *eval = (double*)calloc(nev, sizeof(double));

    //给pase用到的参数赋值
    PASE_PARAMETER param   = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
    param->cycle_type      = 0;   //二网格
    param->block_size      = nev; //特征值个数
    param->max_cycle       = 2;  //二网格迭代次数
    param->max_pre_iter    = 1;   //前光滑次数
    param->max_post_iter   = 1;   //后光滑次数
    param->atol            = 1e-10;
    param->rtol            = 1e-12;
    param->print_level     = 1;
    param->max_level       = 3;   //AMG层数
    PASEGetCommandLineInfo(argc, argv, &(param->block_size), &(param->atol), &(param->max_pre_iter));
    param->min_coarse_size = param->block_size * 30; //最粗层网格最少有30*nev维
    //param->min_coarse_size = 500;
    PrintParameter(param);

    //pase求解
    PASE_EigenSolver((void*)A, (void*)B, eval, (void**)evec, nev, param, 
            gcge_ops, pase_ops);

    //释放空间
    free(eval);  eval = NULL;
    gcge_ops->MultiVecDestroy((void***)(&evec), nev, gcge_ops);
    PASE_OPS_Free(&pase_ops); 
    GCGE_OPS_Free(&gcge_ops);
    free(param); param = NULL;
    ierr = MatDestroy(&petsc_mat_A);
    ierr = MatDestroy(&petsc_mat_B);

    /* PetscFinalize */
    ierr = SlepcFinalize();
    return 0;
}

void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
{
    PetscInt N = n*m;
    PetscInt Istart, Iend, II, i, j;
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD,A);
    ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,N,N);
    ierr = MatSetFromOptions(*A);
    ierr = MatSetUp(*A);
    ierr = MatGetOwnershipRange(*A,&Istart,&Iend);
    for (II=Istart;II<Iend;II++) {
      i = II/n; j = II-i*n;
      if (i>0) { ierr = MatSetValue(*A,II,II-n,-1.0,INSERT_VALUES); }
      if (i<m-1) { ierr = MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES); }
      if (j>0) { ierr = MatSetValue(*A,II,II-1,-1.0,INSERT_VALUES); }
      if (j<n-1) { ierr = MatSetValue(*A,II,II+1,-1.0,INSERT_VALUES); }
      ierr = MatSetValue(*A,II,II,4.0,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);

    ierr = MatCreate(PETSC_COMM_WORLD,B);
    ierr = MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,N,N);
    ierr = MatSetFromOptions(*B);
    ierr = MatSetUp(*B);
    ierr = MatGetOwnershipRange(*B,&Istart,&Iend);
    for (II=Istart;II<Iend;II++) {
      ierr = MatSetValue(*B,II,II,1.0,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);

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
