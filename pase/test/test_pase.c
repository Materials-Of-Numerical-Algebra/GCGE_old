/*
 * =====================================================================================
 *
 *       Filename:  test_pase.c
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
#include "pase_solver.h"
#include "gcge.h"
#include "pase.h"
#include "gcge_app_pase.h"
#include "gcge_app_slepc.h"


static char help[] = "Test MultiGrid.\n";
void GetPetscMat(Mat *A, Mat *B, PetscInt n);
void PETSCPrintMat(Mat A, char *name);
void PETSCPrintVec(Vec x);
void PETSCPrintBV(BV x, char *name);
void GCGE_PETSCMultiVecPrint(void **x, GCGE_INT n, GCGE_OPS *ops);
void GCGE_PETSCPrintMat(void *A);
void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *num_levels);
/* 
 *  Description:  测试PASE_MULTIGRID
 */
int
main ( int argc, char *argv[] )
{
    /* SlepcInitialize */
    SlepcInitialize(&argc,&argv,(char*)0,help);
    PetscErrorCode ierr;
    PetscMPIInt    rank, size;
    PetscViewer    viewer;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    /* 得到细网格矩阵 */
    Mat      A, B;
    PASE_INT n = 40;
    PASE_INT nev = 10;
    PASE_INT num_levels = 3;
//    PASE_INT n = 5000;
//    PASE_INT nev = 200;
//    PASE_INT num_levels = 6;
    PASE_INT aux_coarse_level = -1;

    GetCommandLineInfo(argc, argv, &n, &num_levels);
    GCGE_Printf("n: %d\n", n);
    //如果n>0, 那就直接产生差分矩阵, 否则就从文件读入矩阵
    if(n > 0) {
      GetPetscMat(&A, &B, n);
    } else {
      char file_A[PETSC_MAX_PATH_LEN] = "fileinput";
      ierr = PetscOptionsGetString(NULL, NULL, "-mat_A", file_A, sizeof(file_A), NULL);
      char file_B[PETSC_MAX_PATH_LEN] = "fileinput";
      ierr = PetscOptionsGetString(NULL, NULL, "-mat_B", file_B, sizeof(file_B), NULL);
      SLEPC_ReadMatrixBinary(&A, file_A);
      SLEPC_ReadMatrixBinary(&B, file_B);
    }

    //给pase用到的参数赋值
    PASE_PARAMETER param;
    PASE_PARAMETER_Create(&param, num_levels, nev);
    PASE_PARAMETER_Get_from_command_line(param, argc, argv);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_CreateSLEPC(&gcge_ops);
    gcge_ops->MultiVecPrint = GCGE_PETSCMultiVecPrint;
    //创建特征值与特征向量空间

    param->max_initial_direct_count = 30;
    //param->max_cycle_count_each_level[0] = 100;
    param->max_cycle_count_each_level[0] = 15;

    //pase求解
    PASE_EigenSolver((void*)A, (void*)B, NULL, NULL, nev, param, gcge_ops);

    //释放空间
    GCGE_OPS_Free(&gcge_ops);
    PASE_PARAMETER_Destroy(&param);
    ierr = MatDestroy(&A);
    ierr = MatDestroy(&B);

    /* PetscFinalize */
    ierr = SlepcFinalize();
    return 0;
}

void GetPetscMat(Mat *A, Mat *B, PetscInt n)
{
    PetscInt N = n*n;
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
      if (i<n-1) { ierr = MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES); }
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
      ierr = MatSetValue(*B,II,II,1.0/((n+1)*(n+1)),INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);

}

void GCGE_PETSCPrintMat(void *A)
{
  PETSCPrintMat((Mat)A, "A");
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
    for(i=0; i<5; i++)
    {
        GCGE_Printf("%18.14e\n", array[i]);
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

void GCGE_PETSCMultiVecPrint(void **x, GCGE_INT n, GCGE_OPS *ops)
{
  PETSCPrintBV((BV)x, "x");
}

void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *num_levels)
{
  PASE_INT arg_index = 0;

  while (arg_index < argc)
  {
    //矩阵维数是 n*n
    if ( strcmp(argv[arg_index], "-n") == 0 )
    {
      arg_index++;
      *n = atoi(argv[arg_index++]);
    }
    if ( strcmp(argv[arg_index], "-num_levels") == 0 )
    {
      arg_index++;
      *num_levels = atoi(argv[arg_index++]);
    }
    else
    {
      arg_index++;
    }
  }
}
