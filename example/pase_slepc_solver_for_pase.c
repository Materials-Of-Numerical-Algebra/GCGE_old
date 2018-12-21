/*
 * =====================================================================================
 *
 *       Filename:  test_ops.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  09/27/2018 04:47:29 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//#include "memwatch.h"

#include "gcge.h"
#include "gcge_app_slepc.h"
#include "gcge_app_pase.h"
#include "pase.h"

static char help[] = "Use GCGE-SLEPc-PASE to solve an eigensystem Ax=kBx with the matrixes loaded from files.\n";

void PetscGetDifferenceMatrix(Mat *A, PetscInt n, PetscInt m);
int main(int argc, char* argv[])
{
    PetscErrorCode ierr;
    SlepcInitialize(&argc,&argv,(char*)0,help);
    srand(1);

    //创建矩阵
    Mat A, B, P;
    char file_A[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat_A", file_A, sizeof(file_A), NULL);
    char file_B[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat_B", file_B, sizeof(file_B), NULL);
    char file_P[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat_P", file_P, sizeof(file_P), NULL);

    PetscInt  n = 10, m = 10;
    PetscBool flag;
    //读入矩阵, 如果命令行提供了矩阵B和P的地址, 才读矩阵B和P
    if(strcmp(file_A, "fileinput") != 0)
    {
        SLEPC_ReadMatrixBinary(&A, file_A);
    }
    else
    {
        ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
        ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, &flag);
        if (!flag) m=n;
        PetscGetDifferenceMatrix(&A, n, m);
    }
    if(strcmp(file_B, "fileinput") != 0)
    {
        SLEPC_ReadMatrixBinary(&B, file_B);
    }
    if(strcmp(file_P, "fileinput") != 0)
    {
        SLEPC_ReadMatrixBinary(&P, file_P);
    }

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_SLEPC_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);

    //创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);
    PASE_OPS_Setup(pase_ops);

    //创建pase辅助矩阵
    PASE_Matrix pase_mat_A;
    PASE_INT    num_aux_vec = 2;
    PASE_MatrixCreate(&pase_mat_A, num_aux_vec, A, pase_ops);

    BV A_aux_Hh = (BV)(pase_mat_A->aux_Hh);

    int i = 0;
    Vec x;
    for(i=0; i<num_aux_vec; i++) {
        BVGetColumn(A_aux_Hh, i, &x);
        VecSet(x, 0.0);
        BVRestoreColumn(A_aux_Hh, i, &x);
    }

    pase_mat_A->aux_hh[0] = 1.0;
    pase_mat_A->aux_hh[1] = 0.0;
    pase_mat_A->aux_hh[2] = 0.0;
    pase_mat_A->aux_hh[3] = 1.0;

    int nev = 5;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    PASE_MultiVector evec;
    PASE_DefaultMultiVecCreateByMat((void***)(&evec), nev, (void*)pase_mat_A, pase_ops);

    //-------------------------------------------------------
    //在pase中调用GCGE时，就用下面三步
    //创建solver
    GCGE_SOLVER *slepc_pase_solver = GCGE_SOLVER_PASE_Create(pase_mat_A, 
            NULL, nev, eval, evec, pase_ops);

    //求解
    GCGE_SOLVER_Solve(slepc_pase_solver);  

    //释放空间
    GCGE_SOLVER_Free_Some(&slepc_pase_solver);
    //-------------------------------------------------------

    free(eval);
    eval = NULL;
    PASE_DefaultMultiVecDestroy((void***)(&evec), nev, pase_ops);

    PASE_MatrixDestroy(&pase_mat_A, pase_ops);

    //释放矩阵空间
    ierr = MatDestroy(&A);
    if(strcmp(file_B, "fileinput") != 0)
    {
        ierr = MatDestroy(&B);
    }
    if(strcmp(file_P, "fileinput") != 0)
    {
        ierr = MatDestroy(&P);
    }

    GCGE_OPS_Free(&gcge_ops);
    PASE_OPS_Free(&pase_ops);

    return 0;
}

void PetscGetDifferenceMatrix(Mat *A, PetscInt n, PetscInt m)
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
}
