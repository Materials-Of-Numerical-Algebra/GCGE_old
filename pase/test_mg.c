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


static char help[] = "Test MultiGrid.\n";
void GetPetscMat(Mat *A, PetscInt n, PetscInt m);
/* 
 *  Description:  测试PASE_MULTIGRID
 */
   int
main ( int argc, char *argv[] )
{
   /* PetscInitialize */
   PetscInitialize(&argc,&argv,(char*)0,help);
   PetscErrorCode ierr;
   PetscMPIInt rank;
   PetscViewer viewer;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

   /* 测试矩阵声明 */
   Mat                petsc_mat_A, petsc_mat_B;
   PetscInt n = 7, m = 3, row_start, row_end, col_start, col_end;
   /* 得到一个PETSC矩阵 */
   GetPetscMat(&petsc_mat_A, n, m);
   HYPRE_Int idx, j, num_levels = 2;

   PASE_MULTIGRID multi_grid;
   PASE_MULTIGRID_Create(&multi_grid, num_levels, 
        (void *)&petsc_mat_A, (void *)&petsc_mat_B, 
	/* TODO: 如何给出这两个ops */
	GCGE_OPS *gcge_ops, PASE_OPS *pase_ops);

   /* 声明一系列矩阵和向量 */
   Mat *A_array, *B_array, *P_array;
   Vec *U_array, *F_array;

   A_array = (Mat*)(multi_grid->A_array[0]);
   B_array = (Mat*)(multi_grid->B_array[0]);
   P_array = (Mat*)(multi_grid->P_array[0]);

   /* 打印一系列矩阵 */
   PetscPrintf(PETSC_COMM_WORLD, "A_array\n");
   for (idx = 0; idx < num_levels; ++idx)
   {
//      MatGetOwnershipRange(A_array[idx], &row_start, &row_end);
//      MatGetOwnershipRangeColumn(A_array[idx], &col_start, &col_end);
//      printf("%d: row_start = %d, row_end = %d, col_start = %d, col_end = %d\n", 
//	    rank, row_start, row_end, col_start, col_end);
      PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
      MatView(A_array[idx], viewer);
   }
   PetscPrintf(PETSC_COMM_WORLD, "P_array\n");
   for (idx = 0; idx < num_levels-1; ++idx)
   {
      PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
//      MatGetOwnershipRange(P_array[idx], &row_start, &row_end);
//      MatGetOwnershipRangeColumn(P_array[idx], &col_start, &col_end);
//      printf("%d: row_start = %d, row_end = %d, col_start = %d, col_end = %d\n", 
//	    rank, row_start, row_end, col_start, col_end);
      MatView(P_array[idx], viewer);
   }
   /* 通过一系列矩阵生成对应的一系列向量 */
   U_array = malloc(sizeof(Vec)*num_levels);
   F_array = malloc(sizeof(Vec)*num_levels);
   for (idx = 0; idx < num_levels; ++idx)
   {
      MatCreateVecs(A_array[idx], PETSC_NULL, U_array+idx);
      MatCreateVecs(A_array[idx], F_array+idx, PETSC_NULL);
      VecGetOwnershipRange(U_array[idx], &row_start, &row_end);
      for (j = row_start; j < row_end; ++j)
      {
	 VecSetValue(U_array[idx], j, 1.0, INSERT_VALUES);
      }
      VecAssemblyBegin(U_array[idx]);
      VecAssemblyEnd(U_array[idx]);
   }
   /* 测试矩阵向量乘 F[idx] = A[idx] U[idx] */
   for (idx = 0; idx < num_levels; ++idx)
   {
      PetscPrintf(PETSC_COMM_WORLD, "F[%d] = A[%d] U[%d]\n", idx, idx, idx);
      MatMult(A_array[idx], U_array[idx], F_array[idx]);
      VecView(F_array[idx], viewer);
   }
   /* 测试向量插值 U[idx-1] = P[idx-1] U[idx] */
   for (idx = num_levels-1; idx > 0 ; --idx)
   {
      PetscPrintf(PETSC_COMM_WORLD, "U[%d] = P[%d] U[%d]\n", idx-1, idx-1, idx);
      MatMult(P_array[idx-1], U_array[idx], U_array[idx-1]);
      VecView(U_array[idx-1], viewer);
   }
   /* 测试矩阵向量乘 F[idx] = A[idx] U[idx] */
   for (idx = 0; idx < num_levels; ++idx)
   {
      PetscPrintf(PETSC_COMM_WORLD, "F[%d] = A[%d] U[%d]\n", idx, idx, idx);
      MatMult(A_array[idx], U_array[idx], F_array[idx]);
      VecView(F_array[idx], viewer);
   }
   /* 测试向量限制 U[idx+1] = P[idx]' U[idx] */
   for (idx = 0; idx < num_levels-1; ++idx)
   {
      PetscPrintf(PETSC_COMM_WORLD, "U[%d] = P[%d]' U[%d]\n", idx+1, idx, idx);
      MatMultTranspose(P_array[idx], U_array[idx], U_array[idx+1]);
      VecView(U_array[idx+1], viewer);
   }

   /* TODO: 测试BV结构的向量进行投影插值 */
   PASE_MULTIGRID_FromItoJ(PASE_MULTIGRID multi_grid, 
	 PASE_INT level_i, PASE_INT level_j, 
	 PASE_INT *mv_s, PASE_INT *mv_e, 
	 void **pvx_i, void** pvx_j)

   /* 销毁矩阵向量 */
   for (idx = 0; idx < num_levels; ++idx)
   {
      ierr = VecDestroy(&U_array[idx]);
      ierr = VecDestroy(&F_array[idx]);
   }
   free(U_array); 
   free(F_array); 
   ierr = MatDestroy(&petsc_mat_A);
   ierr = MatDestroy(&petsc_mat_B);
   /* PetscFinalize */
   ierr = PetscFinalize();
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
