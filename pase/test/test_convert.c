/*
 * =====================================================================================
 *
 *       Filename:  test_convert.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年12月19日 15时50分13秒
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include "pase_convert.h"

#include "_hypre_parcsr_ls.h"


void GetPetscMat(Mat *A, PetscInt n, PetscInt m);
void GetMultigridMatFromHypreToPetsc(Mat **A_array, Mat **P_array, HYPRE_Int *num_levels, HYPRE_ParCSRMatrix hypre_parcsr_mat);
static char help[] = "Test Matrix Convert and AMG.\n";
/* 
 *  Description:  测试HYPRE矩阵与PETSC矩阵的互相转化
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
   Mat                petsc_mat_A;
   HYPRE_IJMatrix     hypre_ij_mat;
   HYPRE_ParCSRMatrix hypre_parcsr_mat;
   PetscInt n = 7, m = 3, row_start, row_end, col_start, col_end;
   /* 得到一个PETSC矩阵 */
   GetPetscMat(&petsc_mat_A, n, m);
//   MatView(petsc_mat_A, viewer);
   /* 将PETSC矩阵转化为HYPRE矩阵 */
   MatrixConvertPETSC2HYPRE(&hypre_ij_mat, petsc_mat_A);
//   HYPRE_IJMatrixPrint(hypre_ij_mat, "./hypre_ij_mat");
   HYPRE_IJMatrixGetObject(hypre_ij_mat, (void**) &hypre_parcsr_mat);
   /* 声明一系列矩阵和向量 */
   Mat *A_array, *P_array;
   Vec *U_array, *F_array;
   HYPRE_Int idx, j, num_levels = 2;
   /* 利用HYPRE矩阵通过AMG得到一系列PETSC矩阵, 以及插值矩阵 */
   GetMultigridMatFromHypreToPetsc(&A_array, &P_array, &num_levels, hypre_parcsr_mat);
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


   /* 销毁矩阵向量 */
   for (idx = 0; idx < num_levels; ++idx)
   {
      ierr = MatDestroy(&A_array[idx]);
      ierr = VecDestroy(&U_array[idx]);
      ierr = VecDestroy(&F_array[idx]);
   }
   free(A_array); 
   free(U_array); 
   free(F_array); 
   for (idx = 0; idx < num_levels-1; ++idx)
   {
      ierr = MatDestroy(&P_array[idx]);
   }
   free(P_array);
   HYPRE_IJMatrixDestroy(hypre_ij_mat);
   ierr = MatDestroy(&petsc_mat_A);
   /* PetscFinalize */
   ierr = PetscFinalize();
   return 0;
}

void GetPetscMat(Mat *A, PetscInt n, PetscInt m)
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

/* -------------------------- 利用AMG生成各个层的矩阵------------------ */
void GetMultigridMatFromHypreToPetsc(Mat **A_array, Mat **P_array, HYPRE_Int *num_levels, HYPRE_ParCSRMatrix hypre_parcsr_mat)
{
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
   for (idx = 0; idx < *num_levels; ++idx)
   {
      MatrixConvertHYPRE2PETSC((*A_array)+idx, hypre_mat[idx]);
   }
   hypre_mat = hypre_ParAMGDataPArray(amg_data);
   for (idx = 0; idx < *num_levels-1; ++idx)
   {
      MatrixConvertHYPRE2PETSC((*P_array)+idx, hypre_mat[idx]);
   }
   HYPRE_BoomerAMGDestroy(amg);	
   hypre_ParVectorDestroy(hypre_par_vec);
}
