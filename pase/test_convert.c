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

#include "HYPRE_parcsr_ls.h"
#include "temp_multivector.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_utilities.h"


void GetPetscMat(Mat *A, PetscInt n, PetscInt m);
void GetMultigridMatFromHypreToPetsc(Mat **A_array, Mat **P_array, HYPRE_Int *num_levels, HYPRE_ParCSRMatrix hypre_parcsr_mat);
static char help[] = "Test Matrix Convert.\n";
/* 
 *  Description:  测试HYPRE矩阵与PETSC矩阵的互相转化
 */
   int
main ( int argc, char *argv[] )
{
   PetscInitialize(&argc,&argv,(char*)0,help);
   PetscErrorCode ierr;
   PetscMPIInt rank;
   PetscViewer viewer;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

   Mat                petsc_mat_A;
   HYPRE_IJMatrix     hypre_ij_mat;
   HYPRE_ParCSRMatrix hypre_parcsr_mat;
   PetscInt n = 4, m = 4;

   GetPetscMat(&petsc_mat_A, n, m);
   MatView(petsc_mat_A, viewer);

   MatrixConvertPETSC2HYPRE(&hypre_ij_mat, petsc_mat_A);
//   HYPRE_IJMatrixPrint(hypre_ij_mat, "./hypre_ij_mat");

   HYPRE_IJMatrixGetObject(hypre_ij_mat, (void**) &hypre_parcsr_mat);


   Mat *A_array, *P_array;
   HYPRE_Int idx, num_levels = 2;
   GetMultigridMatFromHypreToPetsc(&A_array, &P_array, &num_levels, hypre_parcsr_mat);

   for (idx = 0; idx < num_levels; ++idx)
   {
      MatView(A_array[idx], viewer);
   }
   for (idx = 0; idx < num_levels-1; ++idx)
   {
      MatView(P_array[idx], viewer);
   }

   for (idx = 0; idx < num_levels; ++idx)
   {
      ierr = MatDestroy(&A_array[idx]);
   }
   free(A_array);
   for (idx = 0; idx < num_levels-1; ++idx)
   {
      ierr = MatDestroy(&P_array[idx]);
   }
   free(P_array);
   HYPRE_IJMatrixDestroy(hypre_ij_mat);
   ierr = MatDestroy(&petsc_mat_A);

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
