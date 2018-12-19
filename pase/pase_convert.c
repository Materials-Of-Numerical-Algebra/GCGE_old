/*
 * =====================================================================================
 *
 *       Filename:  pase_convert.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年12月19日 09时57分13秒
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "pase_convert.h"

/**
 * @brief HYPRE hypre_ParCSRMatrix convert to PETSC Mat 
 *
 * @param petsc_mat
 * @param hypre_mat
 */
void MatrixConvertHYPRE2PETSC(Mat *petsc_mat, HYPRE_ParCSRMatrix hypre_mat)
{
   PetscMPIInt rank;
   HYPRE_Int idx, row_start, row_end, col_start, col_end;
   HYPRE_Int *col_ind, size;
   HYPRE_Complex *values;

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

   /* "End" is actually the row number of the last row on this processor. */
   hypre_ParCSRMatrixGetLocalRange(hypre_mat, &row_start, &row_end, &col_start, &col_end);
//   printf ( "hypre, %d: row_start = %d, row_end = %d\n", rank, row_start, row_end );
//   printf ( "hypre, %d: col_start = %d, col_end = %d\n", rank, col_start, col_end );

   MatCreate(PETSC_COMM_WORLD, petsc_mat);
   MatSetSizes(*petsc_mat, row_end-row_start+1, col_end-col_start+1, hypre_mat->global_num_rows, hypre_mat->global_num_cols);
   MatSetType(*petsc_mat, MATAIJ);
   MatSetUp(*petsc_mat);

   for (idx = row_start; idx <= row_end; ++idx)
   {
      /* 返回hypre_mat的第idx行, OUT: size col_ind, values */
      hypre_ParCSRMatrixGetRow(hypre_mat, idx, &size, &col_ind, &values);
//      printf ( "idx  = %d, col_ind = %d, %d\n", idx,  col_ind[0], col_ind[1] );
//      printf ( "size = %d, values  = %f, %f\n", size, values[0],  values[1] );
      /* 对petsc_mat的第idx行赋值 */
      MatSetValues(*petsc_mat, 1, &idx, size, col_ind, values, INSERT_VALUES); 
      hypre_ParCSRMatrixRestoreRow(hypre_mat, idx, &size, &col_ind, &values);
   }
   MatAssemblyBegin(*petsc_mat, MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(*petsc_mat, MAT_FINAL_ASSEMBLY);

   MatGetOwnershipRange(*petsc_mat, &row_start, &row_end);
   MatGetOwnershipRangeColumn(*petsc_mat, &col_start, &col_end);
//   printf ( "petsc, %d: row_start = %d, row_end = %d\n", rank, row_start, row_end );
//   printf ( "petsc, %d: col_start = %d, col_end = %d\n", rank, col_start, col_end );
}

/**
 * @brief HYPRE HYPRE_IJMatrix convert to PETSC Mat
 *
 * @param hypre_ij_mat
 * @param petsc_mat
 */
void MatrixConvertPETSC2HYPRE(HYPRE_IJMatrix *hypre_ij_mat, Mat petsc_mat)
{
   PetscMPIInt rank;
   HYPRE_Int idx, row_start, row_end, col_start, col_end;
   const PetscInt *col_ind;
   const PetscScalar *values;
   PetscInt size;

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

   MatGetOwnershipRange(petsc_mat, &row_start, &row_end);
   MatGetOwnershipRangeColumn(petsc_mat, &col_start, &col_end);
//   printf ( "petsc, %d: row_start = %d, row_end = %d\n", rank, row_start, row_end );
//   printf ( "petsc, %d: col_start = %d, col_end = %d\n", rank, col_start, col_end );

   HYPRE_IJMatrixCreate(PETSC_COMM_WORLD, row_start, row_end-1, col_start, col_end-1, hypre_ij_mat);
   HYPRE_IJMatrixSetObjectType(*hypre_ij_mat,  HYPRE_PARCSR);
   HYPRE_IJMatrixInitialize(*hypre_ij_mat);

   for (idx = row_start; idx < row_end; ++idx)
   {
      /* 返回hypre_mat的第idx行, OUT: size col_ind, values */
      MatGetRow(petsc_mat, idx, &size, &col_ind, &values);
      /* 对petsc_mat的第idx行赋值 */
      HYPRE_IJMatrixSetValues(*hypre_ij_mat, 1, &size, &idx, col_ind, values);
      MatRestoreRow(petsc_mat, idx, &size, &col_ind, &values);
   }
   HYPRE_IJMatrixAssemble(*hypre_ij_mat);

   HYPRE_IJMatrixGetLocalRange(*hypre_ij_mat, &row_start, &row_end, &col_start, &col_end);
//   printf ( "hypre, %d: row_start = %d, row_end = %d\n", rank, row_start, row_end );
//   printf ( "hypre, %d: col_start = %d, col_end = %d\n", rank, col_start, col_end );
}
