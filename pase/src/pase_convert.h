
#ifndef  _PASE_CONVERT_H_
#define  _PASE_CONVERT_H_



#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "_hypre_parcsr_mv.h"
#include <petscmat.h>
#include <petscvec.h>
#include "phg.h"


void MatrixConvertHYPRE2PETSC(Mat *petsc_mat, HYPRE_ParCSRMatrix hypre_mat);
void MatrixConvertPETSC2HYPRE(HYPRE_IJMatrix *hypre_ij_mat, Mat petsc_mat);
void MatrixConvertPHG2HYPRE(HYPRE_IJMatrix *hypre_ij_mat, MAT *phg_mat);
void MatrixConvertPHG2PETSC(Mat *petsc_mat, MAT *phg_mat);

#endif
