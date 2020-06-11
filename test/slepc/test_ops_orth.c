/*
 * =====================================================================================
 *
 *       Filename:  test_solver.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时57分13秒
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
#include <petscsys.h>
#include <petscviewer.h>
#include <petscmat.h>

#include "gcge.h"
#include "gcge_app_slepc.h"

static char help[] = "Use GCGE-SLEPc to solve an eigensystem Ax=kBx with the matrixes loaded from files.\n";

int main(int argc, char* argv[])
{
   PetscErrorCode ierr;
   ierr = SlepcInitialize(&argc,&argv,(char*)0,help);

   int          mv_s[2], mv_e[2], row, col[3], idx;
   double       value[3];
   BV           multivec, multivec_tmp;
   int          nvec = 5, n = 10;
   Vec          vec[3];
   Mat          B;
   PetscViewer  view;


   MatCreate(PETSC_COMM_WORLD,&B);
   MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n);
   MatSetFromOptions(B);
   MatSetUp(B);
   value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
   for (row=1; row<n-1; row++) {
      col[0] = row-1; col[1] = row; col[2] = row+1;
      MatSetValues(B,1,&row,3,col,value,INSERT_VALUES);
   }
   row = n-1; col[0] = row-1; col[1] = row;
   value[0] = -1.0; value[1] = 2.0;
   MatSetValues(B,1,&row,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   row = 0; col[0] = row; col[1] = row+1; 
   value[0] = 2.0; value[1] = -1.0;
   MatSetValues(B,1,&row,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
   MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
   MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
   MatView(B, view);

   MatCreateVecs(B, NULL, &vec[0]);
   BVCreate(PETSC_COMM_WORLD, &multivec);
   BVCreate(PETSC_COMM_WORLD, &multivec_tmp);
   BVSetType(multivec,     BVMAT);
   BVSetType(multivec_tmp, BVMAT);
   BVSetSizesFromVec(multivec,     vec[0], nvec);
   BVSetSizesFromVec(multivec_tmp, vec[0], nvec);

   GCGE_OPS *slepc_ops;
   GCGE_OPS_CreateSLEPC(&slepc_ops);
   slepc_ops->MultiVecSetRandomValue((void**)multivec,     0, nvec, slepc_ops);
   slepc_ops->MultiVecSetRandomValue((void**)multivec_tmp, 0, nvec, slepc_ops);
   slepc_ops->GetVecFromMultiVec((void**)multivec, 1, (void**)&vec[1], slepc_ops);
   slepc_ops->GetVecFromMultiVec((void**)multivec, 2, (void**)&vec[2], slepc_ops);
   slepc_ops->VecAxpby(1.0, (void*)vec[1], 0.0, (void*)vec[2], slepc_ops);
   slepc_ops->RestoreVecForMultiVec((void**)multivec, 1, (void**)&vec[1], slepc_ops);
   slepc_ops->RestoreVecForMultiVec((void**)multivec, 2, (void**)&vec[2], slepc_ops);
   BVView(multivec, view);

   double *prod = (double*)calloc(nvec*nvec, sizeof(double)); 
   int    lda   = nvec; 

   slepc_ops->Orthonormalization = GCGE_Orth_GramSchmidt;
   GCGE_OrthSetup_GramSchmidt(10, 0.75, 1e-8, (void**)multivec_tmp, 0, slepc_ops);
   /* 从0开始是否会有问题，比如第一个向量会不会被归一化 */
   slepc_ops->Orthonormalization((void**)multivec, 0, &nvec, (void*)B, slepc_ops);

   /* 验证B内积是否为单位矩阵 */
   mv_s[0] = 0;
   mv_e[0] = nvec;
   mv_s[1] = 0;
   mv_e[1] = nvec;
   slepc_ops->MatDotMultiVec((void*)B, (void**)multivec, (void**)multivec_tmp, 
	 mv_s, mv_e, slepc_ops);
   slepc_ops->MultiVecInnerProd((void**)multivec, (void**)multivec_tmp, 
	 prod, "sym", mv_s, mv_e, lda, 0, slepc_ops);
   GCGE_Printf("prod:\n");
   for(row=0; row<nvec; row++)
   {
      for(idx=0; idx<nvec; idx++)
      {
	 GCGE_Printf("%f\t", prod[row*lda+idx]);
      }
      GCGE_Printf("\n");
   }

   free(prod); prod = NULL;

   BVDestroy(&multivec);
   BVDestroy(&multivec_tmp);
   VecDestroy(&vec[0]);
   MatDestroy(&B);
   GCGE_OPS_Free(&slepc_ops);

   ierr = SlepcFinalize();
   return ierr;
}
