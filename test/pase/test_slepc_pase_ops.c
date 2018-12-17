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
#include <petscsys.h>
#include <petscviewer.h>
#include <petscmat.h>

#include "gcge.h"
#include "pase.h"
#include "gcge_app_slepc.h"
void PASE_SLEPCPrintMat(PASE_Matrix pase_matrix);
void PASE_SLEPCPrintMultiVec(PASE_MultiVector vecs);
void PASE_SLEPCPrintVec(PASE_Vector vecs);
void PetscGetDifferenceMatrix(Mat *A, PetscInt n, PetscInt m);

static char help[] = "Test SLEPC PASE_OPS.\n";
int main(int argc, char* argv[])
{
    PetscErrorCode ierr;
    SlepcInitialize(&argc,&argv,(char*)0,help);
    srand(1);

    Mat A;
    int n = 3;
    PetscGetDifferenceMatrix(&A, n, n);
    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_SLEPC_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);

    //创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);
    PASE_OPS_Setup(pase_ops);

    PASE_Matrix pase_matrix;
    PASE_INT    num_aux_vec = 2;
    PASE_MatrixCreate(&pase_matrix, num_aux_vec, (void*)A, pase_ops);

    //创建pase辅助向量组
    PASE_MultiVector multi_vec_x;
    PASE_MultiVector multi_vec_y;
    //GCGE_Printf ( "PASE_OPS->MultiVecCreateByMat\n" );
    pase_ops->MultiVecCreateByMat((void ***)(&multi_vec_x), num_aux_vec, 
            (void *)pase_matrix, pase_ops);
    pase_ops->MultiVecCreateByMat((void ***)(&multi_vec_y), num_aux_vec, 
            (void *)pase_matrix, pase_ops);

    //给pase辅助矩阵中向量组赋初值
    //GCGE_Printf ( "set random value for aux_Hh\n" );
    gcge_ops->MultiVecSetRandomValue(pase_matrix->aux_Hh, 
            0, num_aux_vec, gcge_ops);

    //GCGE_Printf("set random value for aux_hh\n");
    int i, j;
    //给pase辅助矩阵中block赋初值
    for(i=0; i<num_aux_vec*num_aux_vec; i++) {
        pase_matrix->aux_hh[i] = ((double)rand())/((double)RAND_MAX+1);
    }
    //对称化
    for(i=0; i<num_aux_vec; i++) {
        for(j=0; j<i; j++) {
            pase_matrix->aux_hh[i*num_aux_vec+j] 
                = pase_matrix->aux_hh[j*num_aux_vec+j];
        }
    }

    //GCGE_Printf("set random value for x\n");
    //给pase辅助向量组x赋初值
    pase_ops->MultiVecSetRandomValue((void**)multi_vec_x, 0, num_aux_vec, pase_ops);

    PASE_Vector vec_x;
    PASE_Vector vec_y;
    pase_ops->VecCreateByMat((void**)(&vec_x), (void*)pase_matrix, pase_ops);
    pase_ops->VecCreateByVec((void**)(&vec_y), (void*)vec_x, pase_ops);
    pase_ops->VecSetRandomValue((void*)vec_x, 3, pase_ops);
    pase_ops->VecSetRandomValue((void*)vec_y, 2, pase_ops);

    //--------------------------------------------------------------
    GCGE_Printf("vec_x:\n");
    PASE_SLEPCPrintVec(vec_x);
    GCGE_Printf("vec_y:\n");
    PASE_SLEPCPrintVec(vec_y);
    double a = 2.1;
    double b = 5.4;
    /* It's ok. */
    pase_ops->VecAxpby(a, (void*)vec_x, b, (void*)vec_y, pase_ops);
    GCGE_Printf( "vec_y = %f * vec_x + %f * vec_y\n", a, b );
    PASE_SLEPCPrintVec(vec_y);

    //--------------------------------------------------------------
    /* It's ok. */
#if 1
    pase_ops->MatDotVec((void*)pase_matrix, (void*)vec_x, (void*)vec_y, pase_ops);
    GCGE_Printf("vec_y = pase_matrix * vec_x:\n");
    GCGE_Printf ( "pase_matrix\n" );
    PASE_SLEPCPrintMat(pase_matrix);
    GCGE_Printf("vec_x:\n");
    PASE_SLEPCPrintVec(vec_x);
    GCGE_Printf("vec_y:\n");
    PASE_SLEPCPrintVec(vec_y);
#endif

#if 0
    //--------------------------------------------------------------
    double value_ip = 0.0;
    /* It's ok. */
    pase_ops->VecInnerProd((void*)vec_x, (void*)vec_y, &value_ip, pase_ops);
    GCGE_Printf("vec_x:\n");
    PASE_SLEPCPrintVec(vec_x);
    GCGE_Printf("vec_y:\n");
    PASE_SLEPCPrintVec(vec_y);
    GCGE_Printf("value_ip = %f (vec_x^T * vec_y)\n", value_ip);

#endif
    int mv_s[2];
    int mv_e[2];
    mv_s[0] = 0;
    mv_e[0] = num_aux_vec;
    mv_s[1] = 0;
    mv_e[1] = num_aux_vec;
    //--------------------------------------------------------------
    /* It's ok. */
#if 1
    GCGE_Printf ( "pase_matrix\n" );
    PASE_SLEPCPrintMat(pase_matrix);
    GCGE_Printf("multi_vec_x: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_x);
    pase_ops->MatDotMultiVec((void *)pase_matrix, (void **)multi_vec_x, 
            (void **)multi_vec_y, mv_s, mv_e, pase_ops);
    GCGE_Printf ( "multi_vec_y = pase_matrix * multi_vec_x\n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);
#endif

#if 0
    //--------------------------------------------------------------
    GCGE_Printf("multi_vec_x: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_x);
    GCGE_Printf("multi_vec_y: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);
    /* It's ok. */
    pase_ops->MultiVecAxpby(a, (void**)multi_vec_x, b, (void**)multi_vec_y,
            mv_s, mv_e, pase_ops);
    GCGE_Printf( "multi_vec_y = %f * multi_vec_x + %f * multi_vec_y\n", a, b );
    PASE_SLEPCPrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    GCGE_Printf("multi_vec_x: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_x);
    GCGE_Printf("multi_vec_y: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);
    /* It's ok. */
    pase_ops->MultiVecAxpbyColumn(a, (void**)multi_vec_x, 1, b, 
            (void**)multi_vec_y, 0, pase_ops);
    GCGE_Printf( "multi_vec_y[0] = %f * multi_vec_x[1] + %f * multi_vec_y[0]\n", a, b );
    PASE_SLEPCPrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    //multi_vec_y = multi_vec_x * coef + multi_vec_y
    double coef[4] = {1.2, 2.3, 3.1, 4.4};
    GCGE_Printf("multi_vec_x: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_x);
    GCGE_Printf("multi_vec_y: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);
    GCGE_Printf("coef:\n1.2  3.1\n2.3  4.4\n");
    /* It's ok. */
    pase_ops->MultiVecLinearComb((void**)multi_vec_x, (void**)multi_vec_y,
            mv_s, mv_e, coef, num_aux_vec, 0, 1.0, 1.0, pase_ops);
    GCGE_Printf("multi_vec_y = multi_vec_x * coef + multi_vec_y: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    double values_ip[4];
    /* It's ok. */
    pase_ops->MultiVecInnerProd((void**)multi_vec_x, (void**)multi_vec_y,
            (double*)values_ip, "nonsym", mv_s, mv_e, num_aux_vec,
            0, pase_ops);
    GCGE_Printf("multi_vec_x: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_x);
    GCGE_Printf("multi_vec_y: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);
    GCGE_Printf("values_ip:\n%f\t%f\n%f\t%f\n",
            values_ip[0], values_ip[2], 
            values_ip[1], values_ip[3]);

    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MultiVecSwap((void**)multi_vec_x, (void**)multi_vec_y,
            mv_s, mv_e, pase_ops);
    GCGE_Printf("multi_vec_x: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_x);
    GCGE_Printf("multi_vec_y: \n");
    PASE_SLEPCPrintMultiVec(multi_vec_y);
#endif

    pase_ops->VecDestroy ((void **)&vec_x, pase_ops);
    pase_ops->VecDestroy ((void **)&vec_y, pase_ops);

    pase_ops->MultiVecDestroy ((void ***)&multi_vec_x, num_aux_vec, pase_ops);
    pase_ops->MultiVecDestroy ((void ***)&multi_vec_y, num_aux_vec, pase_ops);

    PASE_MatrixDestroy(&pase_matrix, pase_ops);

    //释放矩阵空间
    MatDestroy(&A);

    GCGE_OPS_Free(&gcge_ops);
    PASE_OPS_Free(&pase_ops);

    ierr = SlepcFinalize();
    return 0;
}

#if 1
void PASE_SLEPCPrintMat(PASE_Matrix pase_matrix)
{
    PetscErrorCode ierr;
    int     nrows;
    BV vecs = (BV)(pase_matrix->aux_Hh);
    const double *aux_Hh;
    ierr = BVGetArrayRead(vecs, &aux_Hh);
    ierr = BVGetSizes(vecs, &nrows, NULL, NULL);

    int     num_aux_vec = pase_matrix->num_aux_vec;
    int     ldd = nrows+num_aux_vec;

    double  *dense_mat = (double*)calloc(ldd*ldd, sizeof(double));
    int     row = 0;
    int     col = 0;
    int     i   = 0;
    int     j   = 0;
    //将A_H部分赋值给dense_mat
    for(row=0; row<nrows; row++)
    {
        dense_mat[row*ldd+row] = 4.0;
	if(row > 0) {
        dense_mat[row*ldd+row-1] = 1.0;
	}
	if(row < nrows-1) {
        dense_mat[row*ldd+row+1] = 1.0;
	}
    }
    //将aux_Hh部分赋值给dense_mat
    for(i=0; i<num_aux_vec; i++)
    {
        for(row=0; row<nrows; row++)
        {
            col = nrows+i;
            dense_mat[col*ldd+row] = aux_Hh[i*nrows+row];
            //对称化
            dense_mat[row*ldd+col] = aux_Hh[i*nrows+row];
        }
    }
    ierr = BVRestoreArrayRead(vecs, &aux_Hh);
    //aux_hh部分
    for(i=0; i<num_aux_vec; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            col = nrows+i;
            row = nrows+j;
            dense_mat[col*ldd+row] = pase_matrix->aux_hh[i*num_aux_vec+j];
        }
    }
    //打印稠密矩阵
    //GCGE_Printf("pase_matrix: n: %d\n",ldd);
    for(i=0; i<ldd; i++)
    {
        for(j=0; j<ldd; j++)
        {
            GCGE_Printf("%f\t", dense_mat[i*ldd+j]);
        }
        GCGE_Printf("\n");
    }
    free(dense_mat); dense_mat = NULL;
}
#endif

void PASE_SLEPCPrintMultiVec(PASE_MultiVector vecs)
{
    int num_aux_vec = vecs->num_aux_vec;
    int nrows = 0;

    BV b_H = (BV)(vecs->b_H);
    const double *b_H_Array;
    PetscErrorCode ierr;
    ierr = BVGetArrayRead(b_H, &b_H_Array);
    ierr = BVGetSizes(b_H, &nrows, NULL, NULL);

    int i = 0;
    int j = 0;
    //打印b_H向量组部分
    for(i=0; i<nrows; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            GCGE_Printf("%f\t", b_H_Array[j*nrows+i]);
        }
        GCGE_Printf("\n");
    }
    ierr = BVRestoreArrayRead(b_H, &b_H_Array);
    double *aux_h = vecs->aux_h;
    //打印aux_h部分
    for(i=0; i<num_aux_vec; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            GCGE_Printf("%f\t", aux_h[j*num_aux_vec+i]);
        }
        GCGE_Printf("\n");
    }
}

void PASE_SLEPCPrintVec(PASE_Vector vecs)
{
    int i = 0;
    int j = 0;
    int nrows = 0;
    PetscErrorCode ierr;
    //打印CSR向量组部分
    Vec b_H = (Vec)(vecs->b_H);
    ierr = VecGetSize(b_H, &nrows);
    const double *b_H_Array;
    ierr = VecGetArrayRead(b_H, &b_H_Array);
    for(i=0; i<nrows; i++)
    {
        GCGE_Printf("%f\n", b_H_Array[i]);
    }
    ierr = VecRestoreArrayRead(b_H, &b_H_Array);
    double *aux_h = vecs->aux_h;
    int num_aux_vec = vecs->num_aux_vec;
    //打印aux_h部分
    for(i=0; i<num_aux_vec; i++)
    {
        GCGE_Printf("%f\n", aux_h[i]);
    }
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
      ierr = MatSetValue(*A,II,II,4.0,INSERT_VALUES);
      if(II<N-1) {
      ierr = MatSetValue(*A,II,II+1,1.0,INSERT_VALUES);
      }
      if(II>0) {
      ierr = MatSetValue(*A,II,II-1,1.0,INSERT_VALUES);
      }
#if 0
      i = II/n; j = II-i*n;
      if (i>0) { ierr = MatSetValue(*A,II,II-n,-1.0,INSERT_VALUES); }
      if (i<m-1) { ierr = MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES); }
      if (j>0) { ierr = MatSetValue(*A,II,II-1,-1.0,INSERT_VALUES); }
      if (j<n-1) { ierr = MatSetValue(*A,II,II+1,-1.0,INSERT_VALUES); }
      ierr = MatSetValue(*A,II,II,4.0,INSERT_VALUES);
#endif
    }
    
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);
}
