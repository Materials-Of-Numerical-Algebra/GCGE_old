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
#include "pase.h"
#include "gcge_app_csr.h"
void PASE_PrintMat(PASE_Matrix pase_matrix);
void PASE_PrintMultiVec(PASE_MultiVector vecs);
void PASE_PrintVec(PASE_Vector vecs);

int main(int argc, char* argv[])
{
    srand(1);

    //创建矩阵
    const char *file_A = "../data/testA";
    CSR_MAT *A = CSR_ReadMatFile(file_A);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_CSR_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);

    //创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);
    PASE_OPS_Setup(pase_ops);

    //创建pase辅助矩阵
    PASE_Matrix pase_matrix;
    PASE_INT    num_aux_vec = 2;
    PASE_MatrixCreate(&pase_matrix, num_aux_vec, A, pase_ops);

    //创建pase辅助向量组
    PASE_MultiVector multi_vec_x;
    PASE_MultiVector multi_vec_y;
    //GCGE_Printf ( "PASE_OPS->MultiVecCreateByMat\n" );
    pase_ops->MultiVecCreateByMat((void ***)(&multi_vec_x), num_aux_vec, 
            (void *)pase_matrix, pase_ops);
    pase_ops->MultiVecCreateByMat((void ***)(&multi_vec_y), num_aux_vec, 
            (void *)pase_matrix, pase_ops);

    //给pase辅助矩阵中向量组赋初值
    //printf ( "set random value for aux_Hh\n" );
    gcge_ops->MultiVecSetRandomValue(pase_matrix->aux_Hh, 
            0, num_aux_vec, gcge_ops);

    //printf("set random value for aux_hh\n");
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
#if 0
    printf("%f, %f, %f, %f\n", 
            pase_matrix->aux_hh[0],
            pase_matrix->aux_hh[1],
            pase_matrix->aux_hh[2],
            pase_matrix->aux_hh[3]);
#endif

    //printf("set random value for x\n");
    //给pase辅助向量组x赋初值
    pase_ops->MultiVecSetRandomValue((void**)multi_vec_x, 0, num_aux_vec, pase_ops);

    //printf ( "multi_vec_x\n");
    //PASE_PrintMultiVec(multi_vec_x);
    //printf ( "pase_matrix\n" );
    //PASE_PrintMat(pase_matrix);
    //gcge_ops->MultiVecPrint(pase_matrix->aux_Hh, num_aux_vec);

    PASE_Vector vec_x;
    PASE_Vector vec_y;
    pase_ops->VecCreateByMat((void**)(&vec_x), (void*)pase_matrix, pase_ops);
    pase_ops->VecCreateByVec((void**)(&vec_y), (void*)vec_x, pase_ops);
    pase_ops->VecSetRandomValue((void*)vec_x, 3, pase_ops);
    pase_ops->VecSetRandomValue((void*)vec_y, 2, pase_ops);

    //--------------------------------------------------------------
    printf("vec_x:\n");
    PASE_PrintVec(vec_x);
    printf("vec_y:\n");
    PASE_PrintVec(vec_y);
    double a = 2.1;
    double b = 5.4;
    /* It's ok. */
    pase_ops->VecAxpby(a, (void*)vec_x, b, (void*)vec_y, pase_ops);
    printf( "vec_y = %f * vec_x + %f * vec_y\n", a, b );
    PASE_PrintVec(vec_y);

    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MatDotVec((void*)pase_matrix, (void*)vec_x, (void*)vec_y, pase_ops);
    printf("vec_y = pase_matrix * vec_x:\n");
    printf ( "pase_matrix\n" );
    PASE_PrintMat(pase_matrix);
    printf("vec_x:\n");
    PASE_PrintVec(vec_x);
    printf("vec_y:\n");
    PASE_PrintVec(vec_y);

    //--------------------------------------------------------------
    double value_ip = 0.0;
    /* It's ok. */
    pase_ops->VecInnerProd((void*)vec_x, (void*)vec_y, &value_ip, pase_ops);
    printf("vec_x:\n");
    PASE_PrintVec(vec_x);
    printf("vec_y:\n");
    PASE_PrintVec(vec_y);
    printf("value_ip = %f (vec_x^T * vec_y)\n", value_ip);

    int mv_s[2];
    int mv_e[2];
    mv_s[0] = 0;
    mv_e[0] = num_aux_vec;
    mv_s[1] = 0;
    mv_e[1] = num_aux_vec;
    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MatDotMultiVec((void *)pase_matrix, (void **)multi_vec_x, 
            (void **)multi_vec_y, mv_s, mv_e, pase_ops);
    printf ( "multi_vec_y = pase_matrix * multi_vec_x\n");
    PASE_PrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    printf("multi_vec_x: \n");
    PASE_PrintMultiVec(multi_vec_x);
    printf("multi_vec_y: \n");
    PASE_PrintMultiVec(multi_vec_y);
    /* It's ok. */
    pase_ops->MultiVecAxpby(a, (void**)multi_vec_x, b, (void**)multi_vec_y,
            mv_s, mv_e, pase_ops);
    printf( "multi_vec_y = %f * multi_vec_x + %f * multi_vec_y\n", a, b );
    PASE_PrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    printf("multi_vec_x: \n");
    PASE_PrintMultiVec(multi_vec_x);
    printf("multi_vec_y: \n");
    PASE_PrintMultiVec(multi_vec_y);
    /* It's ok. */
    pase_ops->MultiVecAxpbyColumn(a, (void**)multi_vec_x, 1, b, 
            (void**)multi_vec_y, 0, pase_ops);
    printf( "multi_vec_y[0] = %f * multi_vec_x[1] + %f * multi_vec_y[0]\n", a, b );
    PASE_PrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    //multi_vec_y = multi_vec_x * coef + multi_vec_y
    double coef[4] = {1.2, 2.3, 3.1, 4.4};
    printf("multi_vec_x: \n");
    PASE_PrintMultiVec(multi_vec_x);
    printf("multi_vec_y: \n");
    PASE_PrintMultiVec(multi_vec_y);
    printf("coef:\n1.2  3.1\n2.3  4.4\n");
    /* It's ok. */
    pase_ops->MultiVecLinearComb((void**)multi_vec_x, (void**)multi_vec_y,
            mv_s, mv_e, coef, num_aux_vec, 0, 1.0, 1.0, pase_ops);
    printf("multi_vec_y = multi_vec_x * coef + multi_vec_y: \n");
    PASE_PrintMultiVec(multi_vec_y);

    //--------------------------------------------------------------
    double values_ip[4];
    /* It's ok. */
    pase_ops->MultiVecInnerProd((void**)multi_vec_x, (void**)multi_vec_y,
            (double*)values_ip, "nonsym", mv_s, mv_e, num_aux_vec,
            0, pase_ops);
    printf("multi_vec_x: \n");
    PASE_PrintMultiVec(multi_vec_x);
    printf("multi_vec_y: \n");
    PASE_PrintMultiVec(multi_vec_y);
    printf("values_ip:\n%f\t%f\n%f\t%f\n",
            values_ip[0], values_ip[2], 
            values_ip[1], values_ip[3]);

    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MultiVecSwap((void**)multi_vec_x, (void**)multi_vec_y,
            mv_s, mv_e, pase_ops);
    printf("multi_vec_x: \n");
    PASE_PrintMultiVec(multi_vec_x);
    printf("multi_vec_y: \n");
    PASE_PrintMultiVec(multi_vec_y);

    pase_ops->VecDestroy ((void **)&vec_x, pase_ops);
    pase_ops->VecDestroy ((void **)&vec_y, pase_ops);

    pase_ops->MultiVecDestroy ((void ***)&multi_vec_x, num_aux_vec, pase_ops);
    pase_ops->MultiVecDestroy ((void ***)&multi_vec_y, num_aux_vec, pase_ops);

    PASE_MatrixDestroy(&pase_matrix, pase_ops);

    //释放矩阵空间
    CSR_MatFree(&A);

    GCGE_OPS_Free(&gcge_ops);
    PASE_OPS_Free(&pase_ops);

    return 0;
}

void PASE_PrintMat(PASE_Matrix pase_matrix)
{
    CSR_MAT *A_H = (CSR_MAT*)(pase_matrix->A_H);
    int     num_aux_vec = pase_matrix->num_aux_vec;
    int     nrows = A_H->N_Rows;
    int     ldd = nrows+num_aux_vec;
    double  *dense_mat = (double*)calloc(ldd*ldd, sizeof(double));
    int     row = 0;
    int     col = 0;
    int     i   = 0;
    int     j   = 0;
    int     *rowptr  = A_H->RowPtr;
    int     *kcol    = A_H->KCol;
    double  *entries = A_H->Entries;
    //将A_H部分赋值给dense_mat
    for(row=0; row<nrows; row++)
    {
        for(i=rowptr[row]; i<rowptr[row+1]; i++)
        {
            col = kcol[i];
            dense_mat[col*ldd+row] = entries[i];
        }
    }
    //将aux_Hh部分赋值给dense_mat
    CSR_VEC **vecs = (CSR_VEC**)(pase_matrix->aux_Hh);
    for(i=0; i<num_aux_vec; i++)
    {
        for(row=0; row<nrows; row++)
        {
            col = nrows+i;
            dense_mat[col*ldd+row] = vecs[i]->Entries[row];
            //对称化
            dense_mat[row*ldd+col] = vecs[i]->Entries[row];
        }
    }
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
    //printf("pase_matrix: n: %d\n",ldd);
    for(i=0; i<ldd; i++)
    {
        for(j=0; j<ldd; j++)
        {
            printf("%f\t", dense_mat[i*ldd+j]);
        }
        printf("\n");
    }
    free(dense_mat); dense_mat = NULL;
}

void PASE_PrintMultiVec(PASE_MultiVector vecs)
{
    int num_aux_vec = vecs->num_aux_vec;
    int i = 0;
    int j = 0;
    //打印CSR向量组部分
    CSR_VEC **b_H = (CSR_VEC**)(vecs->b_H);
    int nrows = b_H[0]->size;
    for(i=0; i<nrows; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            printf("%f\t", b_H[j]->Entries[i]);
        }
        printf("\n");
    }
    double *aux_h = vecs->aux_h;
    //打印aux_h部分
    for(i=0; i<num_aux_vec; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            printf("%f\t", aux_h[j*num_aux_vec+i]);
        }
        printf("\n");
    }
}

void PASE_PrintVec(PASE_Vector vecs)
{
    int i = 0;
    int j = 0;
    //打印CSR向量组部分
    CSR_VEC *b_H = (CSR_VEC*)(vecs->b_H);
    int nrows = b_H->size;
    for(i=0; i<nrows; i++)
    {
        printf("%f\n", b_H->Entries[i]);
    }
    double *aux_h = vecs->aux_h;
    int num_aux_vec = vecs->num_aux_vec;
    //打印aux_h部分
    for(i=0; i<num_aux_vec; i++)
    {
        printf("%f\n", aux_h[i]);
    }
}
