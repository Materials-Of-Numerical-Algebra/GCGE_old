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
void PASE_PrintMultiVec(PASE_MultiVector vecs, char *name);
void PASE_PrintVec(PASE_Vector vecs, char *name);

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
    PASE_INT         n_multivec = 4;
    //GCGE_Printf ( "PASE_OPS->MultiVecCreateByMat\n" );
    pase_ops->MultiVecCreateByMat((void ***)(&multi_vec_x), n_multivec, 
            (void *)pase_matrix, pase_ops);
    pase_ops->MultiVecCreateByMat((void ***)(&multi_vec_y), n_multivec, 
            (void *)pase_matrix, pase_ops);

    //给pase辅助矩阵中向量组赋初值
    gcge_ops->MultiVecSetRandomValue(pase_matrix->aux_Hh, 
            0, num_aux_vec, gcge_ops);

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

    //给pase辅助向量组x赋初值
    pase_ops->MultiVecSetRandomValue((void**)multi_vec_x, 0, n_multivec, pase_ops);

    //printf ( "multi_vec_x\n");
    //PASE_PrintMultiVec(multi_vec_x);
    //printf ( "pase_matrix\n" );
    //PASE_PrintMat(pase_matrix);
    //gcge_ops->MultiVecPrint(pase_matrix->aux_Hh, n_multivec);

    PASE_Vector vec_x;
    PASE_Vector vec_y;
    pase_ops->VecCreateByMat((void**)(&vec_x), (void*)pase_matrix, pase_ops);
    pase_ops->VecCreateByVec((void**)(&vec_y), (void*)vec_x, pase_ops);
    pase_ops->VecSetRandomValue((void*)vec_x, 3, pase_ops);
    pase_ops->VecSetRandomValue((void*)vec_y, 2, pase_ops);

    //--------------------------------------------------------------
    PASE_PrintVec(vec_x, "vaxpby_x");
    PASE_PrintVec(vec_y, "vaxpby_y");
    double a = 2.1;
    double b = 5.4;
    /* It's ok. */
    pase_ops->VecAxpby(a, (void*)vec_x, b, (void*)vec_y, pase_ops);
    printf( "a = %f; b = %f;\n", a, b );
    PASE_PrintVec(vec_y, "vaxpby_y_new");

    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MatDotVec((void*)pase_matrix, (void*)vec_x, (void*)vec_y, pase_ops);
    PASE_PrintVec(vec_x, "mdv_x");
    PASE_PrintVec(vec_y, "mdv_y");

    //--------------------------------------------------------------
    double vvalue_ip = 0.0;
    /* It's ok. */
    pase_ops->VecInnerProd((void*)vec_x, (void*)vec_y, &vvalue_ip, pase_ops);
    PASE_PrintVec(vec_x, "vip_x");
    PASE_PrintVec(vec_y, "vip_y");
    printf("vip = %f;\n", vvalue_ip);

    int mv_s[2];
    int mv_e[2];
    mv_s[0] = 2;
    mv_e[0] = 4;
    mv_s[1] = 1;
    mv_e[1] = 3;
    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MatDotMultiVec((void *)pase_matrix, (void **)multi_vec_x, 
            (void **)multi_vec_y, mv_s, mv_e, pase_ops);
    PASE_PrintMat(pase_matrix);
    PASE_PrintMultiVec(multi_vec_x, "mdmv_x");
    PASE_PrintMultiVec(multi_vec_y, "mdmv_y");
    printf("mdmv_mv_s = [%d, %d];\n", mv_s[0]+1, mv_s[1]+1);
    printf("mdmv_mv_e = [%d, %d];\n", mv_e[0], mv_e[1]);

    //--------------------------------------------------------------
    PASE_PrintMultiVec(multi_vec_x, "mvaxpby_x");
    PASE_PrintMultiVec(multi_vec_y, "mvaxpby_y");
    /* It's ok. */
    pase_ops->MultiVecAxpby(a, (void**)multi_vec_x, b, (void**)multi_vec_y,
            mv_s, mv_e, pase_ops);
    printf( "a = %f; b = %f;\n", a, b );
    PASE_PrintMultiVec(multi_vec_y, "mvaxpby_y_new");
    printf("mvaxpby_mv_s = [%d, %d];\n", mv_s[0]+1, mv_s[1]+1);
    printf("mvaxpby_mv_e = [%d, %d];\n", mv_e[0], mv_e[1]);

    //--------------------------------------------------------------
    /* It's ok. */
    //pase_ops->MultiVecAxpbyColumn(a, (void**)multi_vec_x, 1, b, 
    //        (void**)multi_vec_y, 0, pase_ops);

    //--------------------------------------------------------------
    //multi_vec_y = multi_vec_x * coef + multi_vec_y
    double *coef = (double*)calloc(n_multivec*n_multivec, sizeof(double));
    for( j=0; j<n_multivec*n_multivec; j++ )
    {
       coef[j] = ((double)rand())/((double)RAND_MAX+1);
    }
    printf("coef = [\n");
    for(i=0; i<n_multivec; i++)
    {
        for(j=0; j<n_multivec; j++)
        {
            printf("%18.15f\t", coef[j*n_multivec+i]);
        }
        printf("\n");
    }
    printf("];\n");
    PASE_PrintMultiVec(multi_vec_x, "mvlb_x");
    PASE_PrintMultiVec(multi_vec_y, "mvlb_y");
    /* It's ok. */
    pase_ops->MultiVecLinearComb((void**)multi_vec_x, (void**)multi_vec_y,
            mv_s, mv_e, coef, n_multivec, 0, 1.0, 1.0, pase_ops);
    PASE_PrintMultiVec(multi_vec_y, "mvlb_y_new");
    free(coef);  coef = NULL;
    printf("mvlb_mv_s = [%d, %d];\n", mv_s[0]+1, mv_s[1]+1);
    printf("mvlb_mv_e = [%d, %d];\n", mv_e[0], mv_e[1]);
    printf("mvlb_alpha = %18.15lf; mvlb_beta = %18.15lf; \n", 1.0, 1.0);

    //--------------------------------------------------------------
    double *values_ip = (double*)calloc(n_multivec*n_multivec, sizeof(double));
    /* It's ok. */
    mv_s[0] = 2;
    mv_e[0] = 4;
    mv_s[1] = 1;
    mv_e[1] = 3;
    pase_ops->MultiVecInnerProd((void**)multi_vec_x, (void**)multi_vec_y,
            values_ip+mv_s[1]*n_multivec+mv_s[0], "nonsym", mv_s, mv_e, 
            n_multivec, 0, pase_ops);
    PASE_PrintMultiVec(multi_vec_x, "mvip_x");
    PASE_PrintMultiVec(multi_vec_y, "mvip_y");
    printf("values_ip = [\n");
    for(i=0; i<n_multivec; i++)
    {
        for(j=0; j<n_multivec; j++)
        {
            printf("%18.15f\t", values_ip[j*n_multivec+i]);
        }
        printf("\n");
    }
    printf("];\n");
    printf("mvip_mv_s = [%d, %d];\n", mv_s[0]+1, mv_s[1]+1);
    printf("mvip_mv_e = [%d, %d];\n", mv_e[0], mv_e[1]);

    //--------------------------------------------------------------
    /* It's ok. */
    pase_ops->MultiVecSwap((void**)multi_vec_x, (void**)multi_vec_y,
            mv_s, mv_e, pase_ops);
    PASE_PrintMultiVec(multi_vec_x, "mvswap_x");
    PASE_PrintMultiVec(multi_vec_y, "mvswap_y");

    pase_ops->VecDestroy ((void **)&vec_x, pase_ops);
    pase_ops->VecDestroy ((void **)&vec_y, pase_ops);

    pase_ops->MultiVecDestroy ((void ***)&multi_vec_x, n_multivec, pase_ops);
    pase_ops->MultiVecDestroy ((void ***)&multi_vec_y, n_multivec, pase_ops);

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
    printf("pase_matrix = [\n");
    for(i=0; i<ldd; i++)
    {
        for(j=0; j<ldd; j++)
        {
            printf("%18.15f\t", dense_mat[i*ldd+j]);
        }
        printf("\n");
    }
    printf("];\n");
    free(dense_mat); dense_mat = NULL;
}

void PASE_PrintMultiVec(PASE_MultiVector vecs, char *name)
{
    int num_vec = vecs->num_vec;
    int num_aux_vec = vecs->num_aux_vec;
    int i = 0;
    int j = 0;
    //打印CSR向量组部分
    CSR_VEC **b_H = (CSR_VEC**)(vecs->b_H);
    int nrows = b_H[0]->size;
    printf("%s = [\n", name);
    for(i=0; i<nrows; i++)
    {
        for(j=0; j<num_vec; j++)
        {
            printf("%18.15f\t", b_H[j]->Entries[i]);
        }
        printf("\n");
    }
    double *aux_h = vecs->aux_h;
    //打印aux_h部分
    for(i=0; i<num_aux_vec; i++)
    {
        for(j=0; j<num_vec; j++)
        {
            printf("%18.15f\t", aux_h[j*num_aux_vec+i]);
        }
        printf("\n");
    }
    printf("];\n");
}

void PASE_PrintVec(PASE_Vector vecs, char *name)
{
    int i = 0;
    int j = 0;
    //打印CSR向量组部分
    CSR_VEC *b_H = (CSR_VEC*)(vecs->b_H);
    int nrows = b_H->size;
    printf("%s = [\n", name);
    for(i=0; i<nrows; i++)
    {
        printf("%18.15f\n", b_H->Entries[i]);
    }
    double *aux_h = vecs->aux_h;
    int num_aux_vec = vecs->num_aux_vec;
    //打印aux_h部分
    for(i=0; i<num_aux_vec; i++)
    {
        printf("%18.15f\n", aux_h[i]);
    }
    printf("];\n");
}


/*
 
%Matlab 测试程序

%MatDotMultiVec
max_mdmv = max(max( pase_matrix * ...
    mdmv_x(:,mdmv_mv_s(1):mdmv_mv_e(1))...
    - mdmv_y(:,mdmv_mv_s(2):mdmv_mv_e(2))))

%MultiVecAxpby
max_mvaxpby = max(max( a * mvaxpby_x(:,mvaxpby_mv_s(1):mvaxpby_mv_e(1))+ ...
    b * mvaxpby_y(:,mvaxpby_mv_s(2):mvaxpby_mv_e(2)) ...
    - mvaxpby_y_new(:,mvaxpby_mv_s(2):mvaxpby_mv_e(2))))

%MultiVecInnerProd
max_mvip = max(max(mvip_x(:,mvip_mv_s(1):mvip_mv_e(1))' ...
    * mvip_y(:,mvip_mv_s(2):mvip_mv_e(2)) ...
    - values_ip(mvip_mv_s(1):mvip_mv_e(1),mvip_mv_s(2):mvip_mv_e(2))))

%MultiVecLinearComb
max_mvlb = max(max( mvlb_x(:,mvlb_mv_s(1):mvlb_mv_e(1)) ...
    * coef( 1:mvlb_mv_e(1)-mvlb_mv_s(1)+1,1:mvlb_mv_e(2)-mvlb_mv_s(2)+1 ) ...
    * mvlb_alpha...
    + mvlb_beta ...
    * mvlb_y(:,mvlb_mv_s(2):mvlb_mv_e(2)) ... 
    - mvlb_y_new(:,mvlb_mv_s(2):mvlb_mv_e(2)) ))

*/
