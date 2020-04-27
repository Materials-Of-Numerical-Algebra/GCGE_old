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
#include "gcge_app_csr.h"
#include "gcge_app_pase.h"
#include "pase.h"

void PASE_PrintMat(PASE_Matrix pase_matrix);
void PASE_PrintMultiVec(PASE_MultiVector vecs);
void PASE_PrintVec(PASE_Vector vecs);

int main(int argc, char* argv[])
{
#if GCGE_USE_MPI
    MPI_Init(&argc,  &argv);
#endif
    srand(1);

    //创建矩阵
    //const char *file_A = "../data/test_csr_pase_A_3";
    //const char *file_A = "../data/testA";
    //const char *file_A = "../data/A_3.txt";
    const char *file_A = "../data/A_5.txt";
    CSR_MAT *A = CSR_ReadMatFile(file_A);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_CSR_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);

    //创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);

    //创建pase辅助矩阵
    PASE_Matrix pase_mat_A;
    PASE_INT    num_aux_vec = 2;
    PASE_MatrixCreate(&pase_mat_A, num_aux_vec, A, pase_ops);

    CSR_VEC **vecs = (CSR_VEC**)(pase_mat_A->aux_Hh);
#if 0
    vecs[0]->Entries[0] = 1;
    vecs[0]->Entries[1] = 2;
    vecs[0]->Entries[2] = 3;
    vecs[1]->Entries[0] = 2;
    vecs[1]->Entries[1] = 1;
    vecs[1]->Entries[2] = 2;

    pase_mat_A->aux_hh[0] = 15;
    pase_mat_A->aux_hh[1] = 4;
    pase_mat_A->aux_hh[2] = 4;
    pase_mat_A->aux_hh[3] = 20;
#else

    memset(vecs[0]->Entries, 0.0, 1089*sizeof(double));
    memset(vecs[1]->Entries, 0.0, 1089*sizeof(double));
    pase_mat_A->aux_hh[0] = 1.0;
    pase_mat_A->aux_hh[1] = 0.0;
    pase_mat_A->aux_hh[2] = 0.0;
    pase_mat_A->aux_hh[3] = 1.0;
#endif

    int nev = 5;
    GCGE_Printf("line 74 in main\n");
    GCGE_SOLVER *csr_pase_solver = GCGE_PASE_Solver_Init(pase_mat_A, 
            NULL, nev, argc, argv, pase_ops);

    GCGE_SOLVER_Solve(csr_pase_solver);  

    GCGE_SOLVER_Free_All(&csr_pase_solver);

    PASE_MatrixDestroy(&pase_mat_A, pase_ops);

    //释放矩阵空间
    CSR_MatFree(&A);

    GCGE_OPS_Free(&gcge_ops);
    PASE_OPS_Free(&pase_ops);

#if GCGE_USE_MPI
    MPI_Finalize();
#endif
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
