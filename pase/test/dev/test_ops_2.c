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

void PASE_MatrixConvertDenseMatrix(PASE_Matrix pase_matrix, double **dense_mat, int *size);
void Print_DenseMatrix(double *dense_mat, int *size);
void PASE_MultiVectorConvertDenseMatrix(PASE_MultiVector vecs, double **dense_mat, int *size);
void DenseMatrixDotDenseMatrix(double *matA, double *matX, double **matY, int *sizeA, int *sizeX, int *sizeY);

int main(int argc, char* argv[])
{
#if GCGE_USE_MPI
    MPI_Init(&argc,  &argv);
#endif

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
                = pase_matrix->aux_hh[j*num_aux_vec+i];
        }
    }
    PASE_PrintMat(pase_matrix);

    //给pase辅助向量组x赋初值
    pase_ops->MultiVecSetRandomValue((void**)multi_vec_x, 0, n_multivec, pase_ops);
    PASE_PrintMultiVec(multi_vec_x, "multi_vec_x");

    printf ( "dense_matA----------------------------------------------\n" );

    double *dense_matA, *dense_matX, *dense_matY;
    int    sizeA[2], sizeX[2], sizeY[2];
    PASE_MatrixConvertDenseMatrix(pase_matrix, &dense_matA, sizeA);
    Print_DenseMatrix(dense_matA, sizeA);
    
    printf ( "dense_matX----------------------------------------------\n" );
    PASE_MultiVectorConvertDenseMatrix(multi_vec_x, &dense_matX, sizeX);
    Print_DenseMatrix(dense_matX, sizeX);

    printf ( "dense_matY = dense_matA * dense_matX--------------------\n" );
    DenseMatrixDotDenseMatrix(dense_matA, dense_matX, &dense_matY, sizeA, sizeX, sizeY);
    Print_DenseMatrix(dense_matY, sizeY);

    free(dense_matA);
    free(dense_matX);
    free(dense_matY);

    printf ( "--------MatDotMultiVec------------------------------------------\n" );
    int mv_s[2];
    int mv_e[2];
    mv_s[0] = 2;
    mv_e[0] = 4;
    mv_s[1] = 2;
    mv_e[1] = 4;
    pase_ops->MatDotMultiVec((void *)pase_matrix, (void **)multi_vec_x, 
            (void **)multi_vec_y, mv_s, mv_e, pase_ops);
    PASE_PrintMat(pase_matrix);
    PASE_PrintMultiVec(multi_vec_y, "mdmv_y");
    printf("mdmv_mv_s = [%d, %d];\n", mv_s[0], mv_s[1]);
    printf("mdmv_mv_e = [%d, %d];\n", mv_e[0], mv_e[1]);
    printf ( "--------MatDotMultiVec------------------------------------------\n" );

    pase_ops->MultiVecDestroy ((void ***)&multi_vec_x, n_multivec, pase_ops);
    pase_ops->MultiVecDestroy ((void ***)&multi_vec_y, n_multivec, pase_ops);

    PASE_MatrixDestroy(&pase_matrix, pase_ops);

    //释放矩阵空间
    CSR_MatFree(&A);

    GCGE_OPS_Free(&gcge_ops);
    PASE_OPS_Free(&pase_ops);

#if GCGE_USE_MPI
    MPI_Finalize();
#endif

    return 0;
}

void DenseMatrixDotDenseMatrix(double *matA, double *matX, double **matY, int *sizeA, int *sizeX, int *sizeY)
{
   int i, j, k;
   int lddA, lddX, lddY;
   sizeY[0] = sizeA[0];
   sizeY[1] = sizeX[1];
   lddA = sizeA[0];
   lddX = sizeX[0];
   lddY = sizeY[0];

   *matY = (double*)calloc(sizeY[0]*sizeY[1], sizeof(double));

   printf ( "size A = %d, %d\n", sizeA[0], sizeA[1]);
   printf ( "size X = %d, %d\n", sizeX[0], sizeX[1]);
   printf ( "size Y = %d, %d\n", sizeY[0], sizeY[1]);
   for (i = 0; i < sizeY[0]; ++i)
   {
      for (j = 0; j < sizeY[1]; ++j)
      {
//	 printf ( "i = %d, j = %d\n", i, j );
	 (*matY)[j*lddY+i] = 0;
	 for (k = 0; k < sizeA[1]; ++k)
	 {
	    (*matY)[j*lddY+i] += matA[k*lddA+i]*matX[j*lddX+k];
	 }
      }
   }
}

void PASE_MultiVectorConvertDenseMatrix(PASE_MultiVector vecs, double **dense_mat, int *size)
{
   int     num_vec = vecs->num_vec;
   int     num_aux_vec = vecs->num_aux_vec;

   CSR_VEC **b_H = (CSR_VEC**)(vecs->b_H);
   int     nrows = b_H[0]->size;

   int     ldd = nrows+num_aux_vec;
   *dense_mat = (double*)calloc(ldd*num_vec, sizeof(double));

   size[0] = ldd;
   size[1] = num_vec;

   int i, j;
   for(j=0; j<num_vec; j++)
   {
      for(i=0; i<nrows; i++)
      {
	 (*dense_mat)[ldd*j+i] = b_H[j]->Entries[i];
      }
   }

   double *aux_h = vecs->aux_h;
   for(j=0; j<num_vec; j++)
   {
      for(i=0; i<num_aux_vec; i++)
      {
	 (*dense_mat)[ldd*j+i+nrows] = aux_h[j*num_aux_vec+i];
      }
   }
}

/* 只适用于方阵, dense按列存储 */
void PASE_MatrixConvertDenseMatrix(PASE_Matrix pase_matrix, double **dense_mat, int *size)
{
    CSR_MAT *A_H = (CSR_MAT*)(pase_matrix->A_H);
    int     num_aux_vec = pase_matrix->num_aux_vec;
    int     nrows = A_H->N_Rows;
    int     ldd = nrows+num_aux_vec;
    *dense_mat = (double*)calloc(ldd*ldd, sizeof(double));
    int     row = 0;
    int     col = 0;
    int     i   = 0;
    int     j   = 0;
    int     *rowptr  = A_H->RowPtr;
    int     *kcol    = A_H->KCol;
    double  *entries = A_H->Entries;

    size[0] = ldd;
    size[1] = ldd;
    //将A_H部分赋值给dense_mat
    for(row=0; row<nrows; row++)
    {
        for(i=rowptr[row]; i<rowptr[row+1]; i++)
        {
            col = kcol[i];
            (*dense_mat)[col*ldd+row] = entries[i];
        }
    }
    //将aux_Hh部分赋值给dense_mat
    CSR_VEC **vecs = (CSR_VEC**)(pase_matrix->aux_Hh);
    for(i=0; i<num_aux_vec; i++)
    {
        for(row=0; row<nrows; row++)
        {
            col = nrows+i;
            (*dense_mat)[col*ldd+row] = vecs[i]->Entries[row];
            //对称化
            (*dense_mat)[row*ldd+col] = vecs[i]->Entries[row];
        }
    }
    //aux_hh部分
    for(i=0; i<num_aux_vec; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            col = nrows+i;
            row = nrows+j;
            (*dense_mat)[col*ldd+row] = pase_matrix->aux_hh[i*num_aux_vec+j];
        }
    }
}

void Print_DenseMatrix(double *dense_mat, int *size)
{
   int i, j;
   int ldd = size[0];
   for(i=0; i<size[0]; i++)
   {
      for(j=0; j<size[1]; j++)
      {
	 printf("%18.15f\t", dense_mat[j*ldd+i]);
      }
      printf ( "\n" );
   }
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
