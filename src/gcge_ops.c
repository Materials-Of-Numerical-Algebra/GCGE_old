/*
 * =====================================================================================
 *
 *       Filename:  gcge_ops.c
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

#include "gcge_ops.h"
/* if use mpi, multivec inner prod will be improved by MPI_Typre_vector and MPI_Op_create */
#if GCGE_USE_MPI
GCGE_INT SIZE_B, SIZE_E, LDA;
void user_fn_submatrix_sum(double *in,  double *inout,  int *len,  MPI_Datatype* dptr)
{
    int i, j;
    double *b,  *a;
    double one = 1.0;
    int    inc = 1;
    for (i = 0; i < *len; ++i)
    {
        for (j = 0; j < SIZE_B; ++j)
        {
            b = inout+j*LDA;
            a = in+j*LDA;
            daxpy_(&SIZE_E, &one, a, &inc, b, &inc);
            /*
               for (k = 0; k < SIZE_E; ++k)
               {
               b[k] += a[k];
               }
               */
        }
    }
}  
#endif

//MultiVec默认值的设定是要依赖Get与Restore这两个函数的
void GCGE_Default_GetVecFromMultiVec(void **V, GCGE_INT j, void **x, GCGE_OPS *ops)
{
    *x = V[j];
}

void GCGE_Default_RestoreVecForMultiVec(void **V, GCGE_INT j, void **x, GCGE_OPS *ops)
{
    V[j] = *x;
}

void GCGE_Default_MultiVecCreateByVec(void ***multi_vec, GCGE_INT n_vec, void *vec, GCGE_OPS *ops)
{
    GCGE_INT i = 0;
    *multi_vec = (void**)malloc(n_vec*sizeof(void*));
    for(i=0; i<n_vec; i++)
    {
        ops->VecCreateByVec((*multi_vec)+i, vec, ops);
    }
}

void GCGE_Default_MultiVecCreateByMat(void ***multi_vec, GCGE_INT n_vec, void *mat, GCGE_OPS *ops)
{
    GCGE_INT i = 0;
    *multi_vec = (void**)malloc(n_vec*sizeof(void*));
    for(i=0; i<n_vec; i++)
    {
        ops->VecCreateByMat((*multi_vec)+i, mat, ops);
    }
}

void GCGE_Default_MultiVecCreateByMultiVec(void ***multi_vec, GCGE_INT n_vec, void **init_vec, GCGE_OPS *ops)
{
    GCGE_INT i = 0;
    *multi_vec = (void**)malloc(n_vec*sizeof(void*));
    for(i=0; i<n_vec; i++)
    {
        ops->VecCreateByVec((*multi_vec)+i, init_vec[0], ops);
    }
}

void GCGE_Default_MultiVecDestroy(void ***MultiVec, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    GCGE_INT i = 0;
    for(i=0; i<n_vec; i++)
    {
        ops->VecDestroy(&(*MultiVec)[i], ops);  (*MultiVec)[i] = NULL;
    }
    free(*MultiVec); *MultiVec = NULL;
}

void GCGE_Default_MultiVecSetRandomValue(void **multi_vec, GCGE_INT start, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    GCGE_INT i = 0;
    void     *x;
    for(i=0; i<n_vec; i++)
    {
        ops->GetVecFromMultiVec(multi_vec, start + i, &x, ops);
        ops->VecSetRandomValue(x, ops);
        ops->RestoreVecForMultiVec(multi_vec, start + i, &x, ops);
    }//end for i iteration
}//end for this subprogram

//矩阵 mat 乘多个向量: y(:,start[1]:end[1]) = mat * x(:, start[0]:end[0]) 
void GCGE_Default_MatDotMultiVec(void *mat, void **x, void **y, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    GCGE_INT i = 0;
    GCGE_INT n_vec = end[0]-start[0];
    void     *xs, *ys;
    for(i=0; i<n_vec; i++)
    {
        ops->GetVecFromMultiVec(x, start[0]+i, &xs, ops);
        ops->GetVecFromMultiVec(y, start[1]+i, &ys, ops);
        ops->MatDotVec(mat, xs, ys, ops);
        ops->RestoreVecForMultiVec(x, start[0]+i, &xs, ops);
        ops->RestoreVecForMultiVec(y, start[1]+i, &ys, ops);
    }
}

//矩阵 mat 的转置 乘多个向量: y(:,start[1]:end[1]) = mat * x(:, start[0]:end[0]) 
void GCGE_Default_MatTransposeDotMultiVec(void *mat, void **x, void **y, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    GCGE_INT i = 0;
    GCGE_INT n_vec = end[0]-start[0];
    void     *xs, *ys;
    for(i=0; i<n_vec; i++)
    {
        ops->GetVecFromMultiVec(x, start[0]+i, &xs, ops);
        ops->GetVecFromMultiVec(y, start[1]+i, &ys, ops);
        ops->MatTransposeDotVec(mat, xs, ys, ops);
        ops->RestoreVecForMultiVec(x, start[0]+i, &xs, ops);
        ops->RestoreVecForMultiVec(y, start[1]+i, &ys, ops);
    }
}

//多个向量组的线性组合: y(:, start[1]:end[1]) = a*x(:,start[0]:end[0]) + b*y(:,start[1]:end[1])
void GCGE_Default_MultiVecAxpby(GCGE_DOUBLE a, void **x, GCGE_DOUBLE b, void **y, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    GCGE_INT i = 0;
    GCGE_INT n_vec = end[0]-start[0];
    void     *xs, *ys;
    for(i=0; i<n_vec; i++)
    {
        ops->GetVecFromMultiVec(x, start[0]+i, &xs, ops);
        ops->GetVecFromMultiVec(y, start[1]+i, &ys, ops);
        ops->VecAxpby(a, x[start[0]+i], b, y[start[1]+i], ops);
        ops->RestoreVecForMultiVec(x, start[0]+i, &xs, ops);
        ops->RestoreVecForMultiVec(y, start[1]+i, &ys, ops);
    }
}

//对向量进行线性组合: y(:, col_y) = a*x(:,col_x) + b*y(:,col_y)
void GCGE_Default_MultiVecAxpbyColumn(GCGE_DOUBLE a, void **x, GCGE_INT col_x, 
        GCGE_DOUBLE b, void **y, GCGE_INT col_y, struct GCGE_OPS_ *ops)
{
    void     *xs, *ys;
    ops->GetVecFromMultiVec(x, col_x, &xs, ops);
    if(col_x == col_y)
    {
        ys = xs;
    }
    else
    {
        ops->GetVecFromMultiVec(y, col_y, &ys, ops);
    }
    ops->VecAxpby(a, xs, b, ys, ops);
    ops->RestoreVecForMultiVec(x, col_x, &xs, ops);
    if(col_x != col_y)
    {
        ops->RestoreVecForMultiVec(y, col_y, &ys, ops);
    }
}

/* vec_y[j] = \sum_{i=sx}^{ex} vec_x[i] a[i][j] */
//对连续的向量组进行线性组合: y(:,start[1]:end[1]) = x(:,start[0]:end[0])*a
//a 为一个系数矩阵, 按列排列的
void GCGE_Default_MultiVecLinearComb(void **x, void **y, GCGE_INT *start, GCGE_INT *end,
        GCGE_DOUBLE *a, GCGE_INT lda, GCGE_INT if_Vec, 
        GCGE_DOUBLE alpha, GCGE_DOUBLE beta, struct GCGE_OPS_ *ops)
{
    GCGE_INT idx_x = 0;
    GCGE_INT idx_y = 0;
    GCGE_INT nx = end[0]-start[0];
    GCGE_INT ny = end[1]-start[1];
    void     *xs, *ys;
    for(idx_y=start[1]; idx_y<end[1]; idx_y++)
    {
        //取出y的相应的列
        ops->GetVecFromMultiVec(y, idx_y,    &ys, ops);
        //ops->VecAxpby(0.0, ys, 0.0, ys);
        //取出x的相应的列
        ops->GetVecFromMultiVec(x, start[0], &xs, ops);
        //ys = a(0,idy)*xs + 0.0*ys
        //ops->VecAxpby(a[idx_y*lda+start[0]], xs, 0.0, ys);
        ops->VecAxpby(a[(idx_y-start[1])*lda], xs, beta, ys, ops);
        //把xs返回给x(:,start[0])
        ops->RestoreVecForMultiVec(x, start[0], &xs, ops);
        //下面进行迭代
        for(idx_x = start[0]+1; idx_x<end[0]; idx_x++)
        {
            ops->GetVecFromMultiVec(x, idx_x, &xs, ops);
            ops->VecAxpby(a[(idx_y-start[1])*lda+idx_x-start[0]], xs, 1.0, ys, ops);
            ops->RestoreVecForMultiVec(x, idx_x, &xs, ops);
        }//end for idy
        //返回ys
        ops->RestoreVecForMultiVec(y, idx_y, &ys, ops);
    }//end for idx_x
}//end for this subprogram

// lda : leading dimension of matrix a
// a(0: end[0]-start[0], 0:end[1]-start[1]) = V(:,start[0]:end[0])^T * W(:,start[1]:end[1])
// 做内积得到值是存在矩阵a的相应的位置
void GCGE_Default_MultiVecInnerProd(void **V, void **W, GCGE_DOUBLE *a, char *is_sym, 
        GCGE_INT *start, GCGE_INT *end, GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops)
{
    //lda表示做完相减存储到V的位置？
    GCGE_INT iv = 0, iw = 0;
    GCGE_DOUBLE *p = a;
    void     *vs, *ws;
    /* TODO some error 确实*/
    if(strcmp(is_sym, "sym") == 0)
    {
        //表示矩阵a是对称的，并且只存储他的上三角部分
        GCGE_INT length = end[1] - start[1];
        //p = a;
        //iw: 表示的是列号
        for(iw=0; iw<length; iw++)
        {
            ops->GetVecFromMultiVec(W, iw+start[1], &ws, ops);
            //计算的是矩阵的上三角部分
            for(iv=0; iv<=iw; iv++, p++)
            {
                ops->GetVecFromMultiVec(V, iv+start[0], &vs, ops);
                //ops->VecInnerProd(vs, ws, p);
                ops->VecLocalInnerProd(vs, ws, p, ops);
                ops->RestoreVecForMultiVec(V, iv+start[0], &vs, ops);
                //printf ( "iw = %d, iv = %d\n", iw, iv );
            }//end for iv
            ops->RestoreVecForMultiVec(W, iw+start[1], &ws, ops);
            p += lda - (iw+1);
        }//end for iw
        //对称化: 也就是产生矩阵的下三角部分，这里只需要用到对称行就可以直接得到
        //printf ( "-----------------\n" );
        for(iw=0; iw<length; iw++)
        {
            for(iv=iw+1; iv<length; iv++)
            {
                //printf ( "iw = %d, iv = %d\n", iw, iv );
                //a(iv,iw) = a(iw,iv)
                a[iw*lda+iv] = a[iv*lda+iw];
            }//end for iv（行号）
        }//end for iw(列号)
    }//处理完对称矩阵的情况
    else
    {
        //这里是处理非对称矩阵情况
        //jvw： 记录的是每一列，我们没有计算的那些元素的个数
        GCGE_INT jvw = lda - (end[0] - start[0]);
        //p = a;
        for(iw=start[1]; iw<end[1]; iw++)
        {
            //iw列
            ops->GetVecFromMultiVec(W, iw, &ws, ops);
            //iv行：从 start[0]到end[0]都需要计算
            for(iv=start[0]; iv<end[0]; iv++, p++)
            {
                ops->GetVecFromMultiVec(V, iv, &vs, ops);
                //ops->VecInnerProd(vs, ws, p);
                ops->VecLocalInnerProd(vs, ws, p, ops);
                ops->RestoreVecForMultiVec(V, iv, &vs, ops);
            }//end for iv
            ops->RestoreVecForMultiVec(W, iw, &ws, ops);
            //把p的位置进行相应地移动
            p += jvw;
        }//end for iw
    }//处理完了非对称的情况

#if GCGE_USE_MPI

    SIZE_B = end[1]-start[1];
    SIZE_E = end[0]-start[0];
    LDA    = lda;

    MPI_Datatype SUBMATRIX;
    MPI_Type_vector(SIZE_B, SIZE_E, LDA, MPI_DOUBLE, &SUBMATRIX);
    MPI_Type_commit(&SUBMATRIX);

    MPI_Op SUBMATRIX_SUM;
    MPI_Op_create((MPI_User_function*)user_fn_submatrix_sum, 1, &SUBMATRIX_SUM);
    MPI_Allreduce(MPI_IN_PLACE, a, 1, SUBMATRIX, SUBMATRIX_SUM, MPI_COMM_WORLD);
    MPI_Op_free(&SUBMATRIX_SUM);
    MPI_Type_free(&SUBMATRIX);

#endif

}//end for this subprogram

// lda : leading dimension of matrix a
// a(0: end[0]-start[0], 0:end[1]-start[1]) = V(:,start[0]:end[0])^T * W(:,start[1]:end[1])
// 做内积得到值是存在矩阵a的相应的位置
void GCGE_Default_MultiVecInnerProdLocal(void **V, void **W, GCGE_DOUBLE *a, char *is_sym, 
        GCGE_INT *start, GCGE_INT *end, GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops)
{
    //lda表示做完相减存储到V的位置？
    GCGE_INT iv = 0, iw = 0;
    GCGE_DOUBLE *p = a;
    void     *vs, *ws;
    /* TODO some error 确实*/
    if(strcmp(is_sym, "sym") == 0)
    {
        //表示矩阵a是对称的，并且只存储他的上三角部分
        GCGE_INT length = end[1] - start[1];
        //p = a;
        //iw: 表示的是列号
        for(iw=0; iw<length; iw++)
        {
            ops->GetVecFromMultiVec(W, iw+start[1], &ws, ops);
            //计算的是矩阵的上三角部分
            for(iv=0; iv<=iw; iv++, p++)
            {
                ops->GetVecFromMultiVec(V, iv+start[0], &vs, ops);
                //ops->VecInnerProd(vs, ws, p);
                ops->VecLocalInnerProd(vs, ws, p, ops);
                ops->RestoreVecForMultiVec(V, iv+start[0], &vs, ops);
                //printf ( "iw = %d, iv = %d\n", iw, iv );
            }//end for iv
            ops->RestoreVecForMultiVec(W, iw+start[1], &ws, ops);
            p += lda - (iw+1);
        }//end for iw
        //对称化: 也就是产生矩阵的下三角部分，这里只需要用到对称行就可以直接得到
        //printf ( "-----------------\n" );
        for(iw=0; iw<length; iw++)
        {
            for(iv=iw+1; iv<length; iv++)
            {
                //printf ( "iw = %d, iv = %d\n", iw, iv );
                //a(iv,iw) = a(iw,iv)
                a[iw*lda+iv] = a[iv*lda+iw];
            }//end for iv（行号）
        }//end for iw(列号)
    }//处理完对称矩阵的情况
    else
    {
        //这里是处理非对称矩阵情况
        //jvw： 记录的是每一列，我们没有计算的那些元素的个数
        GCGE_INT jvw = lda - (end[0] - start[0]);
        //p = a;
        for(iw=start[1]; iw<end[1]; iw++)
        {
            //iw列
            ops->GetVecFromMultiVec(W, iw, &ws, ops);
            //iv行：从 start[0]到end[0]都需要计算
            for(iv=start[0]; iv<end[0]; iv++, p++)
            {
                ops->GetVecFromMultiVec(V, iv, &vs, ops);
                //ops->VecInnerProd(vs, ws, p);
                ops->VecLocalInnerProd(vs, ws, p, ops);
                ops->RestoreVecForMultiVec(V, iv, &vs, ops);
            }//end for iv
            ops->RestoreVecForMultiVec(W, iw, &ws, ops);
            //把p的位置进行相应地移动
            p += jvw;
        }//end for iw
    }//处理完了非对称的情况

}//end for this subprogram


//把V_1和V_2的相应的列组交换: 即： V_1(:,start[0]:end[0])与V_2(:,start[1]:end[1])相互狡猾
void GCGE_Default_MultiVecSwap(void **V_1, void **V_2, GCGE_INT *start, GCGE_INT *end, 
        struct GCGE_OPS_ *ops)
{
    GCGE_INT i = 0;
    GCGE_INT size = end[0] - start[0];
    void     *vs, *ws;
    for(i=0; i<size; i++)
    {
        /*TODO*/
        //这里交换是否可以省掉一步???
        ops->GetVecFromMultiVec(V_1, start[0]+i, &vs, ops);
        ops->GetVecFromMultiVec(V_2, start[1]+i, &ws, ops);
        ops->RestoreVecForMultiVec(V_1, start[0]+i, &ws, ops);
        ops->RestoreVecForMultiVec(V_2, start[1]+i, &vs, ops);
    }
}

//进行稠密矩阵特征值求解 (这里还是按照Laplack的要求进行设置的)
void GCGE_Default_DenseMatEigenSolver( GCGE_DOUBLE *a, GCGE_INT lda, 
        GCGE_INT nrows, GCGE_DOUBLE *eval, GCGE_DOUBLE *evec, GCGE_INT lde, 
        GCGE_INT il, GCGE_INT iu, GCGE_INT *iwork, GCGE_DOUBLE *dwork)
{
    //调用Laplack的subroutine进行计算dsyevx_
    
    char        *jobz  = "V";
    char        *range = "I";
    char        *uplo  = "U";
    GCGE_DOUBLE vl     = 0.0;
    GCGE_DOUBLE vu     = 0.0;
    GCGE_DOUBLE abstol = 0.0;
    GCGE_INT    nev    = iu-il+1;
    GCGE_INT    lwork  = 8*nrows;
    GCGE_INT    liwork = 5*nrows;
    GCGE_INT    *ifail = iwork+5*nrows;
    GCGE_INT    info   = 0;

    dsyevx_(jobz, range, uplo, &nrows, a, &lda, &vl, &vu, &il, &iu, 
            &abstol, &nev, eval, evec, &lde, dwork, &lwork, 
            iwork, ifail, &info);

    //对dsyevx:
    /* param:
     * jobz: "V"，表示计算特征值及特征向量
     * range: "I"，表示计算第il到iu个特征对
     * uplo: "U"，表示给的是对称矩阵对上三角部分
     * nrows: 矩阵A的行数
     * a: 矩阵A
     * lda: 矩阵A的leading dimension
     * vl: 不用
     * vu: 不用
     * il: 个数的起始位置
     * iu: 个数的终点位置
     * abstol:
     * nev: 特征值个数
     * eval: 特征值
     * evec: 特征向量
     * lde: evec的leading dimension
     * isuppz: -----不要设，跳过这个参数------
     * work, 
     * lwork: 8*N
     * iwork, 
     * liwork: -----不要设，跳过这个参数------
     * ifail: int*, 长度2*nev
     * info
     */
    //对dsyevr:
    /* param:
     * jobz: "V"，表示计算特征值及特征向量
     * range: "I"，表示计算第il到iu个特征对
     * uplo: "U"，表示给的是对称矩阵对上三角部分
     * nrows: 矩阵A的行数
     * a: 矩阵A
     * lda: 矩阵A的leading dimension
     * vl: 不用
     * vu: 不用
     * il: 个数的起始位置
     * iu: 个数的终点位置
     * abstol:
     * nev: 特征值个数
     * eval: 特征值
     * evec: 特征向量
     * lde: evec的leading dimension
     * isuppz: int *
     * work, 
     * lwork: 26*N
     * iwork, 
     * liwork: 10*N
     * ifail: -----不要设，跳过这个参数------
     * info
     */
    //对dsyev:
    /* param:
     * jobz: "V"，表示计算特征值及特征向量
     * range: -----不要设，跳过这个参数------
     * uplo: "U"，表示给的是对称矩阵对上三角部分
     * nrows: 矩阵A的行数
     * a: 矩阵A
     * lda: 矩阵A的leading dimension
     * vl: -----不要设，跳过这个参数------
     * vu: -----不要设，跳过这个参数------
     * il: -----不要设，跳过这个参数------
     * iu: -----不要设，跳过这个参数------
     * abstol:-----不要设，跳过这个参数------
     * nev: -----不要设，跳过这个参数------
     * eval: 特征值
     * evec: -----不要设，跳过这个参数------
     * lde: -----不要设，跳过这个参数------
     * isuppz: -----不要设，跳过这个参数------
     * work, 
     * lwork: 3*N
     * iwork, -----不要设，跳过这个参数------
     * liwork: -----不要设，跳过这个参数------
     * ifail: -----不要设，跳过这个参数------
     * info
     */
    //对dsyevd:
    /* param:
     * jobz: "V"，表示计算特征值及特征向量
     * range: -----不要设，跳过这个参数------
     * uplo: "U"，表示给的是对称矩阵对上三角部分
     * nrows: 矩阵A的行数
     * a: 矩阵A
     * lda: 矩阵A的leading dimension
     * vl: -----不要设，跳过这个参数------
     * vu: -----不要设，跳过这个参数------
     * il: -----不要设，跳过这个参数------
     * iu: -----不要设，跳过这个参数------
     * abstol:-----不要设，跳过这个参数------
     * nev: -----不要设，跳过这个参数------
     * eval: 特征值
     * evec: -----不要设，跳过这个参数------
     * lde: -----不要设，跳过这个参数------
     * isuppz: -----不要设，跳过这个参数------
     * work, 
     * lwork: 1+6*N+2*N**2
     * iwork, 
     * liwork: 3+5*N
     * ifail: -----不要设，跳过这个参数------
     * info
     */
    /*
     *lwork = 26*nrows;
     *liwork = 10*nrows;
     dsyevr_(jobz, range, uplo, nrows, a, lda, vl, vu, il, iu, 
     abstol, nev, eval, evec, lde, isuppz, work, lwork, 
     iwork, liwork, info);
     */

    /*
     *lwork = 3*nrows;
     dsyev_(jobz, uplo, nrows, a, lda, eval, work, lwork, info);
     memcpy(evec, a, nrows*nrows*sizeof(GCGE_DOUBLE));
     */

    /*
     *lwork = 1+6*nrows+2*nrows*nrows;
     dsyevd_(jobz, uplo, nrows, a, lda, eval, work, iwork, info);
     memcpy(evec, a, nrows*nrows*sizeof(GCGE_DOUBLE));
     */
}

//两个稠密矩阵相乘，这里调用BLAS的Subroutine
void GCGE_Default_DenseMatDotDenseMat(char *transa, char *transb, 
        GCGE_INT    *nrows, GCGE_INT    *ncols, GCGE_INT    *mid, 
        GCGE_DOUBLE *alpha, GCGE_DOUBLE *a,     GCGE_INT    *lda, 
        GCGE_DOUBLE *b,     GCGE_INT    *ldb,   GCGE_DOUBLE *beta, 
        GCGE_DOUBLE *c,     GCGE_INT    *ldc)
{
    dgemm_(transa, transb, nrows, ncols, mid, 
            alpha, a, lda, b, ldb, beta, c, ldc);
}

//两个对称矩阵的乘积，调用BLAS的Subroutine
void GCGE_Default_DenseSymMatDotDenseMat(char *side, char *uplo, 
        GCGE_INT    *nrows, GCGE_INT    *ncols,
        GCGE_DOUBLE *alpha, GCGE_DOUBLE *a,   GCGE_INT    *lda,
        GCGE_DOUBLE *b,     GCGE_INT    *ldb, GCGE_DOUBLE *beta,
        GCGE_DOUBLE *c,     GCGE_INT    *ldc)
{
    dsymm_(side, uplo, nrows, ncols, alpha, a, lda,
            b, ldb, beta, c, ldc);
}

//计算向量x和y的内积
GCGE_DOUBLE GCGE_Default_ArrayDotArray(GCGE_DOUBLE *x, GCGE_DOUBLE *y, GCGE_INT length)
{
#if 0
    GCGE_INT i;
    GCGE_DOUBLE value = 0.0;
    for( i=0; i<length; i++ )
        value += a[i]*b[i];
    return value;
#else
    GCGE_INT inc = 1;
    return ddot_(&length, x, &inc, y, &inc);
#endif
}

//计算向量a的范数
GCGE_DOUBLE GCGE_Default_ArrayNorm(GCGE_DOUBLE *x, GCGE_INT length)
{
#if 0
    GCGE_INT i;
    GCGE_DOUBLE value = 0.0;
    for( i=0; i<length; i++ )
        value += a[i]*a[i];
    return sqrt(value);
#else
    GCGE_INT inc = 1;
    return dnrm2_(&length, x, &inc);
#endif
}

//计算y=a*x+b*y
void GCGE_Default_ArrayAXPBY(GCGE_DOUBLE a, GCGE_DOUBLE *x, GCGE_DOUBLE b, GCGE_DOUBLE *y, GCGE_INT length)
{
#if 0
    GCGE_INT i;
    for( i=0; i<length; i++ )
        y[i] = a*x[i] + b*y[i];
#else
    GCGE_INT inc = 1;
    if(b != 1.0)
    {
        dscal_(&length, &b, y, &inc);
    }
    daxpy_(&length, &a, x, &inc, y, &inc);
#endif
}

/* TODO y = x, should shift parameters */
void GCGE_Default_ArrayCopy(GCGE_DOUBLE *x, GCGE_DOUBLE *y, GCGE_INT length)
{
#if 0
    memcpy(y, x, length*sizeof(GCGE_DOUBLE));
#else
    GCGE_INT inc = 1;
    dcopy_(&length, x, &inc, y, &inc);
#endif
}

//x=a*x
void GCGE_Default_ArrayScale(GCGE_DOUBLE a, GCGE_DOUBLE *x, GCGE_INT length)
{
#if 0
    GCGE_INT i;
    for( i=0; i<length; i++ )
        a[i] *= alpha;
#else
    GCGE_INT inc = 1;
    dscal_(&length, &a, x, &inc);
#endif
}


//创建OPS的一个向量，进行初始的设置
void GCGE_OPS_Create(GCGE_OPS **ops)
{
    *ops = (GCGE_OPS*)malloc(sizeof(GCGE_OPS));
    (*ops)->VecSetRandomValue = NULL;
    (*ops)->MatDotVec = NULL;
    (*ops)->MatTransposeDotVec = NULL;
    (*ops)->VecAxpby = NULL;
    (*ops)->VecInnerProd = NULL;
    (*ops)->VecLocalInnerProd = NULL;
    (*ops)->VecCreateByVec = NULL;
    (*ops)->VecDestroy = NULL;
    (*ops)->MultiVecCreateByVec = NULL;
    (*ops)->MultiVecCreateByMat = NULL;
    (*ops)->MultiVecCreateByMultiVec = NULL;
    (*ops)->MultiVecDestroy = NULL;
    (*ops)->MultiVecSetRandomValue = NULL;
    (*ops)->MatDotMultiVec = NULL;
    (*ops)->MatTransposeDotMultiVec = NULL;
    (*ops)->MultiVecAxpby = NULL;
    (*ops)->MultiVecAxpbyColumn = NULL;
    (*ops)->MultiVecLinearComb = NULL;
    (*ops)->MultiVecInnerProd = NULL;
    (*ops)->MultiVecInnerProdLocal = NULL;
    (*ops)->MultiVecSwap = NULL;
    (*ops)->GetVecFromMultiVec = NULL;
    (*ops)->RestoreVecForMultiVec = NULL;
    (*ops)->DenseMatEigenSolver = NULL;
    (*ops)->DenseMatDotDenseMat = NULL;
    (*ops)->DenseSymMatDotDenseMat = NULL;
    (*ops)->Orthonormalization = NULL;
    (*ops)->DenseMatCreate  = NULL;
    (*ops)->DenseMatDestroy = NULL;
    (*ops)->ArrayDotArray = NULL;
    (*ops)->ArrayNorm  = NULL;
    (*ops)->ArrayAXPBY = NULL;
    (*ops)->ArrayCopy  = NULL;
    (*ops)->ArrayScale = NULL;
    (*ops)->LinearSolver = NULL;
}
//OPS的setup: 主要是进行对OPS的赋值， 同时也判断赋值是否充足和合理了
GCGE_INT GCGE_OPS_Setup(GCGE_OPS *ops)
{
    GCGE_INT error = 0;
    if(ops->MatDotVec == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the MatDotVec operation!\n");
        return error;
    }
    if(ops->VecAxpby == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecAxpby operation!\n");
        return error;
    }
    if(ops->VecInnerProd == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecInnerProd operation!\n");
        return error;
    }
    if(ops->VecLocalInnerProd == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecLocalInnerProd operation!\n");
        return error;
    }
    if(ops->VecCreateByVec == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecCreateByVec operation!\n");
        return error;
    }
    if(ops->VecCreateByMat == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecCreateByMat operation!\n");
        return error;
    }
    if(ops->VecDestroy == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecDestroy operation!\n");
        return error;
    }
    if(ops->VecSetRandomValue == NULL)
    {
        error = 1;
        printf("ERROR: Please provide the VecSetRandomValue operation!\n");
        return error;
    }
    if(ops->GetVecFromMultiVec == NULL)
    {
        ops->GetVecFromMultiVec = GCGE_Default_GetVecFromMultiVec;
    }
    if(ops->RestoreVecForMultiVec == NULL)
    {
        ops->RestoreVecForMultiVec = GCGE_Default_RestoreVecForMultiVec;
    }
    if(ops->MultiVecCreateByVec == NULL)
    {
        ops->MultiVecCreateByVec = GCGE_Default_MultiVecCreateByVec;
    }
    if(ops->MultiVecCreateByMat == NULL)
    {
        ops->MultiVecCreateByMat = GCGE_Default_MultiVecCreateByMat;
    }
    if(ops->MultiVecCreateByMultiVec == NULL)
    {
        ops->MultiVecCreateByMultiVec = GCGE_Default_MultiVecCreateByMultiVec;
    }
    if(ops->MultiVecDestroy == NULL)
    {
        ops->MultiVecDestroy = GCGE_Default_MultiVecDestroy; 
    }
    if(ops->MultiVecSetRandomValue == NULL)
    {
        ops->MultiVecSetRandomValue = GCGE_Default_MultiVecSetRandomValue;
    }
    if(ops->MatDotMultiVec == NULL)
    {
        ops->MatDotMultiVec = GCGE_Default_MatDotMultiVec;
    }
    if(ops->MatTransposeDotMultiVec == NULL)
    {
        ops->MatTransposeDotMultiVec = GCGE_Default_MatTransposeDotMultiVec;
    }
    if(ops->MultiVecAxpby == NULL)
    {
        ops->MultiVecAxpby = GCGE_Default_MultiVecAxpby;
    }
    if(ops->MultiVecAxpbyColumn == NULL)
    {
        ops->MultiVecAxpbyColumn = GCGE_Default_MultiVecAxpbyColumn;
    }
    if(ops->MultiVecLinearComb == NULL)
    {
        ops->MultiVecLinearComb = GCGE_Default_MultiVecLinearComb;
    }
    if(ops->MultiVecInnerProd == NULL)
    {
        ops->MultiVecInnerProd = GCGE_Default_MultiVecInnerProd;
    }
    if(ops->MultiVecInnerProdLocal == NULL)
    {
        ops->MultiVecInnerProdLocal = GCGE_Default_MultiVecInnerProdLocal;
    }
    if(ops->MultiVecSwap == NULL)
    {
        ops->MultiVecSwap = GCGE_Default_MultiVecSwap;
    }
    if(ops->DenseMatEigenSolver == NULL)
    {
        ops->DenseMatEigenSolver = GCGE_Default_DenseMatEigenSolver;
    }
    if(ops->DenseMatDotDenseMat == NULL)
    {
        ops->DenseMatDotDenseMat = GCGE_Default_DenseMatDotDenseMat;
    }
    if(ops->DenseSymMatDotDenseMat == NULL)
    {
        ops->DenseSymMatDotDenseMat = GCGE_Default_DenseSymMatDotDenseMat;
    }
    if(ops->ArrayDotArray == NULL)
    {
        ops->ArrayDotArray = GCGE_Default_ArrayDotArray;
    }
    if(ops->ArrayNorm == NULL)
    {
        ops->ArrayNorm = GCGE_Default_ArrayNorm;
    }
    if(ops->ArrayAXPBY == NULL)
    {
        ops->ArrayAXPBY = GCGE_Default_ArrayAXPBY;
    }
    if(ops->ArrayCopy == NULL)
    {
        ops->ArrayCopy = GCGE_Default_ArrayCopy;
    }
    if(ops->ArrayScale == NULL)
    {
        ops->ArrayScale = GCGE_Default_ArrayScale;
    }
}                                           

void GCGE_OPS_Free(GCGE_OPS **ops)
{
    free(*ops); *ops = NULL;
}

void GCGE_OPS_SetLinearSolverWorkspace(GCGE_OPS *ops, void *linear_solver_workspace)
{
    ops->linear_solver_workspace = linear_solver_workspace;
}
