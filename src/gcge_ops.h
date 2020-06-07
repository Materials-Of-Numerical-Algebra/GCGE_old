/*
 * =====================================================================================
 *
 *       Filename:  gcge_para.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时50分16秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef  _GCGE_OPS_H_
#define  _GCGE_OPS_H_

#include <math.h>
#include "gcge_config.h"
//把每一个操作都写好，这样以后进行算法设计的时候才能方便。
#if GCGE_USE_MPI
#include <mpi.h>
#endif


typedef struct GCGE_OPS_ {

    //添加一个空型的ops
    void *ops;

    void (*VecSetRandomValue)       (void *vec, struct GCGE_OPS_ *ops);
    /* r = mat  * x */
    void (*MatDotVec)               (void *mat, void *x, void *r, struct GCGE_OPS_ *ops);
    /* r = mat' * x */
    void (*MatTransposeDotVec)      (void *mat, void *x, void *r, struct GCGE_OPS_ *ops);
    /* y = ax+by */
    void (*VecAxpby)                (GCGE_DOUBLE a, void *x, GCGE_DOUBLE b, void *y, struct GCGE_OPS_ *ops);
    /* value_ip = x'y */
    void (*VecInnerProd)            (void *x, void *y, GCGE_DOUBLE *value_ip, struct GCGE_OPS_ *ops);
    /* value_ip = x'y for each proc */
    void (*VecLocalInnerProd)       (void *x, void *y, GCGE_DOUBLE *value_ip, struct GCGE_OPS_ *ops);
    void (*VecCreateByVec)          (void **des_vec, void *src_vec, struct GCGE_OPS_ *ops);
    void (*VecCreateByMat)          (void **vec, void *mat, struct GCGE_OPS_ *ops);
    void (*VecDestroy)              (void **vec, struct GCGE_OPS_ *ops);

    /* kernal function should use this op to get j-th vector */
    /* x = V[j] */
    void (*GetVecFromMultiVec)      (void **V, GCGE_INT j, void **x, struct GCGE_OPS_ *ops);
    void (*RestoreVecForMultiVec)   (void **V, GCGE_INT j, void **x, struct GCGE_OPS_ *ops);

    /* get multigrid operator for num_levels = 4
     * P0     P1       P2
     * A0     A1       A2        A3
     * B0  P0'B0P0  P1'B1P1   P2'B2P2 
     * A0 is the original matrix */
    void (*MultiGridCreate)         (void ***A_array, void ***B_array, void ***P_array, GCGE_INT *num_levels, void *A, void *B, struct GCGE_OPS_ *ops);
    /* free A1 A2 A3 B1 B2 B3 P0 P1 P2 
     * A0 and B0 are just pointers */
    void (*MultiGridDestroy)        (void ***A_array, void ***B_array, void ***P_array, GCGE_INT *num_levels, struct GCGE_OPS_ *ops);
    /* for level_i = 0 < level_j = 3 
     * v_tmp = P0' vec_i 
     * v_tmp = P1' v_tmp 
     * vec_j = P2' v_tmp */
    /* for level_i = 2 > level_j = 0 
     * v_tmp = P1 vec_i 
     * vec_j = P0 v_tmp */
    void (*VecFromItoJ)             (void **P_array, GCGE_INT level_i, GCGE_INT level_j, void *vec_i, void *vec_j, void **vec_tmp, struct GCGE_OPS_ *ops);
    void (*MultiVecFromItoJ)        (void **P_array, GCGE_INT level_i, GCGE_INT level_j, void **multi_vec_i, void **multi_vec_j, void ***multi_vec_tmp, GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops);

    /* option */
    /* TODO */
    void (*LinearSolver)            (void *Matrix, void *b, void *x, struct GCGE_OPS_ *ops);
    void *linear_solver_workspace;
    void (*MultiLinearSolver)       (void *Matrix, void **b, void **x, int *start, int *end, struct GCGE_OPS_ *ops); 
    void *multi_linear_solver_workspace;

    /* TODO DenseMatCreate, DenseMatDestroy should in function Orthonormalization 
     * Add struct member name void *orth_workspace to save tmp variables */
    void (*Orthonormalization)      (void **V, GCGE_INT start, GCGE_INT *end, void *B, struct GCGE_OPS_ *ops);
    void *orth_workspace;
    void (*DenseMatCreate)          (void **densemat, GCGE_INT nrows, GCGE_INT ncols);
    void (*DenseMatDestroy)         (void **mat);
 

    /* DEEP */

    void (*MultiVecCreateByVec)      (void ***multi_vec, GCGE_INT n_vec, void *vec, struct GCGE_OPS_ *ops);
    void (*MultiVecCreateByMat)      (void ***multi_vec, GCGE_INT n_vec, void *mat, struct GCGE_OPS_ *ops);
    void (*MultiVecCreateByMultiVec) (void ***multi_vec, GCGE_INT n_vec, void **init_vec, struct GCGE_OPS_ *ops);
    void (*MultiVecDestroy)          (void ***multi_vec, GCGE_INT n_vec, struct GCGE_OPS_ *ops);

    /* TODO 这里start n_vec需要修改成 start end */
    void (*MultiVecSetRandomValue)  (void **multi_vec, GCGE_INT start, GCGE_INT n_vec, struct GCGE_OPS_ *ops);
    void (*MatDotMultiVec)          (void *mat, void **x, void **y, GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops);
    void (*MatTransposeDotMultiVec) (void *mat, void **x, void **y, GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops);
    void (*MultiVecAxpby)           (GCGE_DOUBLE a, void **x, GCGE_DOUBLE b, void **y, GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops);
    /* y[col_y] = a*x[col_x] + b*y[col_y] */
    void (*MultiVecAxpbyColumn)     (GCGE_DOUBLE a, void **x, GCGE_INT col_x, GCGE_DOUBLE b, void **y, GCGE_INT col_y, struct GCGE_OPS_ *ops);
    /* vec_y[j] = \sum_{i=sx}^{ex} vec_x[i] a[i-sx][j-sy] */
    //MultiVecLinearComb去掉原来的void* dmat与GCGE_INT lddmat两个参数，加上GCGE_INT if_Vec这个参数
    //加的参数if_Vec表示是否是线性组合得到一个向量，因为之前的线性组合是默认得到一个向量组，但我们应该允许得到一个(单向量结构的)向量
    //对gcge_ops.c中默认的线性组合函数，如果y给的是单向量，因为给的是void**类型，可以直接取y[0], 因此不需要改函数体内部
    //另加上alpha与beta两个参数，表示y=alpha*a*x+beta*y, Default默认只支持alpha=1.0,beta=1.0或beta=0.0的情况
    void (*MultiVecLinearComb)      (void **x, void **y, GCGE_INT *start, GCGE_INT *end,
                                     GCGE_DOUBLE *a, GCGE_INT lda, GCGE_INT if_Vec,
                                     GCGE_DOUBLE alpha, GCGE_DOUBLE beta, struct GCGE_OPS_ *ops);
    //添加了
    //加的参数if_Vec表示是否是多向量组与一个单向量结构进行线性组合
    //这两个函数里加这个参数主要是为了slepc接口的时候不出问题
    void (*MultiVecInnerProd)       (void **V, void **W, GCGE_DOUBLE *a, char *is_sym, 
                                     GCGE_INT *start, GCGE_INT *end, GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops);
    void (*MultiVecInnerProdLocal)  (void **V, void **W, GCGE_DOUBLE *a, char *is_sym, 
                                     GCGE_INT *start, GCGE_INT *end, GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops);
    void (*MultiVecSwap)            (void **V_1, void **V_2, GCGE_INT *start, GCGE_INT *end, 
                                     struct GCGE_OPS_ *ops);
    void (*MultiVecPrint)           (void **x, GCGE_INT n, struct GCGE_OPS_ *ops);

    /* usefull ?? */
    void (*SetDirichletBoundary)    (void**Vecs, GCGE_INT nev, void* A, void* B);


    /* DEEP option */
    //a为要计算特征值的矩阵，lda为a的leading dimension, nrows为要计算前nrows行，
    //eval,evec为特征对，从第1个位置开始存,lde为evec的leading dimension
    //il, iu为要求a的第il到第iu个特征值
    //iwork为要提供的int型工作空间,dwork为要提供的double型工作空间
    void (*DenseMatEigenSolver)     (GCGE_DOUBLE *a, GCGE_INT lda, GCGE_INT nrows, 
                                     GCGE_DOUBLE *eval, GCGE_DOUBLE *evec, GCGE_INT lde, 
                                     GCGE_INT il, GCGE_INT iu,
                                     GCGE_INT *iwork, GCGE_DOUBLE *dwork);
    void (*DenseMatDotDenseMat)     (char *transa, char *transb, GCGE_INT *nrows, GCGE_INT *ncols, 
                                     GCGE_INT *mid, GCGE_DOUBLE *alpha, GCGE_DOUBLE *a,
                                     GCGE_INT *lda, GCGE_DOUBLE *b, GCGE_INT *ldb, 
                                     GCGE_DOUBLE *beta, GCGE_DOUBLE *c, GCGE_INT *ldc);
    void (*DenseSymMatDotDenseMat)  (char *side, char *uplo, GCGE_INT *nrows, GCGE_INT *ncols,
                                     GCGE_DOUBLE *alpha, GCGE_DOUBLE *a, GCGE_INT *lda,
                                     GCGE_DOUBLE *b, GCGE_INT *ldb, GCGE_DOUBLE *beta,
                                     GCGE_DOUBLE *c, GCGE_INT *ldc);

    GCGE_DOUBLE (*ArrayDotArray)    (GCGE_DOUBLE *x, GCGE_DOUBLE *y, GCGE_INT length);
    GCGE_DOUBLE (*ArrayNorm)        (GCGE_DOUBLE *x, GCGE_INT length);
    void (*ArrayAXPBY)              (GCGE_DOUBLE a, GCGE_DOUBLE *x, GCGE_DOUBLE b, GCGE_DOUBLE *y, GCGE_INT length);
    void (*ArrayCopy)               (GCGE_DOUBLE *x, GCGE_DOUBLE *y, GCGE_INT length);
    void (*ArrayScale)              (GCGE_DOUBLE alpha, GCGE_DOUBLE *a, GCGE_INT length);

   
}GCGE_OPS;

extern void dsyev_(char *jobz, char *uplo, 
        GCGE_INT    *nrows,  GCGE_DOUBLE *a,    GCGE_INT *lda, 
        GCGE_DOUBLE *eval,
        GCGE_DOUBLE *work,   GCGE_INT *lwork, 
        GCGE_INT *info);

extern void dsyevx_(char *jobz, char *range, char *uplo, 
        GCGE_INT    *nrows,  GCGE_DOUBLE *a,    GCGE_INT *lda, 
        GCGE_DOUBLE *vl,     GCGE_DOUBLE *vu,   GCGE_INT *il,  GCGE_INT *iu, 
        GCGE_DOUBLE *abstol, GCGE_INT *nev, 
        GCGE_DOUBLE *eval,   GCGE_DOUBLE *evec, GCGE_INT *lde, 
        GCGE_DOUBLE *work,   GCGE_INT *lwork, 
        GCGE_INT    *iwork,  GCGE_INT *liwork,  GCGE_INT *info);

extern void dgemm_(char *transa, char *transb, 
        GCGE_INT    *nrows, GCGE_INT    *ncols, GCGE_INT    *mid, 
        GCGE_DOUBLE *alpha, GCGE_DOUBLE *a,     GCGE_INT    *lda, 
        GCGE_DOUBLE *b,     GCGE_INT    *ldb,   GCGE_DOUBLE *beta, 
        GCGE_DOUBLE *c,     GCGE_INT    *ldc);

extern void dsymm_(char *side, char *uplo, 
        GCGE_INT    *nrows, GCGE_INT    *ncols,
        GCGE_DOUBLE *alpha, GCGE_DOUBLE *a,   GCGE_INT    *lda,
        GCGE_DOUBLE *b,     GCGE_INT    *ldb, GCGE_DOUBLE *beta,
        GCGE_DOUBLE *c,     GCGE_INT    *ldc);

extern void   daxpy_(int *size, double *alpha, double *x, int *incx, double *y, int *incy);
extern double ddot_(int *length, double *x, int *incx, double *y, int *incy);
extern double dnrm2_(int *length, double *x, int *inc);
extern void   dscal_(int *length, double *a, double *x, int *inc);
extern void   dcopy_(int *length, double *x, int *incx, double *y, int *incy);

void GCGE_OPS_Create(GCGE_OPS **ops);
GCGE_INT GCGE_OPS_Setup(GCGE_OPS *ops);
void GCGE_OPS_Free(GCGE_OPS **ops);

void GCGE_OPS_SetLinearSolverWorkspace(GCGE_OPS *ops, void *linear_solver_workspace);

#endif
