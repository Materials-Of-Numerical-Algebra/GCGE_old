/*
 * =====================================================================================
 *
 *       Filename:  gcge_para.h
 *
 *    Description:  
 *        基于GCGE的ops的单向量生成pase的单向量
 *        基于GCGE的ops的多向量生成pase的多向量
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

#ifndef  _PASE_OPS_H_
#define  _PASE_OPS_H_

#include "gcge_ops.h"
#include "pase_config.h"
#include "pase_matvec.h"
//把每一个操作都写好，这样以后进行算法设计的时候才能方便。
#if GCGE_USE_MPI
#include <mpi.h>
#endif


typedef struct PASE_OPS_ {

    GCGE_OPS *gcge_ops;

    void (*VecSetRandomValue)       (void *vec, PASE_INT seed, struct PASE_OPS_ *ops);
    void (*MatDotVec)               (void *Matrix, void *x, void *r, struct PASE_OPS_ *ops);
    void (*VecAxpby)                (PASE_REAL a, void *x, PASE_REAL b, void *y, struct PASE_OPS_ *ops); /* y = ax+by */
    void (*VecInnerProd)            (void *x, void *y, PASE_REAL *value_ip, struct PASE_OPS_ *ops);
    void (*VecLocalInnerProd)       (void *x, void *y, PASE_REAL *value_ip, struct PASE_OPS_ *ops);
    void (*VecCreateByVec)          (void **des_vec, void *src_vec, struct PASE_OPS_ *ops);
    void (*VecCreateByMat)          (void **vec, void *mat, struct PASE_OPS_ *ops);
    void (*VecDestroy)              (void **vec, struct PASE_OPS_ *ops);

    void (*MultiVecCreateByVec)     (void ***multi_vec, PASE_INT n_vec, void *vec, 
                                     struct PASE_OPS_ *ops);
    void (*MultiVecCreateByMat)     (void ***multi_vec, PASE_INT n_vec, void *mat, 
                                     struct PASE_OPS_ *ops);
    void (*MultiVecCreateByMultiVec)(void **init_vec, void ***multi_vec, PASE_INT n_vec, 
                                     struct PASE_OPS_ *ops);
    void (*MultiVecDestroy)         (void ***MultiVec, PASE_INT n_vec, struct PASE_OPS_ *ops);

    /* TODO */
    void (*MultiVecSetRandomValue)  (void **multi_vec, PASE_INT start, PASE_INT n_vec, struct PASE_OPS_ *ops);
    void (*MatDotMultiVec)          (void *mat, void **x, void **y, PASE_INT *start, PASE_INT *end, 
                                     struct PASE_OPS_ *ops);
    void (*MultiVecAxpby)           (PASE_REAL a, void **x, PASE_REAL b, void **y, 
                                     PASE_INT *start, PASE_INT *end, struct PASE_OPS_ *ops);
    void (*MultiVecAxpbyColumn)     (PASE_REAL a, void **x, PASE_INT col_x, PASE_REAL b, 
                                     void **y, PASE_INT col_y, struct PASE_OPS_ *ops);
    /* vec_y[j] = \sum_{i=sx}^{ex} vec_x[i] a[i-sx][j-sy] */
    void (*MultiVecLinearComb)      (void **x, void **y, PASE_INT *start, PASE_INT *end,
                                     PASE_REAL *a, PASE_INT lda, PASE_INT if_Vec, 
                                     PASE_REAL alpha, PASE_REAL beta, struct PASE_OPS_ *ops);
    void (*MultiVecInnerProd)       (void **V, void **W, PASE_REAL *a, 
                                     char *is_sym, PASE_INT *start, PASE_INT *end, PASE_INT lda, 
                                     PASE_INT if_Vec, struct PASE_OPS_ *ops);
    void (*MultiVecSwap)            (void **V_1, void **V_2, PASE_INT *start, PASE_INT *end, 
                                     struct PASE_OPS_ *ops);

    /* TODO kernal function should use this op to get j-th vector */
    void (*GetVecFromMultiVec)      (void **V, PASE_INT j, void **x, struct PASE_OPS_ *ops);
    void (*RestoreVecForMultiVec)   (void **V, PASE_INT j, void **x, struct PASE_OPS_ *ops);
    void (*MultiVecPrint)           (void **x, PASE_INT n, struct PASE_OPS_ *ops);
   
}PASE_OPS;

//创建PASE_OPS
void PASE_OPS_Create(PASE_OPS **ops, GCGE_OPS *gcge_ops);
PASE_INT PASE_OPS_Setup(PASE_OPS *ops);
void PASE_OPS_Free(PASE_OPS **ops);
void PASE_MatrixCreate( PASE_Matrix* pase_matrix,
                        PASE_INT num_aux_vec,
                        void *A_H, 
                        PASE_OPS *ops
                        );
void PASE_MatrixDestroy( PASE_Matrix*  matrix, 
                         PASE_OPS *ops );


#endif
