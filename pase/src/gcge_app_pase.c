/*
 * =====================================================================================
 *
 *       Filename:  gcge_app_pase.c
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
#include "gcge_app_pase.h"

void GCGE_PASE_VecCreateByVec(void **des_vec, void *src_vec, struct GCGE_OPS_ *ops)
{
    PASE_DefaultVecCreateByVec(des_vec, src_vec, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_VecDestroy(void **vec, struct GCGE_OPS_ *ops)
{
    PASE_DefaultVecDestroy(vec, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_VecCreateByMat(void **vec, void *mat, struct GCGE_OPS_ *ops)
{
    PASE_DefaultVecCreateByMat(vec, mat, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_VecSetRandomValue(void *vec, struct GCGE_OPS_ *ops)
{
    PASE_INT seed = 1;
    PASE_DefaultVecSetRandomValue(vec, seed, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MatDotVec(void *Matrix, void *x, void *r, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMatDotVec(Matrix, x, r, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_VecAxpby(GCGE_DOUBLE a, void *x, GCGE_DOUBLE b, void *y, struct GCGE_OPS_ *ops)
{
    PASE_DefaultVecAxpby(a, x, b, y, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_VecInnerProd(void *x, void *y, GCGE_DOUBLE *value_ip, struct GCGE_OPS_ *ops)
{
    PASE_DefaultVecInnerProd(x, y, value_ip, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_VecLocalInnerProd(void *x, void *y, GCGE_DOUBLE *value_ip, struct GCGE_OPS_ *ops)
{
    PASE_DefaultVecLocalInnerProd(x, y, value_ip, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecDestroy(void ***MultiVec, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecDestroy(MultiVec, n_vec, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecCreateByMat(void ***multi_vec, 
        GCGE_INT n_vec, void *mat, struct GCGE_OPS_ *ops)
{
    //((PASE_OPS*)(ops->ops))->MultiVecCreateByMat(multi_vec, n_vec, mat, (PASE_OPS*)(ops->ops));
    PASE_DefaultMultiVecCreateByMat(multi_vec, n_vec, mat, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecSetRandomValue(void **multi_vec, GCGE_INT start, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecSetRandomValue(multi_vec, start, n_vec, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_GetVecFromMultiVec(void **V, GCGE_INT j, void **x, struct GCGE_OPS_ *ops)
{
    PASE_DefaultGetVecFromMultiVec(V, j, x, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_RestoreVecForMultiVec(void **V, GCGE_INT j, void **x, struct GCGE_OPS_ *ops)
{
    PASE_DefaultRestoreVecForMultiVec(V, j, x, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MatDotMultiVec(void *mat, void **x, void **y, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMatDotMultiVec(mat, x, y, start, end, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecSwap(void **V_1, void **V_2, GCGE_INT *start, 
        GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecSwap(V_1, V_2, start, end, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecPrint(void **x, GCGE_INT n, 
        struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecPrint(x, n, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecInnerProd(void **V, void **W, GCGE_DOUBLE *a, char *is_sym, 
        GCGE_INT *start, GCGE_INT *end, GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecInnerProd(V, W, a, is_sym, start, end, lda, if_Vec, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecLinearComb(void **x, void **y, GCGE_INT *start,
        GCGE_INT *end, GCGE_DOUBLE *a, GCGE_INT lda, GCGE_INT if_Vec,
        GCGE_DOUBLE alpha, GCGE_DOUBLE beta, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecLinearComb(x, y, start, end, a, lda, if_Vec, 
          alpha, beta, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_MultiVecAxpby(GCGE_DOUBLE a, void **x, GCGE_DOUBLE b, void **y, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    PASE_DefaultMultiVecAxpby(a, x, b, y, start, end, (PASE_OPS*)(ops->ops));
}

void GCGE_PASE_SetOps(GCGE_OPS *ops, PASE_OPS *pase_ops)
{
    ops->ops                    = (void*)pase_ops;

    ops->VecCreateByVec         = GCGE_PASE_VecCreateByVec;
    ops->VecCreateByMat         = GCGE_PASE_VecCreateByMat;
    ops->VecDestroy             = GCGE_PASE_VecDestroy;

    ops->VecSetRandomValue      = GCGE_PASE_VecSetRandomValue;
    ops->MatDotVec              = GCGE_PASE_MatDotVec;
    ops->VecAxpby               = GCGE_PASE_VecAxpby;
    ops->VecInnerProd           = GCGE_PASE_VecInnerProd;
    ops->VecLocalInnerProd      = GCGE_PASE_VecLocalInnerProd;

    ops->MultiVecDestroy        = GCGE_PASE_MultiVecDestroy;
    ops->MultiVecCreateByMat    = GCGE_PASE_MultiVecCreateByMat;
    ops->MultiVecSetRandomValue = GCGE_PASE_MultiVecSetRandomValue;

    ops->GetVecFromMultiVec     = GCGE_PASE_GetVecFromMultiVec;
    ops->RestoreVecForMultiVec  = GCGE_PASE_RestoreVecForMultiVec;
    ops->MatDotMultiVec         = GCGE_PASE_MatDotMultiVec;

    ops->MultiVecLinearComb     = GCGE_PASE_MultiVecLinearComb;
    ops->MultiVecInnerProd      = GCGE_PASE_MultiVecInnerProd;
    ops->MultiVecSwap           = GCGE_PASE_MultiVecSwap;
    ops->MultiVecAxpby          = GCGE_PASE_MultiVecAxpby;
    ops->MultiVecPrint          = GCGE_PASE_MultiVecPrint;

}

//给一个pase_ops，把pase_ops转化成gcge_ops
void GCGE_SOLVER_SetPASEOps(GCGE_SOLVER *solver, PASE_OPS *pase_ops)
{
    GCGE_PASE_SetOps(solver->ops, pase_ops);
}

//下面是一个对PASE 的GCG_Solver的初始化
//用户只提供要求解特征值问题的矩阵A,B,线性解法器等都用默认值
//如果用户想在命令行提供特征值个数，需要将num_eigenvalues赋值为-1
GCGE_SOLVER* GCGE_PASE_Solver_Init(PASE_Matrix A, PASE_Matrix B, int num_eigenvalues, int argc, char* argv[], PASE_OPS *pase_ops)
{
    //第一步: 定义相应的ops,para以及workspace

    //创建一个solver变量
    GCGE_SOLVER *pase_solver;
    GCGE_SOLVER_Create(&pase_solver);
    if(num_eigenvalues != -1)
        pase_solver->para->nev = num_eigenvalues;
    GCGE_INT error = GCGE_PARA_SetFromCommandLine(pase_solver->para, argc, argv);
    GCGE_SOLVER_SetPASEOps(pase_solver, pase_ops);

    //设置初始值
    int nev = pase_solver->para->nev;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    pase_solver->eval = eval;

    PASE_MultiVector evec;
    pase_solver->ops->MultiVecCreateByMat((void***)(&evec), nev, (void*)A, pase_solver->ops);

    GCGE_SOLVER_SetMatA(pase_solver, A);
    if(B != NULL)
        GCGE_SOLVER_SetMatB(pase_solver, B);
    GCGE_SOLVER_SetEigenvalues(pase_solver, eval);
    GCGE_SOLVER_SetEigenvectors(pase_solver, (void**)evec);
    //setup and solve
    GCGE_SOLVER_Setup(pase_solver);
    //GCGE_PrintParaInfo(pase_solver->para);
    //GCGE_Printf("Set up finish!\n");
    return pase_solver;
}

GCGE_SOLVER* GCGE_SOLVER_PASE_Create(PASE_Matrix A, PASE_Matrix B, int num_eigenvalues, PASE_REAL *eval, PASE_MultiVector evec, PASE_OPS *pase_ops, PASE_INT num_unlock)
{
    GCGE_SOLVER *pase_solver;
    GCGE_SOLVER_Create(&pase_solver);
    GCGE_SOLVER_SetPASEOps(pase_solver, pase_ops);

    GCGE_SOLVER_SetMatA(pase_solver, (void*)A);
    if(B != NULL)
        GCGE_SOLVER_SetMatB(pase_solver, (void*)B);
    GCGE_SOLVER_SetEigenvalues(pase_solver, eval);
    GCGE_SOLVER_SetEigenvectors(pase_solver, (void**)evec);
    pase_solver->para->num_unlock = num_unlock;

    if(num_eigenvalues != -1)
    {
        pase_solver->para->nev = num_eigenvalues;
    }
    pase_solver->para->block_size = pase_solver->para->nev;

    GCGE_SOLVER_Setup(pase_solver);
    return pase_solver;
}
