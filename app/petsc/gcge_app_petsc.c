/*
 * =====================================================================================
 *
 *       Filename:  gcge_app_petsc.c
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

#include "gcge_app_petsc.h"

void PETSC_ReadMatrixBinary(Mat *A, const char *filename)
{
    PetscViewer    viewer;
    PetscPrintf(PETSC_COMM_WORLD, "Getting matrix...\n"); 
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer); 
    MatCreate(PETSC_COMM_WORLD, A); 
    MatSetFromOptions(*A); 
    MatLoad(*A, viewer); 
    PetscViewerDestroy(&viewer);
    PetscPrintf(PETSC_COMM_WORLD, "Getting matrix... Done\n");
}

void PETSC_LinearSolverCreate(KSP *ksp, Mat A, Mat T)
{
    KSPCreate(PETSC_COMM_WORLD,ksp);
    KSPSetOperators(*ksp,A,T);
    KSPSetInitialGuessNonzero(*ksp, 1);

    KSPSetType(*ksp, KSPCG);
    //这里的rtol应取作<=ev_tol
    //  KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits)
    KSPSetTolerances(*ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000);
    PC pc;
    KSPGetPC(*ksp, &pc);
    //PCSetType(pc, PCHYPRE);
    //PCHYPRESetType(pc, "boomeramg");
    PCSetType(pc, PCHMG);
    PCHMGSetInnerPCType(pc, PCGAMG);
    PCHMGSetReuseInterpolation(pc, PETSC_TRUE);
    PCHMGSetUseSubspaceCoarsening(pc, PETSC_TRUE);
    PCHMGUseMatMAIJ(pc, PETSC_FALSE);
    PCHMGSetCoarseningComponent(pc, 0);
    //最后从命令行设置参数
    KSPSetFromOptions(*ksp);
}

void PETSC_VecLocalInnerProd(Vec x, Vec y, double *value)
{
    const PetscScalar *local_x;
    const PetscScalar *local_y;
    PetscInt           low, high, length, i = 0;
    VecGetOwnershipRange(x, &low, &high);
    length = high-low;
    VecGetArrayRead(x,&local_x);
    VecGetArrayRead(y,&local_y);
    *value = 0.0;
    for(i=0; i<length; i++)
    {
        *value += local_x[i]*local_y[i];
    }
    VecRestoreArrayRead(x,&local_x);
    VecRestoreArrayRead(y,&local_y);
}


void GCGE_PETSC_VecCreateByVec(void **d_vec, void *s_vec, GCGE_OPS *ops)
{
	
    VecDuplicate((Vec)s_vec, (Vec*)d_vec);
}
void GCGE_PETSC_VecCreateByMat(void **vec, void *mat, GCGE_OPS *ops)
{
	
    MatCreateVecs((Mat)mat, NULL, (Vec*)vec);
}
void GCGE_PETSC_MultiVecCreateByMat(void ***multi_vec, GCGE_INT n_vec, void *mat, GCGE_OPS *ops)
{

   Vec vector;
   MatCreateVecs((Mat)mat, NULL, &vector);
   VecDuplicateVecs(vector, n_vec, (Vec**)multi_vec);
   VecDestroy(&vector);
}
void GCGE_PETSC_VecDestroy(void **vec, GCGE_OPS *ops)
{
	
    VecDestroy((Vec*)vec);
}
void GCGE_PETSC_MultiVecDestroy(void ***MultiVec, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
	
    VecDestroyVecs(n_vec, (Vec**)MultiVec);
}

void GCGE_PETSC_VecSetRandomValue(void *vec, GCGE_OPS *ops)
{
    PetscRandom    rctx;
    
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    PetscRandomSetFromOptions(rctx);
    VecSetRandom((Vec)vec, rctx);
    PetscRandomDestroy(&rctx);
}
void GCGE_PETSC_MultiVecSetRandomValue(void **multi_vec, GCGE_INT start, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    PetscRandom    rctx;
    
    PetscInt       i;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    PetscRandomSetFromOptions(rctx);
    for(i=0; i<n_vec; i++)
    {
        VecSetRandom(((Vec*)multi_vec)[i], rctx);
    }
    PetscRandomDestroy(&rctx);
}

void GCGE_PETSC_MatDotVec(void *mat, void *x, void *r, GCGE_OPS *ops)
{
     MatMult((Mat)mat, (Vec)x, (Vec)r);
}

void GCGE_PETSC_MatTransposeDotVec(void *mat, void *x, void *r, GCGE_OPS *ops)
{
     MatMultTranspose((Mat)mat, (Vec)x, (Vec)r);
}

void GCGE_PETSC_VecAxpby(GCGE_DOUBLE a, void *x, GCGE_DOUBLE b, void *y, GCGE_OPS *ops)
{
    
	VecScale((Vec)y, b);
    if(x != y)
    {
	    VecAXPY((Vec)y, a, (Vec)x);
    }
}
void GCGE_PETSC_VecInnerProd(void *x, void *y, GCGE_DOUBLE *value_ip, GCGE_OPS *ops)
{
	 VecDot((Vec)x, (Vec)y, value_ip);
}

void GCGE_PETSC_VecLocalInnerProd(void *x, void *y, GCGE_DOUBLE *value_ip, GCGE_OPS *ops)
{
    PETSC_VecLocalInnerProd((Vec)x, (Vec)y, value_ip);
}

void GCGE_PETSC_LinearSolver(void *Matrix, void *b, void *x, struct GCGE_OPS_ *ops)
{
    
    KSPSolve((KSP)(ops->linear_solver_workspace), (Vec)b, (Vec)x);
}

void GCGE_PETSC_SetOps(GCGE_OPS *ops)
{
    /* either-or */
    ops->VecCreateByVec     = GCGE_PETSC_VecCreateByVec;
    ops->VecCreateByMat     = GCGE_PETSC_VecCreateByMat;
    ops->VecDestroy           = GCGE_PETSC_VecDestroy;

    ops->VecSetRandomValue  = GCGE_PETSC_VecSetRandomValue;
    ops->MatDotVec          = GCGE_PETSC_MatDotVec;
    ops->MatTransposeDotVec = GCGE_PETSC_MatTransposeDotVec;
    ops->VecAxpby           = GCGE_PETSC_VecAxpby;
    ops->VecInnerProd       = GCGE_PETSC_VecInnerProd;
    ops->VecLocalInnerProd  = GCGE_PETSC_VecLocalInnerProd;

    ops->MultiVecDestroy           = GCGE_PETSC_MultiVecDestroy;
    ops->MultiVecCreateByMat     = GCGE_PETSC_MultiVecCreateByMat;
    ops->MultiVecSetRandomValue = GCGE_PETSC_MultiVecSetRandomValue;
}

void GCGE_SOLVER_SetPETSCOps(GCGE_SOLVER *solver)
{
    GCGE_PETSC_SetOps(solver->ops);
}

//下面是一个对PETSC 的GCG_Solver的初始化
//用户只提供要求解特征值问题的矩阵A,B,线性解法器等都用默认值
//如果用户想在命令行提供特征值个数，需要将num_eigenvalues赋值为-1
GCGE_SOLVER* GCGE_PETSC_Solver_Init(Mat A, Mat B, int num_eigenvalues, int argc, char* argv[])
{
    //第一步: 定义相应的ops,para以及workspace

    //创建一个solver变量
    GCGE_SOLVER *petsc_solver;
    GCGE_SOLVER_Create(&petsc_solver);
    if(num_eigenvalues != -1)
        petsc_solver->para->nev = num_eigenvalues;
    GCGE_PARA_SetFromCommandLine(petsc_solver->para, argc, argv);
    //设置初始值
    int nev = petsc_solver->para->nev;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    petsc_solver->eval = eval;

	
    Vec *evec;
    Vec vector;
    MatCreateVecs(A, NULL, &vector);
    VecDuplicateVecs(vector, nev, &evec);
    VecDestroy(&vector);

    GCGE_SOLVER_SetMatA(petsc_solver, A);
    if(B != NULL)
        GCGE_SOLVER_SetMatB(petsc_solver, B);
    GCGE_SOLVER_SetPETSCOps(petsc_solver);
    GCGE_SOLVER_SetEigenvalues(petsc_solver, eval);
    GCGE_SOLVER_SetEigenvectors(petsc_solver, (void**)evec);
    //setup and solve
    GCGE_SOLVER_Setup(petsc_solver);
    GCGE_PrintParaInfo(petsc_solver->para);
    GCGE_Printf("Set up finish!\n");
    return petsc_solver;
}

//下面是一个对PETSC 的GCG_Solver的初始化
//用户只提供要求解特征值问题的矩阵A,B,线性解法器使用KSPSolve,用户不接触ksp,设置参数均从命令行设置
GCGE_SOLVER* GCGE_PETSC_Solver_Init_KSPDefault(Mat A, Mat B, Mat P, int num_eigenvalues, int argc, char* argv[])
{
    //首先用矩阵A,B创建petsc_solver
    GCGE_SOLVER *petsc_solver = GCGE_PETSC_Solver_Init(A, B, num_eigenvalues, argc, argv);
    KSP ksp;
    //使用矩阵A和预条件矩阵P创建ksp,ksp的参数均为默认参数,要修改ksp参数只能用命令行参数进行设置
    PETSC_LinearSolverCreate(&ksp, (Mat)A, (Mat)P);
    //把ksp作为ops->linear_solver_workspace
    GCGE_SOLVER_SetOpsLinearSolverWorkspace(petsc_solver, (void*)ksp);
    //将线性解法器设为KSPSolve
    petsc_solver->ops->LinearSolver = GCGE_PETSC_LinearSolver;
    return petsc_solver;
}

//下面是一个对PETSC 的GCG_Solver的初始化
//用户提供要求解特征值问题的矩阵A,B,以及线性解法器结构ksp,使用KSPSolve作为线性解法器
GCGE_SOLVER* GCGE_PETSC_Solver_Init_KSPGivenByUser(Mat A, Mat B, KSP ksp, int num_eigenvalues, int argc, char* argv[])
{
    //首先用矩阵A,B创建petsc_solver
    GCGE_SOLVER *petsc_solver = GCGE_PETSC_Solver_Init(A, B, num_eigenvalues, argc, argv);
    //用户已创建好ksp,直接赋给petsc_solver
    GCGE_SOLVER_SetOpsLinearSolverWorkspace(petsc_solver, (void*)ksp);
    //将线性解法器设为KSPSolve
    petsc_solver->ops->LinearSolver = GCGE_PETSC_LinearSolver;
    return petsc_solver;
}

void GCGE_SOLVER_SetPETSCOpsLinearSolver(GCGE_SOLVER *solver, KSP ksp)
{
    GCGE_SOLVER_SetOpsLinearSolverWorkspace(solver, (void*)ksp);
    //将线性解法器设为KSPSolve
    solver->ops->LinearSolver = GCGE_PETSC_LinearSolver;
}

void GCGE_SOLVER_PETSC_Create(GCGE_SOLVER **petsc_solver)
{
    GCGE_SOLVER_Create(petsc_solver);
    GCGE_SOLVER_SetPETSCOps(*petsc_solver);
}

void GCGE_SOLVER_PETSC_Setup(GCGE_SOLVER *petsc_solver, void *A, void *B, void *P)
{
    //给特征值和特征向量分配nev大小的空间
    GCGE_INT nev = petsc_solver->para->nev;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    Vec *evec;
    petsc_solver->ops->MultiVecCreateByMat((void***)(&evec), nev, (void*)A, petsc_solver->ops);

    //将矩阵与特征对设置到petsc_solver中
    GCGE_SOLVER_SetMatA(petsc_solver, A);
    if(B != NULL)
    {
        GCGE_SOLVER_SetMatB(petsc_solver, B);
    }
    GCGE_SOLVER_SetEigenvalues(petsc_solver, eval);
    GCGE_SOLVER_SetEigenvectors(petsc_solver, (void**)evec);

    //如果读入了预条件矩阵, 就使用PETSc的求解器
    if(P != NULL)
    {
        KSP ksp;
        //设定线性求解器
        PETSC_LinearSolverCreate(&ksp, A, P);
        //PetscViewer viewer;
        //KSPView(ksp, viewer);
        //给petsc_solver设置KSP为线性求解器
        GCGE_SOLVER_SetPETSCOpsLinearSolver(petsc_solver, ksp);
    }
    //对petsc_solver进行setup，检查参数，分配工作空间等
    GCGE_SOLVER_Setup(petsc_solver);
}

void GCGE_SOLVER_PETSC_Free(GCGE_SOLVER **petsc_solver)
{
    //释放特征值、特征向量、KSP空间
    free((*petsc_solver)->eval); (*petsc_solver)->eval = NULL;
    (*petsc_solver)->ops->MultiVecDestroy((void***)(&(*petsc_solver)->evec), 
            (*petsc_solver)->para->nev, (*petsc_solver)->ops);
    if((*petsc_solver)->ops->linear_solver_workspace != NULL)
    {
        KSPDestroy((KSP*)(&(*petsc_solver)->ops->linear_solver_workspace));
    }
    GCGE_SOLVER_Free(petsc_solver);
}
