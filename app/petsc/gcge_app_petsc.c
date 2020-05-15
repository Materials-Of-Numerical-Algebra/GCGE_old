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
    //PetscRandomSetFromOptions(rctx);
    PetscMPIInt    rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscRandomSetSeed(rctx, 0x12345678 + 76543*rank+rand());
    PetscRandomSeed(rctx);
    VecSetRandom((Vec)vec, rctx);
    PetscRandomDestroy(&rctx);
}
void GCGE_PETSC_MultiVecSetRandomValue(void **multi_vec, GCGE_INT start, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    PetscRandom    rctx;
    
    PetscInt       i;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);
    ///PetscRandomSetFromOptions(rctx);
    PetscMPIInt    rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscRandomSetSeed(rctx, 0x12345678 + 76543*rank+rand());
    PetscRandomSeed(rctx);
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

/* 返回 A_array B_array P_array num_levels */
/* get multigrid operator for num_levels = 4
 * P0     P1       P2
 * A0     A1       A2        A3
 * B0  P0'B0P0  P1'B1P1   P2'B2P2 
 * A0 is the original matrix */
/* 这里之所以是***A_array, 是因为，使用这个函数的时候需要将void**的变量取地址放入函数
 * Mat == void*
 * 还原到petsc中A_array其实是
 * Mat *A_array; ops->MultiGridCreate(&A_array, ...); */
void GCGE_PETSC_MultiGridCreate(void ***A_array, void ***B_array, void ***P_array, GCGE_INT *num_levels, void *A, void *B, struct GCGE_OPS_ *ops)
{
   /* P 是行多列少, P*v是从粗到细 */
   PetscInt m, n, level;
   Mat   *petsc_A_array = NULL, *petsc_B_array = NULL, *petsc_P_array = NULL;
   PC    pc;
   Mat   *Aarr=NULL, *Parr=NULL;

   PCCreate(PETSC_COMM_WORLD,&pc);
   PCSetOperators(pc,(Mat)A,(Mat)A);
   PCSetType(pc,PCGAMG);
   //PCMGSetLevels(pc, (*multi_grid)->num_levels, NULL);
   PCGAMGSetNlevels(pc, *num_levels);
   PCGAMGSetType(pc, PCGAMGCLASSICAL);
   //	type 	- PCGAMGAGG, PCGAMGGEO, or PCGAMGCLASSICAL
   //PetscReal th[2] = {0.0, 0.9};
   //PCGAMGSetThreshold(pc, th, 2);
   PCSetUp(pc);
   /* the size of Aarr is num_levels-1, Aarr is the coarsest matrix */
   PCGetCoarseOperators(pc, num_levels, &Aarr);
   /* the size of Parr is num_levels-1 */
   PCGetInterpolations(pc, num_levels, &Parr);

   /* we should make that zero is the refinest level */
   /* when num_levels == 5, 1 2 3 4 of A_array == 3 2 1 0 of Aarr */

   petsc_A_array = malloc(sizeof(Mat)*(*num_levels));
   petsc_P_array = malloc(sizeof(Mat)*((*num_levels)-1));

   //将原来的最细层A矩阵指针给A_array
   petsc_A_array[0] = (Mat)A;
   MatGetSize(petsc_A_array[0], &m, &n);
   PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", 0, m, n );
   for (level = 1; level < (*num_levels); ++level)
   {
      petsc_A_array[level] = Aarr[(*num_levels)-level-1];
      MatGetSize(petsc_A_array[level], &m, &n);
      PetscPrintf(PETSC_COMM_WORLD, "A_array[%d], m = %d, n = %d\n", level, m, n );

      petsc_P_array[level-1] = Parr[(*num_levels)-level-1];
      MatGetSize(petsc_P_array[level-1], &m, &n);
      PetscPrintf(PETSC_COMM_WORLD, "P_array[%d], m = %d, n = %d\n", level-1, m, n );
   }
   (*A_array) = (void**)petsc_A_array;
   (*P_array) = (void**)petsc_P_array;

   PetscFree(Aarr);
   PetscFree(Parr);
   PCDestroy(&pc);

   if (B!=NULL)
   {
      petsc_B_array = malloc(sizeof(Mat)*(*num_levels));
      petsc_B_array[0] = (Mat)B;
      MatGetSize(petsc_B_array[0], &m, &n);
      PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", 0, m, n );
      /* B0  P0^T B0 P0  P1^T B1 P1   P2^T B2 P2 */
      for ( level = 1; level < (*num_levels); ++level )
      {
	 MatPtAP(petsc_B_array[level-1], petsc_P_array[level-1], 
	       MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
	 MatGetSize(petsc_B_array[level], &m, &n);
	 PetscPrintf(PETSC_COMM_WORLD, "B_array[%d], m = %d, n = %d\n", level, m, n );
      }
      (*B_array) = (void**)petsc_B_array;
   }
}

/* free A1 A2 A3 B1 B2 B3 P0 P1 P2 
 * A0 and B0 are just pointers */
void GCGE_PETSC_MultiGridDestroy(void ***A_array, void ***B_array, void ***P_array, GCGE_INT *num_levels, struct GCGE_OPS_ *ops)
{
    Mat *petsc_A_array, *petsc_B_array, *petsc_P_array;
    petsc_A_array = (Mat *)(*A_array);
    petsc_P_array = (Mat *)(*P_array);
    int level; 
    for ( level = 1; level < (*num_levels); ++level )
    {
        MatDestroy(&(petsc_A_array[level]));
        MatDestroy(&(petsc_P_array[level-1]));
    }
    free(petsc_A_array);
    free(petsc_P_array);
    (*A_array) = NULL;
    (*P_array) = NULL;

    if (B_array!=NULL)
    {
       petsc_B_array = (Mat *)(*B_array);
       for ( level = 1; level < (*num_levels); ++level )
       {
	  MatDestroy(&(petsc_B_array[level]));
       }
       free(petsc_B_array);
       (*B_array) = NULL;
    }
}

void GCGE_PETSC_SetOps(GCGE_OPS *ops)
{
   /* either-or */
   ops->VecCreateByVec     = GCGE_PETSC_VecCreateByVec;
   ops->VecCreateByMat     = GCGE_PETSC_VecCreateByMat;
   ops->VecDestroy         = GCGE_PETSC_VecDestroy;

   ops->VecSetRandomValue  = GCGE_PETSC_VecSetRandomValue;
   ops->MatDotVec          = GCGE_PETSC_MatDotVec;
   ops->MatTransposeDotVec = GCGE_PETSC_MatTransposeDotVec;
   ops->VecAxpby           = GCGE_PETSC_VecAxpby;
   ops->VecInnerProd       = GCGE_PETSC_VecInnerProd;
   ops->VecLocalInnerProd  = GCGE_PETSC_VecLocalInnerProd;

   ops->MultiVecDestroy        = GCGE_PETSC_MultiVecDestroy;
   ops->MultiVecCreateByMat    = GCGE_PETSC_MultiVecCreateByMat;
   ops->MultiVecSetRandomValue = GCGE_PETSC_MultiVecSetRandomValue;

   ops->MultiGridCreate  = GCGE_PETSC_MultiGridCreate;
   ops->MultiGridDestroy = GCGE_PETSC_MultiGridDestroy;
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
