/*
 * =====================================================================================
 *
 *       Filename:  gcge_app_slepc.c
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
#include "gcge_app_slepc.h"

void SLEPC_ReadMatrixBinary(Mat *A, const char *filename)
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

void SLEPC_LinearSolverCreate(KSP *ksp, Mat A, Mat T)
{
    
    KSPCreate(PETSC_COMM_WORLD,ksp);
    KSPSetOperators(*ksp,A,T);
    KSPSetInitialGuessNonzero(*ksp, 1);

    KSPSetType(*ksp, KSPCG);
    //这里的rtol应取作<=ev_tol
    //  KSPSetTolerances(KSP ksp,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt maxits)
    KSPSetTolerances(*ksp, 1e-25, PETSC_DEFAULT, PETSC_DEFAULT, 15);
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
    //PetscViewer viewer;
    KSPView(*ksp, PETSC_VIEWER_STDOUT_WORLD);
}

void SLEPC_VecLocalInnerProd(Vec x, Vec y, double *value)
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


void GCGE_SLEPC_VecCreateByVec(void **d_vec, void *s_vec, GCGE_OPS *ops)
{
	
    VecDuplicate((Vec)s_vec, (Vec*)d_vec);
}
void GCGE_SLEPC_VecCreateByMat(void **vec, void *mat, GCGE_OPS *ops)
{
	
    MatCreateVecs((Mat)mat, NULL, (Vec*)vec);
}
void GCGE_SLEPC_VecDestroy(void **vec, GCGE_OPS *ops)
{
	
    VecDestroy((Vec*)vec);
}

void GCGE_SLEPC_VecSetRandomValue(void *vec, GCGE_OPS *ops)
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
void GCGE_SLEPC_MultiVecSetRandomValue(void **multi_vec, GCGE_INT start, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    PetscRandom    rctx;
    
    PetscInt       i;
    Vec            x;
    PetscRandomCreate(PETSC_COMM_WORLD,&rctx);

    //PetscRandomSetFromOptions(rctx);
    PetscMPIInt    rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscRandomSetSeed(rctx, 0x12345678 + 76543*rank+rand());
    PetscRandomSeed(rctx);

    for(i=0; i<n_vec; i++)
    {
        BVGetColumn((BV)multi_vec, i, &x);
        VecSetRandom(x, rctx);
        BVRestoreColumn((BV)multi_vec, i, &x);
    }
    PetscRandomDestroy(&rctx);
}

void GCGE_SLEPC_MatDotVec(void *mat, void *x, void *r, GCGE_OPS *ops)
{
     MatMult((Mat)mat, (Vec)x, (Vec)r);
}

void GCGE_SLEPC_MatTransposeDotVec(void *mat, void *x, void *r, GCGE_OPS *ops)
{
     MatMultTranspose((Mat)mat, (Vec)x, (Vec)r);
}

void GCGE_SLEPC_VecAxpby(GCGE_DOUBLE a, void *x, GCGE_DOUBLE b, void *y, GCGE_OPS *ops)
{
    
	VecScale((Vec)y, b);
    if(x != y)
    {
	    VecAXPY((Vec)y, a, (Vec)x);
    }
}
void GCGE_SLEPC_VecInnerProd(void *x, void *y, GCGE_DOUBLE *value_ip, GCGE_OPS *ops)
{
	 VecDot((Vec)x, (Vec)y, value_ip);
}

void GCGE_SLEPC_VecLocalInnerProd(void *x, void *y, GCGE_DOUBLE *value_ip, GCGE_OPS *ops)
{
    SLEPC_VecLocalInnerProd((Vec)x, (Vec)y, value_ip);
}

void GCGE_SLEPC_LinearSolver(void *Matrix, void *b, void *x, struct GCGE_OPS_ *ops)
{
    
    KSPSolve((KSP)(ops->linear_solver_workspace), (Vec)b, (Vec)x);
}

void GCGE_SLEPC_GetVecFromMultiVec(void **V, GCGE_INT j, void **x, GCGE_OPS *ops)
{
     BVGetColumn((BV)V, j, (Vec*)x);
}

void GCGE_SLEPC_RestoreVecForMultiVec(void **V, GCGE_INT j, void **x, GCGE_OPS *ops)
{
     BVRestoreColumn((BV)V, j, (Vec*)x);
}

void GCGE_SLEPC_MultiVecCreateByMat(void ***multi_vec, 
        GCGE_INT n_vec, void *mat, struct GCGE_OPS_ *ops)
{
	
	Vec            vector;
    MatCreateVecs((Mat)mat, NULL, &vector);
    BVCreate(PETSC_COMM_WORLD, (BV*)multi_vec);
    BVSetType((BV)(*multi_vec), BVMAT);
    BVSetSizesFromVec((BV)(*multi_vec), vector, n_vec);
    VecDestroy(&vector);
}

void GCGE_SLEPC_MultiVecDestroy(void ***MultiVec, GCGE_INT n_vec, struct GCGE_OPS_ *ops)
{
    GCGE_INT n = 1;
    
    BVGetSizes((BV)(*MultiVec), &n, NULL, NULL);
    BVDestroy((BV*)MultiVec);
}

void GCGE_SLEPC_MatDotMultiVec(void *mat, void **x, void **y, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    
    BVSetActiveColumns((BV)x, start[0], end[0]);
    BVSetActiveColumns((BV)y, start[1], end[1]);
    BVMatMult((BV)x, (Mat)mat, (BV)y);
}

/* vec_y[j] = \sum_{i=sx}^{ex} vec_x[i] a[i][j] */
//对连续的向量组进行线性组合: y(:,start[1]:end[1]) = x(:,start[0]:end[0])*a
void GCGE_SLEPC_MultiVecLinearComb(void **x, void **y, 
        GCGE_INT *start, GCGE_INT *end, GCGE_DOUBLE *a, GCGE_INT lda, 
        GCGE_INT if_Vec, GCGE_DOUBLE alpha, GCGE_DOUBLE beta,
        struct GCGE_OPS_ *ops)
{
    
    BVSetActiveColumns((BV)x, start[0], end[0]);
    if(if_Vec == 0)
    {
        BVSetActiveColumns((BV)y, start[1], end[1]);
        Mat dense_mat;
        //y = x * dense_mat，因此dense_mat的行数为x的列数,列数为y的列数
        GCGE_INT nrows = end[0];
        GCGE_INT ncols = end[1];
        MatCreateSeqDense(PETSC_COMM_SELF, nrows, ncols, NULL, &dense_mat);
        //将稠密矩阵a中的元素赋值给dense_mat
        GCGE_DOUBLE *q;
        MatDenseGetArray(dense_mat, &q);
        memset(q, 0.0, nrows*ncols*sizeof(GCGE_DOUBLE));
        GCGE_INT i = 0;
        for(i=start[1]; i<end[1]; i++)
        {
            //默认输入的矩阵a是要从第一个元素开始用的,所以q的位置要+start[0]
            memcpy(q+i*nrows+start[0], a+(i-start[1])*lda, (end[0]-start[0])*sizeof(GCGE_DOUBLE));
        }
        MatDenseRestoreArray(dense_mat, &q);
        BVMult((BV)y, alpha, beta, (BV)x, dense_mat);
        MatDestroy(&dense_mat);
    }
    else
    {
        BVMultVec((BV)x, alpha, beta, (Vec)(y[0]), a);
    }

}

// lda : leading dimension of matrix a
// a(start[0]:end[0],start[1]:end[1]) = V(:,start[0]:end[0])^T * W(:,start[1]:end[1])
void GCGE_SLEPC_MultiVecInnerProd(void **V, void **W, GCGE_DOUBLE *a, 
        char *is_sym, GCGE_INT *start, GCGE_INT *end, 
        GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops)
{
    
    BVSetActiveColumns((BV)V, start[0], end[0]);

    //如果W是BV结构
    if(if_Vec == 0)
    {
        //dense_mat = VT*W，因此dense_mat的行数为V的列数,列数为W的列数
        GCGE_INT nrows = end[0];
        GCGE_INT ncols = end[1];
        //col_length表示实际用到的稠密矩阵的列数,即W的列数
        GCGE_INT col_length = end[1]-start[1];
        GCGE_INT row_length = end[0]-start[0];
        BVSetActiveColumns((BV)W, start[1], end[1]);
        //计算VT*W的L2内积,要先把W的矩阵设为NULL
        BVSetMatrix((BV)W, NULL, PETSC_TRUE);
        Mat dense_mat;
        MatCreateSeqDense(PETSC_COMM_SELF, nrows, ncols, NULL, &dense_mat);

        BVDot((BV)W, (BV)V, dense_mat);

        //将稠密矩阵a中的元素赋值给dense_mat
        GCGE_DOUBLE *q;
        MatDenseGetArray(dense_mat, &q);
        GCGE_INT i = 0;
 
        for(i=0; i<col_length; i++)
        {
            //默认输入的矩阵a是要从第一个元素开始用的,q的位置要从start[1]列开始用,每列加start[0]
            memcpy(a+i*lda, q+(start[1]+i)*nrows+start[0], row_length*sizeof(GCGE_DOUBLE));
        }
        MatDenseRestoreArray(dense_mat, &q);
        MatDestroy(&dense_mat);
    }
    else
    {
        //如果W是Vec*结构
        BVSetMatrix((BV)V, NULL, PETSC_TRUE);
        BVDotVec((BV)V, (Vec)(W[0]), a);
    }

}

// lda : leading dimension of matrix a
// a(start[0]:end[0],start[1]:end[1]) = V(:,start[0]:end[0])^T * W(:,start[1]:end[1])
void GCGE_SLEPC_MultiVecInnerProdLocal(void **V, void **W, GCGE_DOUBLE *a, 
        char *is_sym, GCGE_INT *start, GCGE_INT *end, 
        GCGE_INT lda, GCGE_INT if_Vec, struct GCGE_OPS_ *ops)
{
    
    BVSetActiveColumns((BV)V, start[0], end[0]);

    const PetscScalar *V_array, *W_array;
    GCGE_INT     V_nrows,  V_ncols = end[0]-start[0];
    GCGE_INT     W_nrows,  W_ncols = end[1]-start[1];
    GCGE_DOUBLE alpha = 1.0;
    GCGE_DOUBLE beta  = 0.0;
    BVGetArrayRead((BV)V, &V_array);
    BVGetSizes((BV)V, &V_nrows, NULL, NULL);
    //如果W是BV结构
    if(if_Vec == 0)
    {
	BVGetArrayRead((BV)W, &W_array);
	BVGetSizes((BV)W, &W_nrows, NULL, NULL);

	char transa = 'T', transb = 'N';
	ops->DenseMatDotDenseMat(&transa, &transb, &V_ncols, &W_ncols, &V_nrows, 
	      &alpha, (double*)V_array+start[0]*V_nrows, &V_nrows, 
	      (double*)W_array+start[1]*W_nrows, &W_nrows, 
	      &beta, a, &lda);

	BVRestoreArrayRead((BV)W, &W_array);
    }
    else
    {
        //如果W是Vec*结构
	VecGetArrayRead((Vec)(W[0]), &W_array);
	W_nrows = V_nrows;

	char transa = 'T', transb = 'N';
	ops->DenseMatDotDenseMat(&transa, &transb, &V_ncols, &W_ncols, &V_nrows, 
	      &alpha, (double*)V_array+start[0]*V_nrows, &V_nrows, 
	      (double*)W_array, &W_nrows, &beta, a, &lda);

	VecRestoreArrayRead((Vec)(W[0]), &W_array);
    }
    BVRestoreArrayRead((BV)V, &V_array);

}

//计算 y = a * x + b * y
void GCGE_SLEPC_MultiVecAxpby(GCGE_DOUBLE a, void **x, GCGE_DOUBLE 
        b, void **y, GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    
    BVSetActiveColumns((BV)x, start[0], end[0]);
    BVSetActiveColumns((BV)y, start[1], end[1]);
    //If matrix Q is NULL, then an AXPY operation Y = beta*Y + alpha*X is done
    if(x == y)
    {
        BVScale((BV)y, a+b);
    }
    else
    {
        BVMult((BV)y, a, b, (BV)x, NULL);
    }
}

//把V_1和V_2的相应的列组交换: 即： V_1(:,start[0]:end[0])与V_2(:,start[1]:end[1])相互交换
void GCGE_SLEPC_MultiVecSwap(void **V_1, void **V_2, 
        GCGE_INT *start, GCGE_INT *end, struct GCGE_OPS_ *ops)
{
    
    if(V_1 != V_2)
    {
        BVSetActiveColumns((BV)V_1, start[0], end[0]);
        BVSetActiveColumns((BV)V_2, start[1], end[1]);
        //BVCopy是将前面的copy给后面的
        BVCopy((BV)V_2, (BV)V_1);
    }
    else
    {
        Vec x, y; 
        GCGE_INT i = 0;
        GCGE_INT length = end[0]-start[0];

        for(i=0; i<length; i++)
        {
            BVGetColumn((BV)V_1, i+start[0], &x);
            BVGetColumn((BV)V_2, i+start[1], &y);
            //VecCopy是将前面的copy给后面的
            VecCopy(y, x);
            BVRestoreColumn((BV)V_1, i+start[0], &x);
            BVRestoreColumn((BV)V_2, i+start[1], &y);
        }
    }
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
void GCGE_SLEPC_MultiGridCreate(void ***A_array, void ***B_array, void ***P_array, GCGE_INT *num_levels, void *A, void *B, struct GCGE_OPS_ *ops)
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
   /* not force coarse grid onto one processor */
   PCGAMGSetUseParallelCoarseGridSolve(pc,PETSC_TRUE);
   /* this will generally improve the loading balancing of the work on each level 
    * should use parmetis */
//   PCGAMGSetRepartition(pc, PETSC_TRUE);
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
void GCGE_SLEPC_MultiGridDestroy(void ***A_array, void ***B_array, void ***P_array, GCGE_INT *num_levels, struct GCGE_OPS_ *ops)
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

void GCGE_SLEPC_MultiVecPrint(void **x, GCGE_INT n, struct GCGE_OPS_ *ops)
{
   BVView((BV)x, PETSC_VIEWER_STDOUT_WORLD);
}

void GCGE_SLEPC_SetOps(GCGE_OPS *ops)
{
    /* either-or */
    ops->VecCreateByVec     = GCGE_SLEPC_VecCreateByVec;
    ops->VecCreateByMat     = GCGE_SLEPC_VecCreateByMat;
    ops->VecDestroy         = GCGE_SLEPC_VecDestroy;

    ops->VecSetRandomValue  = GCGE_SLEPC_VecSetRandomValue;
    ops->MatDotVec          = GCGE_SLEPC_MatDotVec;
    ops->MatTransposeDotVec = GCGE_SLEPC_MatTransposeDotVec;
    ops->VecAxpby           = GCGE_SLEPC_VecAxpby;
    ops->VecInnerProd       = GCGE_SLEPC_VecInnerProd;
    ops->VecLocalInnerProd  = GCGE_SLEPC_VecLocalInnerProd;

    ops->MultiVecDestroy        = GCGE_SLEPC_MultiVecDestroy;
    ops->MultiVecCreateByMat    = GCGE_SLEPC_MultiVecCreateByMat;
    ops->MultiVecSetRandomValue = GCGE_SLEPC_MultiVecSetRandomValue;

    ops->GetVecFromMultiVec    = GCGE_SLEPC_GetVecFromMultiVec;
    ops->RestoreVecForMultiVec = GCGE_SLEPC_RestoreVecForMultiVec;
    //ops->MatDotMultiVec = GCGE_SLEPC_MatDotMultiVec;

    ops->MultiVecLinearComb = GCGE_SLEPC_MultiVecLinearComb;
    ops->MultiVecInnerProd  = GCGE_SLEPC_MultiVecInnerProd;
    ops->MultiVecInnerProdLocal = GCGE_SLEPC_MultiVecInnerProdLocal;
    ops->MultiVecSwap  = GCGE_SLEPC_MultiVecSwap;
    ops->MultiVecAxpby = GCGE_SLEPC_MultiVecAxpby;
    ops->MultiVecPrint = GCGE_SLEPC_MultiVecPrint;

    ops->MultiGridCreate  = GCGE_SLEPC_MultiGridCreate;
    ops->MultiGridDestroy = GCGE_SLEPC_MultiGridDestroy;
}

void GCGE_SOLVER_SetSLEPCOps(GCGE_SOLVER *solver)
{
    GCGE_SLEPC_SetOps(solver->ops);
}

void GCGE_OPS_CreateSLEPC(GCGE_OPS **ops)
{
    GCGE_OPS_Create(ops);
    GCGE_SLEPC_SetOps(*ops);
    GCGE_OPS_Setup(*ops);
}

//下面是一个对SLEPC 的GCG_Solver的初始化
//用户只提供要求解特征值问题的矩阵A,B,线性解法器等都用默认值
//如果用户想在命令行提供特征值个数，需要将num_eigenvalues赋值为-1
GCGE_SOLVER* GCGE_SLEPC_Solver_Init(Mat A, Mat B, int num_eigenvalues, int argc, char* argv[])
{
    //第一步: 定义相应的ops,para以及workspace

    //创建一个solver变量
    GCGE_SOLVER *slepc_solver;
    GCGE_SOLVER_Create(&slepc_solver);
    if(num_eigenvalues != -1)
        slepc_solver->para->nev = num_eigenvalues;
    GCGE_PARA_SetFromCommandLine(slepc_solver->para, argc, argv);
    //设置初始值
    int nev = slepc_solver->para->nev;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    slepc_solver->eval = eval;

	
    Vec *evec;
	Vec vector;
    MatCreateVecs(A, NULL, &vector);
    VecDuplicateVecs(vector, nev, &evec);

    GCGE_SOLVER_SetMatA(slepc_solver, A);
    if(B != NULL)
        GCGE_SOLVER_SetMatB(slepc_solver, B);
    GCGE_SOLVER_SetSLEPCOps(slepc_solver);
    GCGE_SOLVER_SetEigenvalues(slepc_solver, eval);
    GCGE_SOLVER_SetEigenvectors(slepc_solver, (void**)evec);
    //setup and solve
    GCGE_SOLVER_Setup(slepc_solver);
    GCGE_PrintParaInfo(slepc_solver->para);
    GCGE_Printf("Set up finish!\n");
    return slepc_solver;
}

//下面是一个对SLEPC 的GCG_Solver的初始化
//用户只提供要求解特征值问题的矩阵A,B,线性解法器使用KSPSolve,用户不接触ksp,设置参数均从命令行设置
GCGE_SOLVER* GCGE_SLEPC_Solver_Init_KSPDefault(Mat A, Mat B, Mat P, int num_eigenvalues, int argc, char* argv[])
{
    //首先用矩阵A,B创建slepc_solver
    GCGE_SOLVER *slepc_solver = GCGE_SLEPC_Solver_Init(A, B, num_eigenvalues, argc, argv);
    KSP ksp;
    //使用矩阵A和预条件矩阵P创建ksp,ksp的参数均为默认参数,要修改ksp参数只能用命令行参数进行设置
    SLEPC_LinearSolverCreate(&ksp, (Mat)A, (Mat)P);
    //把ksp作为ops->linear_solver_workspace
    GCGE_SOLVER_SetOpsLinearSolverWorkspace(slepc_solver, (void*)ksp);
    //将线性解法器设为KSPSolve
    slepc_solver->ops->LinearSolver = GCGE_SLEPC_LinearSolver;
    return slepc_solver;
}

//下面是一个对SLEPC 的GCG_Solver的初始化
//用户提供要求解特征值问题的矩阵A,B,以及线性解法器结构ksp,使用KSPSolve作为线性解法器
GCGE_SOLVER* GCGE_SLEPC_Solver_Init_KSPGivenByUser(Mat A, Mat B, KSP ksp, int num_eigenvalues, int argc, char* argv[])
{
    //首先用矩阵A,B创建slepc_solver
    GCGE_SOLVER *slepc_solver = GCGE_SLEPC_Solver_Init(A, B, num_eigenvalues, argc, argv);
    //用户已创建好ksp,直接赋给slepc_solver
    GCGE_SOLVER_SetOpsLinearSolverWorkspace(slepc_solver, (void*)ksp);
    //将线性解法器设为KSPSolve
    slepc_solver->ops->LinearSolver = GCGE_SLEPC_LinearSolver;
    return slepc_solver;
}

void GCGE_SOLVER_SetSLEPCOpsLinearSolver(GCGE_SOLVER *solver, KSP ksp)
{
    GCGE_SOLVER_SetOpsLinearSolverWorkspace(solver, (void*)ksp);
    //将线性解法器设为KSPSolve
    solver->ops->LinearSolver = GCGE_SLEPC_LinearSolver;
}
