/*
 * =====================================================================================
 *
 *       Filename:  test_amg_gcge.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2020年05月16日 15时50分13秒
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include "gcge.h"
//#include "pase_convert.h"
#include "gcge_app_slepc.h"


static char help[] = "Test MultiGrid in GCGE.\n";
void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
void PETSCPrintMat(Mat A, char *name);
void PETSCPrintVec(Vec x);
void PETSCPrintBV(BV x, char *name);
void RedistributeDataOfMultiGridMatrixOnEachProcess(
     Mat * petsc_A_array, Mat *petsc_B_array, Mat *petsc_P_array, 
     PetscReal *proc_rate, PetscInt num_levels, PetscInt unit);


/* 
 *  Description:  测试PASE_MULTIGRID
 */
int
main ( int argc, char *argv[] )
{
   SlepcInitialize(&argc,&argv,(char*)0,help);

   PetscMPIInt   rank, size;
   PetscViewer   viewer;
   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_size(PETSC_COMM_WORLD, &size);

   /* 创建gcge_ops */
   GCGE_OPS *gcge_ops;
   GCGE_OPS_Create(&gcge_ops);
   GCGE_SLEPC_SetOps(gcge_ops);
   GCGE_OPS_Setup(gcge_ops);

   /* 得到PETSC矩阵A, B, 规模为n*m */
   Mat      petsc_mat_A, petsc_mat_B;
   PetscInt n = 25, m = 25;
   GetPetscMat(&petsc_mat_A, &petsc_mat_B, n, m);

   /* 生成分层矩阵 num_levels可能会在创建多层矩阵之后变小 */
   PetscInt idx, num_levels = 3;
   Mat *petsc_A_array, *petsc_B_array, *petsc_P_array;
   /* INPUT : petsc_mat_A petsc_mat_B num_levels 
    * OUTPUT: petsc_A_array petsc_B_array petsc_P_array num_levels
    * For example, getting multigrid operator for num_levels = 4
    * P0     P1       P2
    * A0     A1       A2        A3
    * B0  P0'B0P0  P1'B1P1   P2'B2P2 */
   gcge_ops->MultiGridCreate((void***)&petsc_A_array, (void***)&petsc_B_array, (void***)&petsc_P_array, 
	 &num_levels, (void*)petsc_mat_A, (void*)petsc_mat_B, gcge_ops);
   PetscInt local_m, local_n;
   for (idx = 0; idx < num_levels; ++idx)
   {
      MatGetLocalSize(petsc_A_array[idx], &local_m, &local_n);
      PetscPrintf(PETSC_COMM_SELF,"[%D] original A[%D] local size %D * %D\n", rank, idx, local_m, local_n);
   }

   /* 对整个进程, 重新分配各层矩阵数据 
    * 保证每层实际拥有数据的进程数nbigranks是unit的倍数(比如LSSC4 unit=36)
    * nbigranks[level] = size*proc_rate[level]
    * 进程号从0开始到nbigranks-1
    * proc_rate[0]是无效参数，即不改变最细层的矩阵分配 
    * 若proc_rate设为(0,1)之外，则不进行数据重分配 表示不会改变idx-1层的P和idx层的A */
   GCGE_INT    unit = 2;
   GCGE_DOUBLE *proc_rate = malloc(num_levels*sizeof(GCGE_DOUBLE));
   for (idx = 0; idx < num_levels; ++idx)
   {
      proc_rate[idx]   = -1.0;
   }
   proc_rate[num_levels-1] = 0.5;
   RedistributeDataOfMultiGridMatrixOnEachProcess(
	 petsc_A_array, petsc_B_array, petsc_P_array, 
	 proc_rate, num_levels, unit);
   for (idx = 0; idx < num_levels; ++idx)
   {
      MatGetLocalSize(petsc_A_array[idx], &local_m, &local_n);
      PetscPrintf(PETSC_COMM_SELF,"[%D] new A[%D] local size %D * %D\n", rank, idx, local_m, local_n);
   }

   /* 对于AMG分配工作空间，5*num_levels个BV + 5*num_vecs个double + 1*num_vecs个int
    *
    * start_bx  end_bx  描述 multi_vec_rhs multi_vec_sol
    * start_rpw end_rpw 描述 multi_vec_r   multi_vec_p   multi_vec_w 
    *
    * dtmp itmp 在BCG中存储
    * GCGE_DOUBLE *rho1       = bpcg->dtmp,
    *             *rho2       = rho1       + num_vecs,
    *             *pTw        = rho2       + num_vecs,
    *             *init_error = pTw        + num_vecs,
    *             *last_error = init_error + num_vecs;
    * GCGE_INT    *unlock     = bpcg->itmp;
    */
   GCGE_INT num_vecs      = 3;
   GCGE_INT start_bx[2]   = {0, 0},     end_bx[2]  = {num_vecs, num_vecs};     
   GCGE_INT start_rpw[3]  = {0, 0, 0},  end_rpw[3] = {num_vecs, num_vecs, num_vecs}; 
   GCGE_DOUBLE *dtmp      = malloc(5*num_vecs*sizeof(GCGE_DOUBLE));   // 5*num_vecs
   GCGE_INT    *itmp      = malloc(1*num_vecs*sizeof(GCGE_INT));      // 1*num_vecs
   /* 每一层的BCG都有自己的参数：最大迭代数，收敛比率，误差限 */
   GCGE_INT    *max_it    = malloc(num_levels*sizeof(GCGE_INT));
   GCGE_DOUBLE *conv_rate = malloc(num_levels*sizeof(GCGE_DOUBLE));
   GCGE_DOUBLE *tol       = malloc(num_levels*sizeof(GCGE_DOUBLE));
   for (idx = 0; idx < num_levels; ++idx)
   {
      max_it[idx]      = 100;
      conv_rate[idx]   = 1e-6;
      tol[idx]         = 1e-15;
   }
   /* 生成每层上的多向量 */
   BV *multi_vec_rhs = malloc(num_levels*sizeof(BV));
   BV *multi_vec_sol = malloc(num_levels*sizeof(BV));
   BV *multi_vec_r   = malloc(num_levels*sizeof(BV));
   BV *multi_vec_p   = malloc(num_levels*sizeof(BV));
   BV *multi_vec_w   = malloc(num_levels*sizeof(BV));
   /* 将multi_vec_rhs赋值为零，multi_vec_sol随机取值, 精确解就是零 */
   for (idx = 0; idx < num_levels; ++idx)
   {
      gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_rhs[idx]), num_vecs, 
	    (void*)petsc_A_array[idx], gcge_ops);
      gcge_ops->MultiVecAxpby(0.0, (void**)(multi_vec_rhs[idx]), 0.0, (void**)(multi_vec_rhs[idx]), 
	    start_bx, end_bx, gcge_ops);
      gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_sol[idx]), num_vecs, 
	    (void*)petsc_A_array[idx], gcge_ops);
      gcge_ops->MultiVecSetRandomValue((void**)(multi_vec_sol[idx]), 0, num_vecs, gcge_ops);

      gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_r[idx]),   num_vecs, 
	    (void*)petsc_A_array[idx], gcge_ops);
      gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_p[idx]),   num_vecs, 
	    (void*)petsc_A_array[idx], gcge_ops);
      gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_w[idx]),   num_vecs, 
	    (void*)petsc_A_array[idx], gcge_ops);
   }
   /* 在调用内置BAMG或者BPCG或者PCG之前先调用相应的Setup函数 */
   GCGE_MultiLinearSolverSetup_BAMG(
	 max_it,                   conv_rate,                     tol, 
	 (void**)petsc_A_array,    (void**)petsc_P_array, 
	 num_levels,
	 (void***)multi_vec_rhs,   (void***)multi_vec_sol, 
	 start_bx,                 end_bx,
	 (void***)multi_vec_r,     (void***)multi_vec_p,     (void***)multi_vec_w,
	 start_rpw,                end_rpw, 
	 dtmp,                     itmp, 
	 NULL, 
	 gcge_ops);
   //    BVView(multi_vec_sol[0], viewer);
   BVNorm(multi_vec_sol[0], NORM_FROBENIUS, dtmp);
   PetscPrintf(PETSC_COMM_WORLD, "norm of initi sol is %e\n", dtmp[0]);
   GCGE_MultiLinearSolver_BAMG((void *)petsc_mat_A, 
	 (void **)(multi_vec_rhs[0]), (void **)(multi_vec_sol[0]), 
	 start_bx, end_bx, gcge_ops);
   BVNorm(multi_vec_sol[0], NORM_FROBENIUS, dtmp);
   PetscPrintf(PETSC_COMM_WORLD, "norm of final sol is %e\n", dtmp[0]);
   //   BVView(multi_vec_sol[0], viewer);

   /* 销毁BAMG工作空间 */
   for (idx = 0; idx < num_levels; ++idx)
   {
      gcge_ops->MultiVecDestroy((void***)(&multi_vec_rhs[idx]), num_vecs, gcge_ops);
      gcge_ops->MultiVecDestroy((void***)(&multi_vec_sol[idx]), num_vecs, gcge_ops);
      gcge_ops->MultiVecDestroy((void***)(&multi_vec_r[idx]),   num_vecs, gcge_ops);
      gcge_ops->MultiVecDestroy((void***)(&multi_vec_p[idx]),   num_vecs, gcge_ops);
      gcge_ops->MultiVecDestroy((void***)(&multi_vec_w[idx]),   num_vecs, gcge_ops);
   }
   free(multi_vec_rhs); free(multi_vec_sol);
   free(multi_vec_r);   free(multi_vec_p);   free(multi_vec_w);
   free(dtmp);          free(itmp);
   free(max_it);        free(conv_rate);     free(tol);
   /* 销毁MG各层矩阵 */
   gcge_ops->MultiGridDestroy((void***)&petsc_A_array, (void***)&petsc_B_array, (void***)&petsc_P_array,
	 &num_levels, gcge_ops);
   /* 销毁petsc矩阵 */
   MatDestroy(&petsc_mat_A);
   MatDestroy(&petsc_mat_B);
   /* 销毁ops */
   GCGE_OPS_Free(&gcge_ops);
   /* PetscFinalize */
   SlepcFinalize();
   return 0;
}

/**
 * @brief 
 *    nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
 *    if (nbigranks < unit) nbigranks = unit<size?unit:size;
 *
 * @param petsc_A_array
 * @param petsc_B_array
 * @param petsc_P_array
 * @param proc_rate
 * @param num_levels
 * @param unit           保证每层nbigranks是unit的倍数
 */
void RedistributeDataOfMultiGridMatrixOnEachProcess(
     Mat * petsc_A_array, Mat *petsc_B_array, Mat *petsc_P_array, 
     PetscReal *proc_rate, PetscInt num_levels, PetscInt unit)
{
   PetscMPIInt   rank, size;
//   PetscViewer   viewer;

   PetscInt      level, row;
   Mat           new_P_H;
   PetscMPIInt   nbigranks;
   PetscInt      global_nrows, global_ncols; 
   PetscInt      local_nrows,  local_ncols;
   PetscInt      new_local_ncols;
   /* 保证每层nbigranks是unit的倍数 */
   PetscInt      rstart, rend, ncols;
   const PetscInt              *cols; 
   const PetscScalar           *vals;

   MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
   MPI_Comm_size(PETSC_COMM_WORLD, &size);

   if (proc_rate[0]<1.0 && proc_rate[0]>0.0)
   {
      PetscPrintf(PETSC_COMM_WORLD, "Warning the refinest matrix cannot be redistributed\n");
   }

   for (level = 1; level < num_levels; ++level)
   {
      /* 若proc_rate设为(0,1)之外，则不进行数据重分配 */
      if (proc_rate[level]>=1.0 || proc_rate[level]<=0.0)
      {
	 PetscPrintf(PETSC_COMM_WORLD, "Retain data distribution of %D level\n", level);
	 continue; /* 直接到下一次循环 */
      }
      else
      {
	 PetscPrintf(PETSC_COMM_WORLD, "Redistribute data of %D level\n", level);
      }
      MatGetSize(petsc_P_array[level-1], &global_nrows, &global_ncols);
      /* 在设定new_P_H的局部行时已经不能用以前P的局部行，因为当前层的A可能已经改变 */
      MatGetLocalSize(petsc_A_array[level-1], &local_nrows, &local_ncols);
      /* 应该通过ncols_P，即最粗层矩阵大小和进程总数size确定nbigranks */
      nbigranks = ((PetscInt)((((PetscReal)size)*proc_rate[level])/((PetscReal)unit))) * (unit);
      if (nbigranks < unit) nbigranks = unit<size?unit:size;
      PetscPrintf(PETSC_COMM_WORLD, "nbigranks[%D] = %D\n", level, nbigranks);
      /* 对0到nbigranks-1进程平均分配global_ncols */
      new_local_ncols = 0;
      if (rank < nbigranks)
      {
	 new_local_ncols = global_ncols/nbigranks;
	 if (rank < global_ncols%nbigranks)
	 {
	    ++new_local_ncols;
	 }
      }
      /* 创建新的延拓矩阵, 并用原始的P为之赋值 */
      MatCreate(PETSC_COMM_WORLD, &new_P_H);
      MatSetSizes(new_P_H, local_nrows, new_local_ncols, global_nrows, global_ncols);
      //MatSetFromOptions(new_P_H);
      /* can be improved */
      //       MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
      //       MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
      MatSetUp(new_P_H);
      MatGetOwnershipRange(petsc_P_array[level-1], &rstart, &rend);
      for(row = rstart; row < rend; ++row) 
      {
	 MatGetRow(petsc_P_array[level-1], row, &ncols, &cols, &vals);
	 MatSetValues(new_P_H, 1, &row, ncols, cols, vals, INSERT_VALUES);
	 MatRestoreRow(petsc_P_array[level-1], row, &ncols, &cols, &vals);
      }
      MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);

      MatGetLocalSize(petsc_P_array[level-1], &local_nrows, &local_ncols);
      PetscPrintf(PETSC_COMM_SELF, "[%D] original P_H[%D] local size %D * %D\n", 
	    rank, level, local_nrows, local_ncols);
      MatGetLocalSize(new_P_H, &local_nrows, &local_ncols);
      PetscPrintf(PETSC_COMM_SELF, "[%D] new P_H[%D] local size %D * %D\n", 
	    rank, level, local_nrows, local_ncols);
      //       MatView(petsc_P_array[level-1], viewer);
      //       MatView(new_P_H, viewer);

      /* 销毁之前的P_H A_H B_H */
      MatDestroy(&(petsc_P_array[level-1]));
      MatDestroy(&(petsc_A_array[level]));
      if (petsc_B_array!=NULL)
      {
	 MatDestroy(&(petsc_B_array[level]));
      }

      petsc_P_array[level-1] = new_P_H;
      MatPtAP(petsc_A_array[level-1], petsc_P_array[level-1],
	    MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_A_array[level]));
      if (petsc_B_array!=NULL)
      {
	 MatPtAP(petsc_B_array[level-1], petsc_P_array[level-1],
	       MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[level]));
      }
      //       MatView(petsc_A_array[num_levels-1], viewer);
      //       MatView(petsc_B_array[num_levels-1], viewer);
   }
}


void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
{
    PetscInt N = n*m;
    PetscInt Istart, Iend, II, i, j;
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD,A);
    ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,N,N);
//    ierr = MatSetFromOptions(*A);
    ierr = MatSetUp(*A);
    ierr = MatGetOwnershipRange(*A,&Istart,&Iend);
    for (II=Istart;II<Iend;II++) {
      i = II/n; j = II-i*n;
      if (i>0) { ierr = MatSetValue(*A,II,II-n,-1.0,INSERT_VALUES); }
      if (i<m-1) { ierr = MatSetValue(*A,II,II+n,-1.0,INSERT_VALUES); }
      if (j>0) { ierr = MatSetValue(*A,II,II-1,-1.0,INSERT_VALUES); }
      if (j<n-1) { ierr = MatSetValue(*A,II,II+1,-1.0,INSERT_VALUES); }
      ierr = MatSetValue(*A,II,II,4.0,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);

    ierr = MatCreate(PETSC_COMM_WORLD,B);
    ierr = MatSetSizes(*B,PETSC_DECIDE,PETSC_DECIDE,N,N);
//    ierr = MatSetFromOptions(*B);
    ierr = MatSetUp(*B);
    ierr = MatGetOwnershipRange(*B,&Istart,&Iend);
    for (II=Istart;II<Iend;II++) {
      ierr = MatSetValue(*B,II,II,1.0,INSERT_VALUES);
    }
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);

}

void PETSCPrintMat(Mat A, char *name)
{
    PetscErrorCode ierr;
    int nrows = 0, ncols = 0;
    ierr = MatGetSize(A, &nrows, &ncols);
    int row = 0, col = 0;
    const PetscInt *cols;
    const PetscScalar *vals;
    for(row=0; row<nrows; row++) {
        ierr = MatGetRow(A, row, &ncols, &cols, &vals);
        for(col=0; col<ncols; col++) {
            GCGE_Printf("%s(%d, %d) = %18.15e;\n", name, row+1, cols[col]+1, vals[col]);
        }
        ierr = MatRestoreRow(A, row, &ncols, &cols, &vals);
    }
}

void PETSCPrintVec(Vec x)
{
    PetscErrorCode ierr;
    int size = 0;
    int i = 0;
    ierr = VecGetSize(x, &size);
    const PetscScalar *array;
    ierr = VecGetArrayRead(x, &array);
    for(i=0; i<size; i++)
    {
        GCGE_Printf("%18.15e\t", array[i]);
    }
    ierr = VecRestoreArrayRead(x, &array);
    GCGE_Printf("\n");
}

void PETSCPrintBV(BV x, char *name)
{
    PetscErrorCode ierr;
    int n = 0, i = 0;
    ierr = BVGetSizes(x, NULL, NULL, &n);
    Vec xs;
    GCGE_Printf("%s = [\n", name);
    for(i=0; i<n; i++)
    {
        ierr = BVGetColumn(x, i, &xs);
	PETSCPrintVec(xs);
        ierr = BVRestoreColumn(x, i, &xs);
    }
    GCGE_Printf("];\n");
}



//    {
//       Mat new_P_H;
//       PetscMPIInt nbigranks;
//       PetscInt global_nrows, global_ncols; 
//       PetscInt local_nrows,  local_ncols;
//       PetscInt new_local_ncols;
//       MatGetSize(petsc_P_array[num_levels-2], &global_nrows, &global_ncols);
//       MatGetLocalSize(petsc_P_array[num_levels-2], &local_nrows, &local_ncols);
//       PetscPrintf(PETSC_COMM_SELF,"[%d] original P_H local size %D * %D\n", 
//	     rank, local_nrows, local_ncols);
//       /* 应该通过ncols_P，即最粗层矩阵大小和进程总数size确定nbigranks */
//       nbigranks = size/2 + 1;
//       /* 对0到nbigranks-1进程平均分配global_ncols */
//       new_local_ncols = 0;
//       if (rank < nbigranks)
//       {
//	  new_local_ncols = global_ncols/nbigranks;
//	  if (rank < global_ncols%nbigranks)
//	  {
//	     ++new_local_ncols;
//	  }
//       }
//       /* 创建新的延拓矩阵, 并用原始的P为之赋值 */
//       PetscInt rstart, rend, ncols;
//       const PetscInt *cols; 
//       const PetscScalar *vals;
//       MatCreate(PETSC_COMM_WORLD, &new_P_H);
//       MatSetSizes(new_P_H, local_nrows, new_local_ncols, global_nrows, global_ncols);
//       MatSetFromOptions(new_P_H);
//       /* can be improved */
////       MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
////       MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
//       MatSetUp(new_P_H);
//       MatGetOwnershipRange(petsc_P_array[num_levels-2], &rstart, &rend);
//       for(idx = rstart; idx < rend; ++idx) 
//       {
//	  MatGetRow(petsc_P_array[num_levels-2], idx, &ncols, &cols, &vals);
//	  MatSetValues(new_P_H, 1, &idx, ncols, cols, vals, INSERT_VALUES);
//	  MatRestoreRow(petsc_P_array[num_levels-2], idx, &ncols, &cols, &vals);
//       }
//       MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
//       MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);
//       MatGetLocalSize(new_P_H, &local_nrows, &local_ncols);
//       PetscPrintf(PETSC_COMM_SELF,"[%d] new P_H local size %D * %D\n", 
//	     rank, local_nrows, new_local_ncols);
////       MatView(petsc_P_array[num_levels-2], viewer);
////       MatView(new_P_H, viewer);
//
//       /* 销毁之前的P_H A_H B_H */
//       MatDestroy(&(petsc_P_array[num_levels-2]));
//       MatDestroy(&(petsc_A_array[num_levels-1]));
//       MatDestroy(&(petsc_B_array[num_levels-1]));
//
//       petsc_P_array[num_levels-2] = new_P_H;
//       MatPtAP(petsc_A_array[num_levels-2], petsc_P_array[num_levels-2],
//	     MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_A_array[num_levels-1]));
//       MatPtAP(petsc_B_array[num_levels-2], petsc_P_array[num_levels-2],
//	     MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[num_levels-1]));
////       MatView(petsc_A_array[num_levels-1], viewer);
////       MatView(petsc_B_array[num_levels-1], viewer);
//    }
