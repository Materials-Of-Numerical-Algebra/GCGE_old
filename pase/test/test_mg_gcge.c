/*
 * =====================================================================================
 *
 *       Filename:  test_mg_gcge.c
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

    /* 得到PETSC矩阵A, B */
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
       PetscPrintf(PETSC_COMM_SELF,"[%d] Matrix local size %D * %D\n", rank, local_m, local_n);
    }

#if 0
    /* 打印各层矩阵 */
    PetscPrintf(PETSC_COMM_WORLD, "A_array\n");
    for (idx = 0; idx < num_levels; ++idx)
    {
        PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
        MatView(petsc_A_array[idx], viewer);
    }
    PetscPrintf(PETSC_COMM_WORLD, "B_array\n");
    for (idx = 0; idx < num_levels; ++idx)
    {
        PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
        MatView(petsc_B_array[idx], viewer);
    }
    PetscPrintf(PETSC_COMM_WORLD, "P_array\n");
    for (idx = 0; idx < num_levels-1; ++idx)
    {
        PetscPrintf(PETSC_COMM_WORLD, "idx = %d\n", idx);
        MatView(petsc_P_array[idx], viewer);
    }
//    PETSCPrintMat(petsc_A_array[0], "A0");
//    PETSCPrintMat(petsc_B_array[0], "B0");
//    PETSCPrintMat(petsc_P_array[0], "P0");
#endif

    /* 构造新的P_H矩阵，使得部分进程拥有其行 */
    /* 可以写成一个函数
     * INPUT  原始的P
     * OUTPUT 新的P和A */
    {
       Mat new_P_H;
       PetscMPIInt nbigranks;
       PetscInt global_nrows, global_ncols; 
       PetscInt local_nrows,  local_ncols;
       PetscInt new_local_ncols;
       MatGetSize(petsc_P_array[num_levels-2], &global_nrows, &global_ncols);
       MatGetLocalSize(petsc_P_array[num_levels-2], &local_nrows, &local_ncols);
       PetscPrintf(PETSC_COMM_SELF,"[%d] original P_H local size %D * %D\n", 
	     rank, local_nrows, local_ncols);
       /* 应该通过ncols_P，即最粗层矩阵大小和进程总数size确定nbigranks */
       nbigranks = size/2 + 1;
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
       PetscInt rstart, rend, ncols;
       const PetscInt *cols; 
       const PetscScalar *vals;
       MatCreate(PETSC_COMM_WORLD, &new_P_H);
       MatSetSizes(new_P_H, local_nrows, new_local_ncols, global_nrows, global_ncols);
       MatSetFromOptions(new_P_H);
       /* can be improved */
//       MatSeqAIJSetPreallocation(new_P_H, 5, NULL);
//       MatMPIAIJSetPreallocation(new_P_H, 3, NULL, 2, NULL);
       MatSetUp(new_P_H);
       MatGetOwnershipRange(petsc_P_array[num_levels-2], &rstart, &rend);
       for(idx = rstart; idx < rend; ++idx) 
       {
	  MatGetRow(petsc_P_array[num_levels-2], idx, &ncols, &cols, &vals);
	  MatSetValues(new_P_H, 1, &idx, ncols, cols, vals, INSERT_VALUES);
	  MatRestoreRow(petsc_P_array[num_levels-2], idx, &ncols, &cols, &vals);
       }
       MatAssemblyBegin(new_P_H,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(new_P_H,MAT_FINAL_ASSEMBLY);
       MatGetLocalSize(new_P_H, &local_nrows, &local_ncols);
       PetscPrintf(PETSC_COMM_SELF,"[%d] new P_H local size %D * %D\n", 
	     rank, local_nrows, new_local_ncols);
//       MatView(petsc_P_array[num_levels-2], viewer);
//       MatView(new_P_H, viewer);

       /* 销毁之前的P_H A_H B_H */
       MatDestroy(&(petsc_P_array[num_levels-2]));
       MatDestroy(&(petsc_A_array[num_levels-1]));
       MatDestroy(&(petsc_B_array[num_levels-1]));

       petsc_P_array[num_levels-2] = new_P_H;
       MatPtAP(petsc_A_array[num_levels-2], petsc_P_array[num_levels-2],
	     MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_A_array[num_levels-1]));
       MatPtAP(petsc_B_array[num_levels-2], petsc_P_array[num_levels-2],
	     MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(petsc_B_array[num_levels-1]));
//       MatView(petsc_A_array[num_levels-1], viewer);
//       MatView(petsc_B_array[num_levels-1], viewer);
    }

#if 0
    /* 测试向量延拓和限制 */
    PetscInt level_i, level_j;
    level_i = 0; level_j = num_levels-1;
    Vec vec_i, vec_j;
    gcge_ops->VecCreateByMat((void**)&vec_i, (void*)petsc_A_array[level_i], gcge_ops);
    gcge_ops->VecCreateByMat((void**)&vec_j, (void*)petsc_A_array[level_j], gcge_ops);
    gcge_ops->VecSetRandomValue((void*)vec_i, gcge_ops);
    gcge_ops->VecSetRandomValue((void*)vec_j, gcge_ops);
    /* 测试FromItoJ, 需要所有层上的临时向量 */
    Vec *vec_tmp = malloc(num_levels*sizeof(Vec));
    for (idx = 0; idx < num_levels; ++idx)
    {
       gcge_ops->VecCreateByMat((void**)(&vec_tmp[idx]), (void*)petsc_A_array[idx], gcge_ops);
    }
    gcge_ops->VecFromItoJ((void**)petsc_P_array, level_i, level_j, 
	  (void*)vec_i, (void*)vec_j, (void**)vec_tmp, gcge_ops);
    VecView(vec_i, viewer);
    VecView(vec_j, viewer);
    gcge_ops->VecFromItoJ((void**)petsc_P_array, level_j, level_i, 
	  (void*)vec_j, (void*)vec_i, (void**)vec_tmp, gcge_ops);
    VecView(vec_i, viewer);
    VecView(vec_j, viewer);
//    PETSCPrintVec(vec_i);
//    PETSCPrintVec(vec_j);
    for (idx = 0; idx < num_levels; ++idx)
    {
       gcge_ops->VecDestroy((void**)&vec_tmp[idx], gcge_ops);
    }
    free(vec_tmp);
    gcge_ops->VecDestroy((void**)&vec_i, gcge_ops);
    gcge_ops->VecDestroy((void**)&vec_j, gcge_ops);

    /* 测试多向量延拓和限制 */
    level_i = 0; level_j = num_levels-1;
    PetscInt num_vecs = 3;
    BV multi_vec_i, multi_vec_j;
    gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_i), num_vecs, 
	  (void*)petsc_A_array[level_i], gcge_ops);
    gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_j), num_vecs, 
	  (void*)petsc_A_array[level_j], gcge_ops);
    gcge_ops->MultiVecSetRandomValue((void**)multi_vec_i, 0, num_vecs, gcge_ops);
    gcge_ops->MultiVecSetRandomValue((void**)multi_vec_j, 0, num_vecs, gcge_ops);

    PetscInt mv_s[2], mv_e[2];
    mv_s[0] = 0; mv_e[0] = num_vecs;
    mv_s[1] = 0; mv_e[1] = num_vecs;

    /* FromItoJ需要临时向量，存储每一层的多向量 */
    BV *multi_vec_tmp = malloc(num_levels*sizeof(BV));
    for (idx = 0; idx < num_levels; ++idx)
    {
       gcge_ops->MultiVecCreateByMat((void***)(&multi_vec_tmp[idx]), num_vecs, 
	     (void*)petsc_A_array[idx], gcge_ops);
    }
    gcge_ops->MultiVecFromItoJ((void**)petsc_P_array, level_i, level_j, 
	  (void**)multi_vec_i, (void**)multi_vec_j, (void***)multi_vec_tmp, mv_s, mv_e, gcge_ops);
    BVView(multi_vec_i, viewer);
    BVView(multi_vec_j, viewer);
    gcge_ops->MultiVecFromItoJ((void**)petsc_P_array, level_j, level_i, 
	  (void**)multi_vec_j, (void**)multi_vec_i, (void***)multi_vec_tmp, mv_s, mv_e, gcge_ops);
    BVView(multi_vec_i, viewer);
    BVView(multi_vec_j, viewer);
//    PETSCPrintBV(multi_vec_i, "Rfrom");
//    PETSCPrintBV(multi_vec_j, "Rto");
    for (idx = 0; idx < num_levels; ++idx)
    {
       gcge_ops->MultiVecDestroy((void***)(&multi_vec_tmp[idx]), num_vecs, gcge_ops);
    }
    free(multi_vec_tmp);
    gcge_ops->MultiVecDestroy((void***)(&multi_vec_i), num_vecs, gcge_ops);
    gcge_ops->MultiVecDestroy((void***)(&multi_vec_j), num_vecs, gcge_ops);
#endif

    gcge_ops->MultiGridDestroy((void***)&petsc_A_array, (void***)&petsc_B_array, (void***)&petsc_P_array,
	  &num_levels, gcge_ops);

    MatDestroy(&petsc_mat_A);
    MatDestroy(&petsc_mat_B);

    GCGE_OPS_Free(&gcge_ops);
    /* PetscFinalize */
    SlepcFinalize();
    return 0;
}

void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m)
{
    PetscInt N = n*m;
    PetscInt Istart, Iend, II, i, j;
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD,A);
    ierr = MatSetSizes(*A,PETSC_DECIDE,PETSC_DECIDE,N,N);
    ierr = MatSetFromOptions(*A);
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
    ierr = MatSetFromOptions(*B);
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
