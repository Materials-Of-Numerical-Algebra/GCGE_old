/*
 * =====================================================================================
 *
 *       Filename:  test_mg.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年12月26日 15时50分13秒
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include "pase_amg.h"
#include "gcge.h"
#include "gcge_cg.h"
#include "pase.h"
#include "gcge_app_slepc.h"


static char help[] = "Test MultiGrid.\n";
void GetPetscMat(Mat *A, Mat *B, PetscInt n, PetscInt m);
void PETSCPrintMat(Mat A, char *name);
void PETSCPrintVec(Vec x);
void PETSCPrintBV(BV x, char *name);
void PrintParameter(PASE_PARAMETER param);
void PASEGetCommandLineInfo(PASE_INT argc, char *argv[], PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth);
void PASE_BMG_TEST( PASE_MULTIGRID mg, 
               PASE_INT current_level, 
               void **rhs, void **sol, 
               PASE_INT *start, PASE_INT *end,
               PASE_REAL tol, PASE_REAL rate, 
               PASE_INT nsmooth, PASE_INT max_coarest_nsmooth);
/* 
 *  Description:  测试PASE_MULTIGRID
 */
int
main ( int argc, char *argv[] )
{
    /* SlepcInitialize */
    SlepcInitialize(&argc,&argv,(char*)0,help);
    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscViewer    viewer;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    /* 得到细网格矩阵 */
    Mat      A, B;
    PetscInt n = 30, m = 30;
    GetPetscMat(&A, &B, n, m);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_CreateSLEPC(&gcge_ops);
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);

    //进行multigrid分层
    int num_levels = 3;
    PASE_MULTIGRID multi_grid;
    int error = PASE_MULTIGRID_Create(&multi_grid, num_levels, 
            (void *)A, (void *)B, gcge_ops, pase_ops);

    //申请工作空间
    int nev = 5;
    BV *u, *rhs, *u_tmp, *u_tmp_1, *u_tmp_2;
    u       = (BV*)malloc(num_levels*sizeof(BV));
    rhs     = (BV*)malloc(num_levels*sizeof(BV));
    u_tmp   = (BV*)malloc(num_levels*sizeof(BV));
    u_tmp_1 = (BV*)malloc(num_levels*sizeof(BV));
    u_tmp_2 = (BV*)malloc(num_levels*sizeof(BV));
    int i = 0;
    for(i=0; i<num_levels; i++)
    {
        gcge_ops->MultiVecCreateByMat((void***)(&(u[i])), nev, 
	      multi_grid->A_array[i], gcge_ops);
        gcge_ops->MultiVecCreateByMat((void***)(&(rhs[i])), nev, 
	      multi_grid->A_array[i], gcge_ops);
        gcge_ops->MultiVecCreateByMat((void***)(&(u_tmp[i])), nev, 
	      multi_grid->A_array[i], gcge_ops);
        gcge_ops->MultiVecCreateByMat((void***)(&(u_tmp_1[i])), nev, 
	      multi_grid->A_array[i], gcge_ops);
        gcge_ops->MultiVecCreateByMat((void***)(&(u_tmp_2[i])), nev, 
	      multi_grid->A_array[i], gcge_ops);
    }
    double *double_tmp = (double*)calloc(nev*nev, sizeof(double));
    int    *int_tmp    = (int*)calloc(nev*nev, sizeof(int));
    multi_grid->u          = (void***)u;
    multi_grid->rhs        = (void***)rhs;
    multi_grid->u_tmp      = (void***)u_tmp;
    multi_grid->u_tmp_1    = (void***)u_tmp_1;
    multi_grid->u_tmp_2    = (void***)u_tmp_2;
    multi_grid->double_tmp = double_tmp;
    multi_grid->int_tmp    = int_tmp;

    gcge_ops->MultiVecSetRandomValue((void**)(u[0]), 0, nev, gcge_ops);
    int mv_s[2];
    int mv_e[2];
    mv_s[0] = 0;
    mv_e[0] = nev;
    mv_s[1] = 0;
    mv_e[1] = nev;
    gcge_ops->MatDotMultiVec(A, (void**)(u[0]), (void**)(rhs[0]), 
	  mv_s, mv_e, gcge_ops);
    gcge_ops->MultiVecAxpby(0.0, (void**)(u[0]), 0.0, (void**)(u[0]), 
	  mv_s, mv_e, gcge_ops);

    //计算初始残差
    gcge_ops->MatDotMultiVec(A, (void**)(u[0]), (void**)(u_tmp[0]), mv_s, mv_e, gcge_ops);
    gcge_ops->MultiVecAxpby(1.0, (void**)(rhs[0]), -1.0, (void**)(u_tmp[0]), 
	  mv_s, mv_e, gcge_ops);
    gcge_ops->MultiVecInnerProd((void**)(u_tmp[0]), (void**)(u_tmp[0]), 
	  double_tmp, "nonsym", mv_s, mv_e, nev, 0, gcge_ops);
    for(i=0; i<nev; i++) {
        GCGE_Printf("init residual[%d] = %e\n", i, double_tmp[i*nev+i]);
    }

    double tol = 1e-8;
    double rate = 1e-2;
    int nsmooth = 10;
    int max_coarest_nsmooth = 50;
    //PASE_BMG_TEST(multi_grid, 0, (void**)(rhs[0]), (void**)(u[0]), mv_s, mv_e, 
    PASE_BMG(multi_grid, 0, (void**)(rhs[0]), (void**)(u[0]), mv_s, mv_e, 
    	  tol, rate, nsmooth, max_coarest_nsmooth);
    //GCGE_BCG(A, (void**)(rhs[0]), (void**)(u[0]), 0, nev, 2*nsmooth, tol,
    //      gcge_ops, (void**)(u_tmp[0]), (void**)(u_tmp_1[0]), 
    //	    (void**)(u_tmp_2[0]), double_tmp, int_tmp);

    //计算最后的残差
    gcge_ops->MatDotMultiVec(A, (void**)(u[0]), (void**)(u_tmp[0]), mv_s, mv_e, gcge_ops);
    gcge_ops->MultiVecAxpby(1.0, (void**)(rhs[0]), -1.0, (void**)(u_tmp[0]), 
	  mv_s, mv_e, gcge_ops);
    //PETSCPrintBV(u_tmp[0], "resi after");
    gcge_ops->MultiVecInnerProd((void**)(u_tmp[0]), (void**)(u_tmp[0]), 
	  double_tmp, "nonsym", mv_s, mv_e, nev, 0, gcge_ops);
    for(i=0; i<nev; i++) {
        GCGE_Printf("final residual[%d] = %e\n", i, double_tmp[i*nev+i]);
    }

    error = PASE_MULTIGRID_Destroy(&multi_grid);

    //释放空间
    for(i=0; i<num_levels; i++)
    {
        gcge_ops->MultiVecDestroy((void***)(&(u[i])), nev, gcge_ops);
        gcge_ops->MultiVecDestroy((void***)(&(rhs[i])), nev, gcge_ops);
        gcge_ops->MultiVecDestroy((void***)(&(u_tmp[i])), nev, gcge_ops);
        gcge_ops->MultiVecDestroy((void***)(&(u_tmp_1[i])), nev, gcge_ops);
        gcge_ops->MultiVecDestroy((void***)(&(u_tmp_2[i])), nev, gcge_ops);
    }
    free(u); u = NULL;
    free(rhs); rhs = NULL;
    free(u_tmp); u_tmp = NULL;
    free(u_tmp_1); u_tmp_1 = NULL;
    free(u_tmp_2); u_tmp_2 = NULL;

    free(double_tmp); double_tmp = NULL;
    free(int_tmp);    int_tmp = NULL;
    GCGE_OPS_Free(&gcge_ops);
    PASE_OPS_Free(&pase_ops); 
    ierr = MatDestroy(&A);
    ierr = MatDestroy(&B);

    /* PetscFinalize */
    ierr = SlepcFinalize();
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
    for(i=0; i<3; i++)
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

void PASEGetCommandLineInfo(PASE_INT argc, char *argv[], PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth)
{
  PASE_INT arg_index = 0;
  PASE_INT print_usage = 0;

  while (arg_index < argc)
  {
    if ( strcmp(argv[arg_index], "-block_size") == 0 )
    {
      arg_index++;
      *block_size = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-atol") == 0 )
    {
      arg_index++;
      *atol= pow(10, atoi(argv[arg_index++]));
    }
    else if ( strcmp(argv[arg_index], "-nsmooth") == 0 )
    {
      arg_index++;
      *nsmooth= atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-help") == 0 )
    {
      print_usage = 1;
      break;
    }
    else
    {
      arg_index++;
    }
  }

  if(print_usage)
  {
    GCGE_Printf("\n");
    GCGE_Printf("Usage: %s [<options>]\n", argv[0]);
    GCGE_Printf("\n");
    GCGE_Printf("  -n <n>              : problem size in each direction (default: 33)\n");
    GCGE_Printf("  -block_size <n>      : eigenproblem block size (default: 3)\n");
    GCGE_Printf("  -max_levels <n>      : max levels of AMG (default: 5)\n");
    GCGE_Printf("\n");
    exit(-1);
  }
}

void PrintParameter(PASE_PARAMETER param)
{
    GCGE_Printf("PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
    GCGE_Printf("Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
    GCGE_Printf("=============================================================\n" );
    GCGE_Printf("\n");
    GCGE_Printf("Set parameters:\n");
    GCGE_Printf("block size      = %d\n", param->block_size);
    GCGE_Printf("max pre iter    = %d\n", param->max_pre_iter);
    GCGE_Printf("atol            = %e\n", param->atol);
    GCGE_Printf("max cycle       = %d\n", param->max_cycle);
    GCGE_Printf("min coarse size = %d\n", param->min_coarse_size);
    GCGE_Printf("\n");
}


void PASE_BMG_TEST( PASE_MULTIGRID mg, 
               PASE_INT current_level, 
               void **rhs, void **sol, 
               PASE_INT *start, PASE_INT *end,
               PASE_REAL tol, PASE_REAL rate, 
               PASE_INT nsmooth, PASE_INT max_coarest_nsmooth)
{
    PASE_INT nlevel = mg->num_levels;
    //默认0层为最细层
    PASE_INT indicator = 1;
    // obtain the coarsest level
    PASE_INT coarest_level;
    if( indicator > 0 )
        coarest_level = nlevel-1;
    else
        coarest_level = 0;
    //设置最粗层上精确求解的精度
    PASE_REAL coarest_rate = 1e-8;
    void *A;
    PASE_INT mv_s[2];
    PASE_INT mv_e[2];
    void **residual = mg->u_tmp[current_level];
    // obtain the 'enough' accurate solution on the coarest level
    //direct solving the linear equation
    if( current_level == coarest_level )
    {
        //最粗层？？？？？？？
        A = mg->A_array[coarest_level];
        GCGE_BCG(A, rhs, sol, start[1], end[1]-start[1], 
                max_coarest_nsmooth, coarest_rate, mg->gcge_ops, 
                mg->u_tmp[coarest_level], mg->u_tmp_1[coarest_level], 
                mg->u_tmp_2[coarest_level], 
                mg->double_tmp, mg->int_tmp);
    }
    else
    {   
        A = mg->A_array[current_level];
        GCGE_BCG(A, rhs, sol, start[1], end[1]-start[1], 
                nsmooth, rate, mg->gcge_ops, 
                mg->u_tmp[current_level], mg->u_tmp_1[current_level], 
		mg->u_tmp_2[current_level], 
                mg->double_tmp, mg->int_tmp);

        mv_s[0] = start[1];
        mv_e[0] = end[1];
        mv_s[1] = 0;
        mv_e[1] = end[1]-start[1];
        //计算residual = A*sol
        mg->gcge_ops->MatDotMultiVec(A, sol, residual, mv_s, mv_e, mg->gcge_ops);
        //计算residual = rhs-A*sol
        mv_s[0] = 0;
        mv_e[0] = end[1]-start[1];
        mv_s[1] = 0;
        mv_e[1] = end[1]-start[1];
        mg->gcge_ops->MultiVecAxpby(1.0, rhs, -1.0, residual, mv_s, mv_e, mg->gcge_ops);

        // 把 residual 投影到粗网格
        PASE_INT coarse_level = current_level + indicator;
        void **coarse_residual = mg->rhs[coarse_level];
        mv_s[0] = 0;
        mv_e[0] = end[1]-start[1];
        mv_s[1] = 0;
        mv_e[1] = end[1]-start[1];
        PASE_INT error = PASE_MULTIGRID_FromItoJ(mg, current_level, coarse_level, 
                mv_s, mv_e, residual, coarse_residual);
        /* TODO coarse_sol????? */

        //求粗网格解问题，利用递归
        void **coarse_sol = mg->u[coarse_level];
        mv_s[0] = 0;
        mv_e[0] = end[1]-start[1];
        mv_s[1] = 0;
        mv_e[1] = end[1]-start[1];
	//先给coarse_sol赋初值0
	mg->gcge_ops->MultiVecAxpby(0.0, coarse_sol, 0.0, coarse_sol, 
	        mv_s, mv_e, mg->gcge_ops);
        //PASE_BMG(mg, coarse_level, coarse_residual, coarse_sol, 
        PASE_BMG_TEST(mg, coarse_level, coarse_residual, coarse_sol, 
                mv_s, mv_e, tol, rate, nsmooth, max_coarest_nsmooth);
        
        // 把粗网格上的解插值到细网格，再加到前光滑得到的近似解上
        // 可以用 residual 代替
        mv_s[0] = 0;
        mv_e[0] = end[1]-start[1];
        mv_s[1] = 0;
        mv_e[1] = end[1]-start[1];
        error = PASE_MULTIGRID_FromItoJ(mg, coarse_level, current_level, 
                mv_s, mv_e, coarse_sol, residual);
        //计算residual = rhs-A*sol
        mv_s[0] = 0;
        mv_e[0] = end[1]-start[1];
        mv_s[1] = start[1];
        mv_e[1] = end[1];
        mg->gcge_ops->MultiVecAxpby(1.0, residual, 1.0, sol, mv_s, mv_e, mg->gcge_ops);
        
	//后光滑
        GCGE_BCG(A, rhs, sol, start[1], end[1]-start[1], 
                nsmooth, rate, mg->gcge_ops, 
                mg->u_tmp[current_level], mg->u_tmp_1[current_level], 
		mg->u_tmp_2[current_level], 
                mg->double_tmp, mg->int_tmp);
    }//end for (if current_level)
}
