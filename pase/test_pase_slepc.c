/*
 * =====================================================================================
 *
 *       Filename:  test_solver.c
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

//#include "memwatch.h"
#include <petscsys.h>
#include <petscviewer.h>
#include <petscmat.h>

#include "gcge.h"
#include "gcge_app_slepc.h"

static char help[] = "Use GCGE-SLEPc to solve an eigensystem Ax=kBx with the matrixes loaded from files.\n";

void PrintParameter(PASE_PARAMETER param);
void PASEGetCommandLineInfo(PASE_INT argc, char *argv[], PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth);
void PetscGetDifferenceMatrix(Mat *A, PetscInt n, PetscInt m);
int main(int argc, char* argv[])
{
    PetscErrorCode ierr;
    SlepcInitialize(&argc,&argv,(char*)0,help);
    
    GCGE_DOUBLE t_start = GCGE_GetTime();

    //从参数中读入矩阵A, B, P的地址, 矩阵P为求解线性方程组时的预条件矩阵
    Mat A, B, P;
    char file_A[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat_A", file_A, sizeof(file_A), NULL);
    char file_B[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat_B", file_B, sizeof(file_B), NULL);
    char file_P[PETSC_MAX_PATH_LEN] = "fileinput";
    ierr = PetscOptionsGetString(NULL, NULL, "-mat_P", file_P, sizeof(file_P), NULL);

    PetscInt  n = 10, m = 10;
    PetscBool flag;
    //读入矩阵, 如果命令行提供了矩阵B和P的地址, 才读矩阵B和P
    if(strcmp(file_A, "fileinput") != 0)
    {
        SLEPC_ReadMatrixBinary(&A, file_A);
    }
    else
    {
        ierr = PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
        ierr = PetscOptionsGetInt(NULL, NULL, "-m", &m, &flag);
        if (!flag) m=n;
        PetscGetDifferenceMatrix(&A, n, m);
    }
    if(strcmp(file_B, "fileinput") != 0)
    {
        SLEPC_ReadMatrixBinary(&B, file_B);
    }
    if(strcmp(file_P, "fileinput") != 0)
    {
        SLEPC_ReadMatrixBinary(&P, file_P);
    }

    //创建petsc_solver
    GCGE_SOLVER *slepc_solver;
    GCGE_SOLVER_Create(&slepc_solver);
    //设置SLEPC结构的矩阵向量操作
    GCGE_SOLVER_SetSLEPCOps(slepc_solver);

    //设置一些参数-示例
    int nev = 30;
    GCGE_SOLVER_SetNumEigen(slepc_solver, nev);//设置特征值个数
    slepc_solver->para->print_eval = 0;//设置是否打印每次迭代的特征值
    //暂时用GCGE获取初值，残差1e-4
    slepc_solver->para->ev_tol = 1e-4;

    //从命令行读入GCGE_PARA中的一些参数
    GCGE_INT error = GCGE_PARA_SetFromCommandLine(slepc_solver->para, argc, argv);

    //给特征值和特征向量分配nev大小的空间
    nev = slepc_solver->para->nev;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    BV evec;
    slepc_solver->ops->MultiVecCreateByMat((void***)(&evec), nev, (void*)A, slepc_solver->ops);

    //将矩阵与特征对设置到petsc_solver中
    GCGE_SOLVER_SetMatA(slepc_solver, A);
    if(strcmp(file_B, "fileinput") != 0)
    {
        GCGE_SOLVER_SetMatB(slepc_solver, B);
    }
    GCGE_SOLVER_SetEigenvalues(slepc_solver, eval);
    GCGE_SOLVER_SetEigenvectors(slepc_solver, (void**)evec);

    KSP ksp;
    //如果读入了预条件矩阵, 就使用PETSc的求解器
    if(strcmp(file_P, "fileinput") != 0)
    {
        //设定线性求解器
        SLEPC_LinearSolverCreate(&ksp, A, P);
        //PetscViewer viewer;
        //ierr = KSPView(ksp, viewer);
        //给slepc_solver设置KSP为线性求解器
        GCGE_SOLVER_SetSLEPCOpsLinearSolver(slepc_solver, ksp);
    }

    //对slepc_solver进行setup，检查参数，分配工作空间等
    GCGE_SOLVER_Setup(slepc_solver);
    //求解
    GCGE_SOLVER_Solve(slepc_solver);

    if(strcmp(file_P, "fileinput") != 0)
    {
        KSPDestroy(&ksp);
    }

    //释放petsc_solver中非用户创建的空间
    GCGE_SOLVER_Free(&slepc_solver);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_SLEPC_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);

    //PASE----------------------------------------------------
    PASE_PARAMETER param   = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
    param->cycle_type      = 0;   //二网格
    param->block_size      = nev; //特征值个数
    param->max_cycle       = 50;  //二网格迭代次数
    param->max_pre_iter    = 1;   //前光滑次数
    param->max_post_iter   = 1;   //后光滑次数
    param->atol            = 1e-8;
    param->rtol            = 1e-6;
    param->print_level     = 1;
    param->max_level       = 6;   //AMG层数
    PASEGetCommandLineInfo(argc, argv, &(param->block_size), &(param->atol), &(param->max_pre_iter));
    param->min_coarse_size = param->block_size * 30; //最粗层网格最少有30*nev维
    //param->min_coarse_size = 500;
    PrintParameter(param);
    PASE_EigenSolver(A, B, eval, (void**)evec, nev, param, gcge_ops);
    //PASE----------------------------------------------------

    //释放特征值、特征向量、KSP空间
    free(eval); eval = NULL;
    slepc_solver->ops->MultiVecDestroy((void***)(&evec), nev, slepc_solver->ops);

    //释放矩阵空间
    ierr = MatDestroy(&A);
    if(strcmp(file_B, "fileinput") != 0)
    {
        ierr = MatDestroy(&B);
    }
    if(strcmp(file_P, "fileinput") != 0)
    {
        ierr = MatDestroy(&P);
    }

    GCGE_DOUBLE t_end = GCGE_GetTime();
    GCGE_Printf("From ReadMatrix To MatDestroy, Total Time: %f\n", t_end - t_start);

    ierr = SlepcFinalize();

    return ierr;
}

void PetscGetDifferenceMatrix(Mat *A, PetscInt n, PetscInt m)
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
}

void PASEGetCommandLineInfo(PASE_INT argc, char *argv[], PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth)
{
  PASE_INT arg_index = 0;
  PASE_INT print_usage = 0;
  PASE_INT myid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
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
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Usage: %s [<options>]\n", argv[0]);
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "  -n <n>              : problem size in each direction (default: 33)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -block_size <n>      : eigenproblem block size (default: 3)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -max_levels <n>      : max levels of AMG (default: 5)\n");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    exit(-1);
  }
}

void PrintParameter(PASE_PARAMETER param)
{
    PASE_Printf(MPI_COMM_WORLD, "PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
    PASE_Printf(MPI_COMM_WORLD, "Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n" );
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Set parameters:\n");
    PASE_Printf(MPI_COMM_WORLD, "block size      = %d\n", param->block_size);
    PASE_Printf(MPI_COMM_WORLD, "max pre iter    = %d\n", param->max_pre_iter);
    PASE_Printf(MPI_COMM_WORLD, "atol            = %e\n", param->atol);
    PASE_Printf(MPI_COMM_WORLD, "max cycle       = %d\n", param->max_cycle);
    PASE_Printf(MPI_COMM_WORLD, "min coarse size = %d\n", param->min_coarse_size);
    PASE_Printf(MPI_COMM_WORLD, "\n");
}

