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

#include "gcge.h"
#include "gcge_app_pase.h"
#include "gcge_app_csr.h"
#include "pase.h"
#include "pase_mg.h"
#include "pase_solver.h"

static char help[] = "Use GCGE-SLEPc to solve an eigensystem Ax=kBx with the matrixes loaded from files.\n";

void PASE_PrintMat(void *mat);//PASE_Matrix pase_matrix)
void PrintParameter(PASE_PARAMETER param);
void PASEGetCommandLineInfo(PASE_INT argc, char *argv[], PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth);
int main(int argc, char* argv[])
{
    
    srand(1);
    GCGE_DOUBLE t_start = GCGE_GetTime();

    const char *file_A = "/home/zhangning/MATRIX/fem_csr_mat/Stiff_1089.txt";
    const char *file_B = "/home/zhangning/MATRIX/fem_csr_mat/Mass_1089.txt";
    CSR_MAT *A = CSR_ReadMatFile(file_A);
    CSR_MAT *B = CSR_ReadMatFile(file_B);

    //创建petsc_solver
    GCGE_SOLVER *csr_solver;
    GCGE_SOLVER_Create(&csr_solver);
    //设置SLEPC结构的矩阵向量操作
    GCGE_SOLVER_SetCSROps(csr_solver);
    int error = GCGE_OPS_Setup(csr_solver->ops);

    //设置一些参数-示例
    int nev = 5;
    GCGE_SOLVER_SetNumEigen(csr_solver, nev);//设置特征值个数
    csr_solver->para->print_eval = 0;//设置是否打印每次迭代的特征值
    csr_solver->para->ev_max_it  = 20;
    csr_solver->para->print_result = 0;
    csr_solver->para->print_para = 0;
    csr_solver->para->conv_type = "A";
    //暂时用GCGE获取初值，残差1e-4
    csr_solver->para->ev_tol = 1e-2;

    //从命令行读入GCGE_PARA中的一些参数
    error = GCGE_PARA_SetFromCommandLine(csr_solver->para, argc, argv);

    //给特征值和特征向量分配nev大小的空间
    nev = csr_solver->para->nev;
    double *eval = (double *)calloc(nev, sizeof(double)); 
    CSR_VEC **evec;
    csr_solver->ops->MultiVecCreateByMat((void***)(&evec), nev, (void*)A, csr_solver->ops);

    //将矩阵与特征对设置到petsc_solver中
    GCGE_SOLVER_SetMatA(csr_solver, (void*)A);
    GCGE_SOLVER_SetMatB(csr_solver, (void*)B);
    GCGE_SOLVER_SetEigenvalues(csr_solver, eval);
    GCGE_SOLVER_SetEigenvectors(csr_solver, (void**)evec);

    //对csr_solver进行setup，检查参数，分配工作空间等
    GCGE_SOLVER_Setup(csr_solver);
    //求解
    GCGE_SOLVER_Solve(csr_solver);

    //释放petsc_solver中非用户创建的空间
    GCGE_SOLVER_Free(&csr_solver);

    //创建gcge_ops
    GCGE_OPS *gcge_ops;
    GCGE_OPS_Create(&gcge_ops);
    GCGE_CSR_SetOps(gcge_ops);
    GCGE_OPS_Setup(gcge_ops);

    //PASE----------------------------------------------------
    PASE_PARAMETER param   = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
    param->cycle_type      = 0;   //二网格
    param->block_size      = nev; //特征值个数
    param->max_cycle       = 2;  //二网格迭代次数
    param->max_pre_iter    = 1;   //前光滑次数
    param->max_post_iter   = 1;   //后光滑次数
    param->atol            = 1e-10;
    param->rtol            = 1e-12;
    param->print_level     = 1;
    param->max_level       = 3;   //AMG层数
    PASEGetCommandLineInfo(argc, argv, &(param->block_size), &(param->atol), &(param->max_pre_iter));
    param->min_coarse_size = param->block_size * 30; //最粗层网格最少有30*nev维
    //param->min_coarse_size = 500;
    PrintParameter(param);

    //用gcge_ops创建pase_ops
    PASE_OPS *pase_ops;
    PASE_OPS_Create(&pase_ops, gcge_ops);
    pase_ops->PrintMat = PASE_PrintMat;

    PASE_MULTIGRID multigrid;
    error = PASE_MULTIGRID_Create(&multigrid, param->max_level, A, B, 
            gcge_ops, pase_ops);

    const char *file_Ac = "/home/zhangning/MATRIX/fem_csr_mat/Stiff_289.txt";
    const char *file_Bc = "/home/zhangning/MATRIX/fem_csr_mat/Mass_289.txt";
    CSR_MAT *A1 = (void*)CSR_ReadMatFile(file_Ac);
    CSR_MAT *B1 = (void*)CSR_ReadMatFile(file_Bc);
    multigrid->A_array[1] = (void*)A1;
    multigrid->B_array[1] = (void*)B1;
    const char *file_Ac2 = "/home/zhangning/MATRIX/fem_csr_mat/Stiff_81.txt";
    const char *file_Bc2 = "/home/zhangning/MATRIX/fem_csr_mat/Mass_81.txt";
    CSR_MAT *A2 = (void*)CSR_ReadMatFile(file_Ac2);
    CSR_MAT *B2 = (void*)CSR_ReadMatFile(file_Bc2);
    multigrid->A_array[2] = (void*)A2;
    multigrid->B_array[2] = (void*)B2;
    const char *file_P = "/home/zhangning/MATRIX/fem_csr_mat/Prolong_289_1089.txt";
    const char *file_R = "/home/zhangning/MATRIX/fem_csr_mat/Restrict_289_1089.txt";
    CSR_MAT *P1 = (void*)CSR_ReadMatFile(file_P);
    CSR_MAT *R1 = (void*)CSR_ReadMatFile(file_R);
    multigrid->P_array[1] = (void*)P1;
    multigrid->R_array[1] = (void*)R1;
    const char *file_P2 = "/home/zhangning/MATRIX/fem_csr_mat/Prolong_289_81.txt";
    const char *file_R2 = "/home/zhangning/MATRIX/fem_csr_mat/Restrict_289_81.txt";
    CSR_MAT *P2 = (void*)CSR_ReadMatFile(file_P2);
    CSR_MAT *R2 = (void*)CSR_ReadMatFile(file_R2);
    multigrid->P_array[2] = (void*)P2;
    multigrid->R_array[2] = (void*)R2;

    GCGE_Printf("line 145\n");
    PASE_EigenSolver((void*)A, (void*)B, eval, (void**)evec, nev, param, 
            gcge_ops, pase_ops, multigrid);

    PASE_OPS_Free(&pase_ops); 
    free(param); param = NULL;
    //PASE----------------------------------------------------

    //释放特征值、特征向量、KSP空间
    free(eval); eval = NULL;
    gcge_ops->MultiVecDestroy((void***)(&evec), nev, gcge_ops);
    GCGE_OPS_Free(&gcge_ops);
    CSR_MatFree(&A);
    CSR_MatFree(&B);
    CSR_MatFree(&A1);
    CSR_MatFree(&B1);
    CSR_MatFree(&P1);
    CSR_MatFree(&R1);
    CSR_MatFree(&A2);
    CSR_MatFree(&B2);
    CSR_MatFree(&P2);
    CSR_MatFree(&R2);

    GCGE_DOUBLE t_end = GCGE_GetTime();
    GCGE_Printf("From ReadMatrix To MatDestroy, Total Time: %f\n", t_end - t_start);

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

void PASE_PrintMat(void *mat)//PASE_Matrix pase_matrix)
{
    PASE_Matrix pase_matrix = (PASE_Matrix)mat;
    CSR_MAT *A_H = (CSR_MAT*)(pase_matrix->A_H);
    int     num_aux_vec = pase_matrix->num_aux_vec;
    int     nrows = A_H->N_Rows;
    int     ldd = nrows+num_aux_vec;
    double  *dense_mat = (double*)calloc(ldd*ldd, sizeof(double));
    int     row = 0;
    int     col = 0;
    int     i   = 0;
    int     j   = 0;
    int     *rowptr  = A_H->RowPtr;
    int     *kcol    = A_H->KCol;
    double  *entries = A_H->Entries;
    //将A_H部分赋值给dense_mat
    for(row=0; row<nrows; row++)
    {
        for(i=rowptr[row]; i<rowptr[row+1]; i++)
        {
            col = kcol[i];
            dense_mat[col*ldd+row] = entries[i];
        }
    }
    //将aux_Hh部分赋值给dense_mat
    CSR_VEC **vecs = (CSR_VEC**)(pase_matrix->aux_Hh);
    for(i=0; i<num_aux_vec; i++)
    {
        for(row=0; row<nrows; row++)
        {
            col = nrows+i;
            dense_mat[col*ldd+row] = vecs[i]->Entries[row];
            //对称化
            dense_mat[row*ldd+col] = vecs[i]->Entries[row];
        }
    }
    //aux_hh部分
    for(i=0; i<num_aux_vec; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            col = nrows+i;
            row = nrows+j;
            dense_mat[col*ldd+row] = pase_matrix->aux_hh[i*num_aux_vec+j];
        }
    }
    //打印稠密矩阵
    printf("pase_matrix = [\n");
    for(i=0; i<ldd; i++)
    {
        for(j=0; j<ldd; j++)
        {
            printf("%18.15f\t", dense_mat[i*ldd+j]);
        }
        printf("\n");
    }
    printf("];\n");
    free(dense_mat); dense_mat = NULL;
}
