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
#include "gcge.h"
#include "pase.h"
#include "gcge_app_slepc.h"

/* 这两个函数在get_mat_phg.c中 */
int CreateMatrixPHG (void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);
int DestroyMatrixPHG(void **matA, void **matB, void **dofU, void **mapM, void **gridG, int argc, char *argv[]);

static char help[] = "Test MultiGrid.\n";
void PETSCPrintMat(Mat A, char *name);
void PETSCPrintVec(Vec x);
void PETSCPrintBV (BV  x, char *name);

void Multigrid_LinearSolverCreate(PASE_MULTIGRID *multi_grid, GCGE_OPS *gcge_ops, 
      int num_vecs, void *A, void *B);
void GCGE_SOLVER_SetMultigridLinearSolver(GCGE_SOLVER *solver, KSP ksp);
void GCGE_MultiGrid_LinearSolver(void *Matrix, void **b, void **x, struct GCGE_OPS_ *ops);
typedef struct gcge_MultiGrid_struct 
{
  PASE_MULTIGRID *pase_multigrid; 
  void **sol;
  void **rhs;
  int *start;
  int *end;
} gcge_MultiGrid;
typedef struct gcge_MultiGrid_struct *GCGE_MULTIGRID;

/* 
 *  Description:  测试PASE_MULTIGRID
 */

int
main ( int argc, char *argv[] )
{
    /* PetscInitialize */
    SlepcInitialize(&argc,&argv,(char*)0,help);
    PetscErrorCode ierr;

    /* 测试矩阵声明 */
    Mat      petsc_mat_A, petsc_mat_B;

    /* 得到一个PHG矩阵, 并将之转换为PETSC矩阵 */
    void *phg_mat_A, *phg_mat_B, *phg_dof_U, *phg_map_M, *phg_grid_G;
    CreateMatrixPHG (&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);
    MatrixConvertPHG2PETSC((void **)(&petsc_mat_A), &phg_mat_A);
    MatrixConvertPHG2PETSC((void **)(&petsc_mat_B), &phg_mat_B);
    DestroyMatrixPHG(&phg_mat_A, &phg_mat_B, &phg_dof_U, &phg_map_M, &phg_grid_G, argc, argv);

    //创建petsc_solver
    GCGE_SOLVER *slepc_solver;
    GCGE_SOLVER_Create(&slepc_solver);

    //从命令行读入GCGE_PARA中的一些参数
    GCGE_INT error = GCGE_PARA_SetFromCommandLine(slepc_solver->para, argc, argv);
    //设置SLEPC结构的矩阵向量操作
    GCGE_SOLVER_SetSLEPCOps(slepc_solver);
    //设置一些参数-示例
    int nev = 6;
    GCGE_SOLVER_SetNumEigen(slepc_solver, nev);//设置特征值个数
    slepc_solver->para->print_eval = 0;//设置是否打印每次迭代的特征值
    double *eval = (double *)calloc(nev, sizeof(double)); 
    BV evec;
    slepc_solver->ops->MultiVecCreateByMat((void***)(&evec), nev, (void*)petsc_mat_A, slepc_solver->ops);

    //将矩阵与特征对设置到petsc_solver中
    GCGE_SOLVER_SetMatA(slepc_solver, petsc_mat_A);
    GCGE_SOLVER_SetMatB(slepc_solver, petsc_mat_B);
    GCGE_SOLVER_SetEigenvalues(slepc_solver, eval);
    GCGE_SOLVER_SetEigenvectors(slepc_solver, (void**)evec);

    KSP ksp;
    //设定线性求解器
    SLEPC_LinearSolverCreate(&ksp, petsc_mat_A, petsc_mat_A);
    //给slepc_solver设置KSP为线性求解器
    GCGE_SOLVER_SetSLEPCOpsLinearSolver(slepc_solver, ksp);

    //对slepc_solver进行setup，检查参数，分配工作空间等
    GCGE_SOLVER_Setup(slepc_solver);
    //求解
    GCGE_SOLVER_Solve(slepc_solver);

    //释放特征值、特征向量、KSP空间
    free(eval); eval = NULL;
    slepc_solver->ops->MultiVecDestroy((void***)(&evec), nev, slepc_solver->ops);

    KSPDestroy(&ksp);
    //释放petsc_solver中非用户创建的空间
    GCGE_SOLVER_Free(&slepc_solver);

    ierr = MatDestroy(&petsc_mat_A);
    ierr = MatDestroy(&petsc_mat_B);

    /* PetscFinalize */
    ierr = SlepcFinalize();
    return 0;
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
