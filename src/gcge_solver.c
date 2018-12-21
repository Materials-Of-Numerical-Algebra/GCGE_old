/*
 * =====================================================================================
 *
 *       Filename:  gcge_solver.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时50分16秒
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
#include "gcge_solver.h"


void GCGE_SOLVER_Create(GCGE_SOLVER **solver)
{
    *solver = (GCGE_SOLVER*)malloc(sizeof(GCGE_SOLVER));
    GCGE_PARA_Create(&(*solver)->para);
    GCGE_OPS_Create(&(*solver)->ops);
    //GCGE_INT error = GCGE_PARA_SetFromCommandLine((*solver)->para, argc, argv);
    /* TODO should set NULL and Setup to modify */
    GCGE_WORKSPACE_Create(&(*solver)->workspace);
    (*solver)->A = NULL;
    (*solver)->B = NULL;
    (*solver)->eval = NULL;
    (*solver)->evec = NULL;
    //return error;
}
void GCGE_SOLVER_Free(GCGE_SOLVER **solver)
{
    GCGE_WORKSPACE_Free(&(*solver)->workspace, (*solver)->para, (*solver)->ops);
    GCGE_PARA_Free(&(*solver)->para);
    GCGE_OPS_Free(&(*solver)->ops);
    free(*solver); *solver = NULL;
}

void GCGE_SOLVER_Free_Some(GCGE_SOLVER **solver)
{
    GCGE_WORKSPACE_Free(&(*solver)->workspace, (*solver)->para, (*solver)->ops);
    GCGE_PARA_Free(&(*solver)->para);
    free(*solver); *solver = NULL;
}

//把solver以及其中的特征值和特征向量全部都释放掉
void GCGE_SOLVER_Free_All(GCGE_SOLVER **solver)
{    
    GCGE_INT nev; 
    nev = (*solver)->para->nev;
    (*solver)->ops->MultiVecDestroy(&((*solver)->evec), nev, (*solver)->ops);
    free((*solver)->eval); (*solver)->eval = NULL;
    GCGE_SOLVER_Free(solver);     
}
void GCGE_SOLVER_Setup(GCGE_SOLVER *solver)
{
    GCGE_PARA_Setup(solver->para);
    GCGE_OPS_Setup (solver->ops);
    GCGE_WORKSPACE_Setup(solver->workspace, solver->para, solver->ops, solver->A);
}

void GCGE_SOLVER_SetMatA(GCGE_SOLVER *solver, void *A)
{
    solver->A = A;
}
void GCGE_SOLVER_SetMatB(GCGE_SOLVER *solver, void *B)
{
    solver->B = B;
}
void GCGE_SOLVER_SetEigenvalues(GCGE_SOLVER *solver, GCGE_DOUBLE *eval)
{
    solver->eval = eval;
}
void GCGE_SOLVER_SetEigenvectors(GCGE_SOLVER *solver, void **evec)
{
    solver->evec = evec;
}
void GCGE_SOLVER_SetNumEigen(GCGE_SOLVER *solver, GCGE_INT nev)
{
   GCGE_PARA_SetNumEigen(solver->para, nev);
}

void GCGE_SOLVER_SetOpsLinearSolverWorkspace(GCGE_SOLVER *solver, void *linear_solver_workspace)
{
    GCGE_OPS_SetLinearSolverWorkspace(solver->ops, linear_solver_workspace);
}

void GCGE_SOLVER_GetEigenvalues(GCGE_SOLVER *solver, GCGE_DOUBLE **eval)
{
    *eval = solver->eval;
}
void GCGE_SOLVER_GetEigenvectors(GCGE_SOLVER *solver, void ***evec)
{
    *evec = solver->evec;
}

void GCGE_SOLVER_Solve(GCGE_SOLVER *solver)
{    
    GCGE_Solve(solver->A, solver->B, solver->eval, solver->evec, solver->para, solver->ops, solver->workspace);
}
