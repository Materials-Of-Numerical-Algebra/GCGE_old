/*
 * =====================================================================================
 *
 *       Filename:  gcge_app_petsc.h
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

#ifndef _GCGE_APP_SLEPC_H_
#define _GCGE_APP_SLEPC_H_

#include "gcge_config.h"
#include "gcge_ops.h"
#include "gcge_solver.h"

//#include <petscoptions.h>
//#include <petsc/private/vecimpl.h>
//#include <petscblaslapack.h>
#include <petscoptions.h>
#include <petscviewer.h>
#include <petscsys.h>

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscpc.h>

#include <slepcbv.h>



void SLEPC_ReadMatrixBinary(Mat *A, const char *filename);
void SLEPC_LinearSolverCreate(KSP *ksp, Mat A, Mat T, char *eval_type);
void SLEPC_VecLocalInnerProd(Vec x, Vec y, double *value);

void GCGE_SLEPC_SetOps(GCGE_OPS *ops);
void GCGE_SOLVER_SetSLEPCOps(GCGE_SOLVER *solver);
void GCGE_OPS_CreateSLEPC(GCGE_OPS **ops);

GCGE_SOLVER *GCGE_SLEPC_Solver_Init(Mat A, Mat B, int num_eigenvalues, int argc, char* argv[]);
GCGE_SOLVER* GCGE_SLEPC_Solver_Init_KSPDefault(Mat A, Mat B, Mat P, int num_eigenvalues, int argc, char* argv[]);
GCGE_SOLVER* GCGE_SLEPC_Solver_Init_KSPGivenByUser(Mat A, Mat B, KSP ksp, int num_eigenvalues, int argc, char* argv[]);
//设置ksp为线性解法器
void GCGE_SOLVER_SetSLEPCOpsLinearSolver(GCGE_SOLVER *solver, KSP ksp);
#endif
