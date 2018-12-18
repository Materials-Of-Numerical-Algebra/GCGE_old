/*
 * =====================================================================================
 *
 *       Filename:  gcge_app_csr.h
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

#ifndef _GCGE_APP_CSR_H_
#define _GCGE_APP_CSR_H_

#include "gcge_config.h"
#include "gcge_ops.h"
#include "gcge_solver.h"
#include "gcge_app_csr.h"
#include "pase.h"



void GCGE_PASE_SetOps(GCGE_OPS *ops, PASE_OPS *pase_ops);

void GCGE_SOLVER_SetPASEOps(GCGE_SOLVER *solver, GCGE_OPS *ops);

GCGE_SOLVER* GCGE_PASE_Solver_Init(PASE_Matrix A, PASE_Matrix B, int num_eigenvalues, int argc, char* argv[], GCGE_OPS *ops);




#endif
