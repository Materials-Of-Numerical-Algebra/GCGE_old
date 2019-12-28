
/*
 * PASE_Solver_Init:
 * 
 *                                  PASE_Convert
 *                                       +
 *                                   PASE_AMG
 *                                       +
 * PASE_Solver:1 APP--->GCGE_OPS--->PASE_MultiGrid--->PASE_MatVec
 *                                                        +
 *                                                   GCGE_OPS--->PASE_OPS--->GCGE_OPS
 *                                                                   |
 *                                                          GCGE_Solver_PASE_Init
 *
 *
 *             2 APP--->GCGE_OPS
 *                         |
 *                 GCGE_Solver_APP_Init
 *
 *             3 PASE_LinearSolver<---PASE_MultiGrid   
 * */

#ifndef  _PASE_H_
#define  _PASE_H_

#include "pase_convert.h"

#include "pase_config.h"

#include "pase_param.h"

#include "pase_matvec.h"
#include "pase_ops.h"

#include "pase_mg.h"
#include "pase_amg.h"

#include "pase_solver.h"
#endif
