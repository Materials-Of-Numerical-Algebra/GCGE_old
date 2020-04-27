#ifndef _PASE_AMG_H_
#define _PASE_AMG_H_

#include "gcge.h"

#include "pase_config.h"
#include "pase_param.h"
#include "pase_matvec.h"
#include "pase_ops.h"
#include "pase_mg.h"

void PASE_BMG( PASE_MULTIGRID mg, PASE_INT fine_level, 
               void **rhs, void **sol, 
               PASE_INT *start, PASE_INT *end,
               PASE_REAL tol, PASE_REAL rate, 
               PASE_INT nsmooth, PASE_INT max_coarsest_nsmooth);

#endif
