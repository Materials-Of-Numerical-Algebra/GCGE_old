#ifndef __PASE_AMG_H__
#define __PASE_AMG_H__

#include <math.h>
#include "pase_mg.h"
#include "pase_param.h"
#include "pase.h"
#include "gcge.h"

void PASE_BMG( PASE_MULTIGRID mg, PASE_INT fine_level, 
               void **rhs, void **sol, 
               PASE_INT *start, PASE_INT *end,
               PASE_REAL tol, PASE_REAL rate, 
               PASE_INT nsmooth, PASE_INT max_coarsest_nsmooth);

#endif
