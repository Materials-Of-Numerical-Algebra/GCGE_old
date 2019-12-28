#ifndef _PASE_CONFIG_H_
#define _PASE_CONFIG_H_

//=============================================================================
/* 基本数据类型的封装 */
typedef int    PASE_INT;
typedef double PASE_DOUBLE;
typedef double PASE_REAL;
typedef double PASE_SCALAR;

//=============================================================================
#include <stdlib.h>

#define PASE_Malloc  malloc
#define PASE_Free(a) { free(a); a = NULL; }

//=============================================================================
#define PASE_NO    0
#define PASE_YES   1

#define CLK_TCK 1000000

#endif
