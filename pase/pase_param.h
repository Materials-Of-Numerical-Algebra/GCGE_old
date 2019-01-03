#ifndef __PASE_PARAM_H__
#define __PASE_PARAM_H__

#include "pase_config.h"

typedef struct PASE_PARAMETER_PRIVATE_ {

  PASE_INT num_levels;
  PASE_INT initial_level;
  PASE_INT coarest_level;
  PASE_INT finest_level;

  PASE_INT *max_cycle_count_each_level;
  PASE_INT *max_pre_count_each_level;
  PASE_INT *max_post_count_each_level;
  PASE_INT *max_direct_count_each_level;
  PASE_INT max_initial_count;
  
  PASE_INT  nev;
  PASE_INT  num_given_eigs;
  PASE_INT  bmg_step_size;
  PASE_REAL rtol;
  PASE_REAL atol;
  PASE_INT  print_level;

} PASE_PARAMETER_PRIVATE;

typedef PASE_PARAMETER_PRIVATE * PASE_PARAMETER;

void
PASE_PARAMETER_Create(PASE_PARAMETER *param, PASE_INT num_levels, PASE_INT nev);

void 
PASE_PARAMETER_Destroy(PASE_PARAMETER *param);
#endif
