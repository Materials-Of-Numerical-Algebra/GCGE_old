#ifndef _PASE_PARAM_H_
#define _PASE_PARAM_H_

#include "pase_config.h"

typedef struct PASE_PARAMETER_PRIVATE_ {

  PASE_INT  num_levels;
  PASE_INT  initial_level;
  PASE_INT  mg_coarsest_level;
  PASE_INT  initial_aux_coarse_level;
  PASE_INT  finest_aux_coarse_level;
  PASE_INT  finest_level;

  PASE_INT *max_cycle_count_each_level;
  PASE_INT *max_pre_count_each_level;
  PASE_INT *max_post_count_each_level;
  PASE_INT *max_direct_count_each_level;
  PASE_INT  max_initial_direct_count;

  //是否检查最优aux_coarse_level
  PASE_INT  check_efficiency_flag;
  
  PASE_INT  nev;
  PASE_INT  more_nev;
  PASE_INT  num_given_eigs;
  PASE_INT  bmg_step_size;
  PASE_REAL rtol;
  PASE_REAL atol;
  PASE_REAL initial_rtol;
  PASE_REAL aux_rtol;
  PASE_INT  print_level;

} PASE_PARAMETER_PRIVATE;

typedef PASE_PARAMETER_PRIVATE * PASE_PARAMETER;

void
PASE_PARAMETER_Create(PASE_PARAMETER *param, PASE_INT num_levels, PASE_INT nev);

void 
PASE_PARAMETER_Destroy(PASE_PARAMETER *param);

void 
PASE_PARAMETER_Get_from_command_line(PASE_PARAMETER param, PASE_INT argc, char *argv[]);
#endif
