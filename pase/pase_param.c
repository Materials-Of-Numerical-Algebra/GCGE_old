#include "pase_param.h"

void
PASE_PARAMETER_Create(PASE_PARAMETER *param, PASE_INT num_levels, PASE_INT nev)
{

  *param = (PASE_PARAMETER)malloc(sizeof(PASE_PARAMETER_PRIVATE));
  (*param)->num_levels = num_levels;
  (*param)->initial_level = -1;
  (*param)->mg_coarsest_level = -1;
  (*param)->aux_coarse_level = -1;
  (*param)->finest_level = -1;
  (*param)->num_given_eigs = 0;
  (*param)->max_cycle_count_each_level = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  (*param)->max_pre_count_each_level = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  (*param)->max_post_count_each_level = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  (*param)->max_direct_count_each_level = (PASE_INT*)calloc(num_levels, sizeof(PASE_INT));
  (*param)->max_initial_direct_count = 30;
  (*param)->nev = nev;
  (*param)->rtol = 1e-8;
  (*param)->atol = 1e-8;
  (*param)->initial_rtol = 1e-8;
  (*param)->aux_rtol = 1e-10;
  (*param)->print_level = 1;
  (*param)->bmg_step_size = -1;

  PASE_INT i = 0;
  for(i=0; i<num_levels; i++) {
    (*param)->max_cycle_count_each_level[i] = 1;
  }
  (*param)->max_cycle_count_each_level[0] = 5;
  for(i=0; i<num_levels; i++) {
    (*param)->max_pre_count_each_level[i] = 4;
  }
  for(i=0; i<num_levels; i++) {
    (*param)->max_post_count_each_level[i] = 4;
  }
  for(i=0; i<num_levels; i++) {
    (*param)->max_direct_count_each_level[i] = 3;
  }

}

void 
PASE_PARAMETER_Destroy(PASE_PARAMETER *param)
{
  free((*param)->max_cycle_count_each_level);
  free((*param)->max_pre_count_each_level);
  free((*param)->max_post_count_each_level);
  free((*param)->max_direct_count_each_level);
  free(*param); *param = NULL;
}
