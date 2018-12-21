
#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_create"
/**
 * @brief 创建 PASE_MULTIGRID
 *
 * @param A      输入参数
 * @param B      输入参数
 * @param param  输入参数, 包含 AMG 分层的各个参数
 * @param ops    输入参数, 多重网格操作集合
 *
 * @return PASE_MULTIGRID
 */
PASE_MULTIGRID 
PASE_Multigrid_create(void *A, void *B, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops)
{
  PASE_MULTIGRID multigrid = (PASE_MULTIGRID)PASE_Malloc(sizeof(PASE_MULTIGRID_PRIVATE));
  /* TODO */
#if 0
  if(NULL != ops) {
    multigrid->ops = (PASE_MULTIGRID_OPERATOR) PASE_Malloc(sizeof(PASE_MULTIGRID_OPERATOR_PRIVATE));
    *(multigrid->ops) = *ops;
  } else {
    multigrid->ops = PASE_Multigrid_operator_create(A->data_form);
  }

  //PASE_Multigrid_set_up(multigrid, A, B, param);

#endif
  return multigrid;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_get_amg_array"
/**
 * @brief AMG 分层
 *
 * @param multigrid  输入/输出参数 
 * @param A          输入参数
 * @param B          输入参数
 * @param param      输入参数, 包含 AMG 分层的各个参数
 */
void
PASE_Multigrid_set_up(PASE_MULTIGRID multigrid, PASE_MATRIX A, PASE_MATRIX B, PASE_PARAMETER param)
{
  void **A_array, **P_array, **R_array; 
  PASE_INT    level = 0;
  PASE_MATRIX tmp   = NULL;
  multigrid->ops->get_amg_array(A->matrix_data, 
                                param, 
                                &(A_array),
                                &(P_array),
                                &(R_array),
                                &(multigrid->actual_level),
                                &(multigrid->amg_data));
  multigrid->A     = (PASE_MATRIX*)PASE_Malloc(multigrid->actual_level*sizeof(PASE_MATRIX));
  multigrid->B     = (PASE_MATRIX*)PASE_Malloc(multigrid->actual_level*sizeof(PASE_MATRIX));
  multigrid->P     = (PASE_MATRIX*)PASE_Malloc((multigrid->actual_level-1)*sizeof(PASE_MATRIX));
  multigrid->R     = (PASE_MATRIX*)PASE_Malloc((multigrid->actual_level-1)*sizeof(PASE_MATRIX));
  multigrid->aux_A = (PASE_AUX_MATRIX*)calloc(multigrid->actual_level, sizeof(PASE_AUX_MATRIX));
  multigrid->aux_B = (PASE_AUX_MATRIX*)calloc(multigrid->actual_level, sizeof(PASE_AUX_MATRIX));
  multigrid->A[0]  = A;
  multigrid->B[0]  = B;
  for(level=1; level<multigrid->actual_level; level++) {
    multigrid->A[level]                         = PASE_Matrix_assign(A_array[level], A->ops);
    multigrid->A[level]->data_form              = A->data_form;
    multigrid->P[level-1]                       = PASE_Matrix_assign(P_array[level-1], A->ops);
    multigrid->P[level-1]->data_form            = A->data_form;
    multigrid->R[level-1]                       = PASE_Matrix_assign(R_array[level-1], A->ops);
    multigrid->R[level-1]->is_matrix_data_owner = 1;
    multigrid->R[level-1]->data_form            = A->data_form;

    /* B1 = R0 * B0 * P0 */
    tmp                                         = PASE_Matrix_multiply_matrix(multigrid->B[level-1], multigrid->P[level-1]); 
    multigrid->B[level]                         = PASE_Matrix_multiply_matrix(multigrid->R[level-1], tmp); 
    PASE_Matrix_destroy(tmp);
  }
  PASE_Free(R_array);
}

