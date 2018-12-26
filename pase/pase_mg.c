#include "pase_mg.h"
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
PASE_INT 
PASE_MULTIGRID_Create(PASE_MULTIGRID* multi_grid, PASE_INT max_levels, 
        void *A, void *B, GCGE_OPS *gcge_ops, PASE_OPS *pase_ops)
{
  *multi_grid = (PASE_MULTIGRID)PASE_Malloc(sizeof(pase_MultiGrid));
  (*multi_grid)->num_levels = max_levels;
  (*multi_grid)->gcge_ops = gcge_ops;
  (*multi_grid)->pase_ops = pase_ops;
  (*multi_grid)->A_array = (void**)malloc(max_levels*sizeof(void*));
  (*multi_grid)->B_array = (void**)malloc(max_levels*sizeof(void*));
  (*multi_grid)->P_array = (void**)malloc(max_levels*sizeof(void*));
  (*multi_grid)->aux_A   = NULL;
  (*multi_grid)->aux_B   = NULL;
  return 0;
}

PASE_INT 
PASE_MULTIGRID_Destroy(PASE_MULTIGRID* multi_grid)
{
    free((*multi_grid)->A_array);
    (*multi_grid)->A_array = NULL;
    free((*multi_grid)->B_array);
    (*multi_grid)->B_array = NULL;
    free((*multi_grid)->P_array);
    (*multi_grid)->P_array = NULL;
    if((*multi_grid)->aux_A != NULL) {
        PASE_MatrixDestroy(&((*multi_grid)->aux_A), (*multi_grid)->pase_ops);
    }
    if((*multi_grid)->aux_B != NULL) {
        PASE_MatrixDestroy(&((*multi_grid)->aux_B), (*multi_grid)->pase_ops);
    }
    free(*multi_grid); *multi_grid = NULL;
}

/**
 * TODO
 * @brief 将多向量从第level_i层 prolong/restrict 到level_j层
 *
 * @param multigrid 多重网格结构
 * @param level_i   起始层
 * @param level_j   目标层
 * @param mv_s      多向量pvx_i与pvx_j的起始位置
 * @param mv_e      多向量pvx_i与pvx_j的终止位置
 * @param pvx_i     起始多向量
 * @param pvx_j     目标多向量
 *
 * @return 
 */
PASE_INT PASE_MULTIGRID_FromItoJ(PASE_MULTIGRID multi_grid, 
        PASE_INT level_i, PASE_INT level_j, 
        PASE_INT *mv_s, PASE_INT *mv_e, 
        void **pvx_i, void** pvx_j)
{
    //如果level_i(0)<level_j(1),从细层到粗层，限制,P[1]存Restrict限制矩阵
    //如果level_i(1)>level_j(0),从粗层到细层，延拓,P[0]存Prolong延拓矩阵
    multi_grid->gcge_ops->MatDotMultiVec(multi_grid->P_array[level_j], pvx_i, pvx_j, mv_s, mv_e, multi_grid->gcge_ops);
}
