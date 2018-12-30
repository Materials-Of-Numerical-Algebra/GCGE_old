#ifndef _pase_mg_h_
#define _pase_mg_h_

#include "pase.h"

#include "pase_convert.h"
#include "_hypre_parcsr_ls.h"

#define pase_MultiGridDataAArray(data)  ((data)->A_array)
#define pase_MultiGridDataBArray(data)  ((data)->B_array)
#define pase_MultiGridDataPArray(data)  ((data)->P_array)
#define pase_MultiGridDataQArray(data)  ((data)->Q_array)
#define pase_MultiGridDataUArray(data)  ((data)->U_array)
#define pase_MultiGridDataFArray(data)  ((data)->F_array)

#define pase_MultiGridDataNumLevels(data) ((data)->num_levels)
      
typedef struct pase_MultiGrid_struct 
{
   PASE_INT num_levels;
   void     **A_array;
   void     **B_array;
   void     **P_array;
   /* P0P1P2  P1P2  P2 */
   //void     **Q_array;
   /* rhs and x */
   //void     **U_array;
   //void     **F_array;
   void     ***u;
   void     ***rhs;
   void     ***u_tmp;
   void     ***u_tmp_1;
   void     ***u_tmp_2;

   PASE_REAL *double_tmp;
   PASE_INT  *int_tmp;

   PASE_Matrix aux_A;
   PASE_Matrix aux_B;

   GCGE_OPS *gcge_ops;
   PASE_OPS *pase_ops;
   
} pase_MultiGrid;
typedef struct pase_MultiGrid_struct *PASE_MULTIGRID;

PASE_INT 
PASE_MULTIGRID_Create(PASE_MULTIGRID* multi_grid, PASE_INT max_levels, 
        void *A, void *B, GCGE_OPS *gcge_ops, PASE_OPS *pase_ops);

PASE_INT PASE_MULTIGRID_Destroy(PASE_MULTIGRID* multi_grid);


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
        void **pvx_i, void** pvx_j);

#endif
