/*
 * =====================================================================================
 *
 *       Filename:  pash_mv.h
 *
 *    Description:  后期可以考虑将行参类型都变成void *, 以方便修改和在不同计算机上调试
 *                  一般而言, 可以让用户调用的函数以PASE_开头, 内部函数以pase_开头
 *
 *        Version:  1.0
 *        Created:  2017年08月29日 14时15分22秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  LIYU 
 *   Organization:  LSEC
 *
 * =====================================================================================
 */

#ifndef _pase_matvec_h_
#define _pase_matvec_h_

//#include "pase_ops.h"
#include "pase_config.h"



typedef struct pase_Matrix_struct 
{
   
   PASE_INT  num_aux_vec;
   /* N_H阶的并行矩阵 */
   void      *A_H;
   /* num_aux_vec个N_H阶的并行向量 */
   void      **aux_Hh;
   /* num_aux_vec个N_H阶的并行向量, 对于对称矩阵, 这个指针直接直向aux_Hh */
   void      **aux_hH;
   /* num_aux_vec*num_aux_vec的数组 */
   PASE_REAL *aux_hh;
   PASE_INT  if_sym;
   PASE_INT  is_diag;

   //PASE_OPS  *pase_ops;

#if 0
   hypre_ParCSRMatrix*   P;
   /* TODO:
    * 利用mv_TempMultiVector进行存储*/
   hypre_ParVector**     u_h;

   /* 是否是块对角矩阵1, 则是 */
   PASE_INT             diag;
#endif

} pase_Matrix;
typedef struct pase_Matrix_struct *PASE_Matrix;


typedef struct pase_Vector_struct 
{
   PASE_INT   num_aux_vec;
   /* N_H阶的并行矩阵 */
   void*      b_H;
   /* blockSize的数组 */
   PASE_REAL* aux_h;
   //PASE_OPS   *pase_ops;

} pase_Vector;
typedef struct pase_Vector_struct *PASE_Vector;

typedef struct pase_MultiVector_struct 
{
   PASE_INT   num_aux_vec;
   PASE_INT   num_vec;
   /* N_H阶的并行矩阵 */
   void**     b_H;
   /* LDA = num_aux_vec的数组 */
   PASE_REAL* aux_h;
   //PASE_OPS   *pase_ops;

} pase_MultiVector;
typedef struct pase_MultiVector_struct *PASE_MultiVector;

#if 0
/* 这里不应该如此简单的给这样直接的行参, 应该给A_H A_h 以及block_size个h上的向量, 形成PASE矩阵 */
				    
PASE_INT PASE_MatrixCreate( PASE_Matrix* pase_matrix,
                            PASE_INT num_aux_vec,
                            void *A_H, 
                            void *P,
                            void *A_h, 
				            void **u_h, 
				            void **workspace_h, 
				            GCGE_OPS* gcge_ops
				            );
PASE_INT PASE_MatrixDestroy( PASE_Matrix*  matrix );

PASE_INT PASE_MatrixSetAuxSpace( PASE_Matrix  matrix, 
                                 PASE_INT num_aux_vec,
                                 void *P,
                                 void *A_h, 
				                 void **u_h, 
                                 PASE_INT begin_idx,
				                 void **workspace_h,
				                 GCGE_OPS* gcge_ops
				                 );

#endif
#if 0
PASE_INT PASE_MatrixPrint( PASE_Matrix matrix , const char *file_name );
PASE_INT PASE_VectorPrint( PASE_Vector vector , const char *file_name );
PASE_INT PASE_ParVectorCopy(PASE_ParVector x, PASE_ParVector y);
PASE_INT PASE_ParVectorInnerProd( PASE_ParVector x, PASE_ParVector y, PASE_REAL *prod);
PASE_INT PASE_ParVectorCopy(PASE_ParVector x, PASE_ParVector y);
PASE_INT PASE_ParVectorAxpy( PASE_REAL alpha , PASE_ParVector x , PASE_ParVector y );
PASE_INT PASE_ParVectorSetConstantValues( PASE_ParVector v , PASE_REAL value );
PASE_INT PASE_ParVectorScale ( PASE_REAL alpha , PASE_ParVector y );
PASE_INT PASE_ParVectorSetRandomValues( PASE_ParVector v, PASE_INT seed );

PASE_INT PASE_ParVectorGetParVector( HYPRE_ParCSRMatrix P, PASE_INT block_size, HYPRE_ParVector *vector_h, 
      PASE_ParVector vector_Hh, HYPRE_ParVector vector );



/* y = alpha A + beta y */
PASE_INT PASE_ParCSRMatrixMatvec ( PASE_REAL alpha, PASE_ParCSRMatrix A, PASE_ParVector x, PASE_REAL beta, PASE_ParVector y );
PASE_INT PASE_ParCSRMatrixMatvecT( PASE_REAL alpha, PASE_ParCSRMatrix A, PASE_ParVector x, PASE_REAL beta, PASE_ParVector y );





PASE_INT
PASE_ParCSRMatrixSetAuxSpaceByPASE_ParCSRMatrix( MPI_Comm comm , 
				   PASE_ParCSRMatrix  matrix, 
                                   PASE_INT block_size,
                                   HYPRE_ParCSRMatrix P,
                                   PASE_ParCSRMatrix A_h, 
				   PASE_ParVector*   u_h, 
				   HYPRE_ParVector    workspace_H, 
				   PASE_ParVector    workspace_hH
				   );

PASE_INT
PASE_ParCSRMatrixCreateByPASE_ParCSRMatrix( MPI_Comm comm , 
                                   PASE_INT block_size,
                                   HYPRE_ParCSRMatrix A_H, 
                                   HYPRE_ParCSRMatrix P,
                                   PASE_ParCSRMatrix A_h, 
				   PASE_ParVector*   u_h, 
				   PASE_ParCSRMatrix* matrix, 
				   HYPRE_ParVector    workspace_H, 
				   PASE_ParVector    workspace_hH
				   );


















/* 注意向量的类型 */
PASE_INT PASE_ParCSRMatrixMatvec_HYPRE_ParVector ( PASE_REAL alpha, PASE_ParCSRMatrix A, HYPRE_ParVector x, PASE_REAL beta, HYPRE_ParVector y );

#endif

#endif
