#ifndef __PASE_PARAM_H__
#define __PASE_PARAM_H__

#include "pase_config.h"

typedef struct PASE_PARAMETER_PRIVATE_ {
  /* amg_parameter */
  /**
   * 列出 amg setup phase 的相关参数, 
   * 如 强影响因子, 粗话策略, 插值类型 等等.
   */
  PASE_INT coarse_type; // CLJP, falgout, ...
  PASE_INT max_level;
  PASE_INT min_coarse_size;
  
  /* 外部软件包, 用于提供实际的矩阵和向量的数据结构与操作 */
  PASE_INT external_package; // 1: HYPRE
  
  /* mg_solver parameter */
  PASE_INT  cycle_type;
  PASE_INT  block_size;
  PASE_INT  max_cycle;
  PASE_INT  max_pre_iter;
  PASE_INT  max_post_iter;
  PASE_INT  print_level;
  PASE_REAL atol;
  PASE_REAL rtol;
   
  /* linear smoother/solver */
  /**
   * 1. 线性解法器/光滑子类型
   *    CG, GS, SOR, GMRES, AMG 等等.
   *    非负数表明使用默认求解方法, 负数表明用户自行提供方法.
   * 2. 求解参数
   *    最大光滑次数, 收敛阈值 等等.
   */
  PASE_INT linear_smoother;      // 光滑.
  PASE_INT linear_solver;        // 求解. 可用于最粗层网格的求解.
  PASE_INT linear_smoother_pre;  // 前光滑. 如无特别指定, 则默认与 linear_smoother 相同
  PASE_INT linear_smoother_post; // 后光滑. 如无特别指定, 则默认与 linear_smoother 相同
  
  /* eigen smoother/solver */
  /**
   * 特征值解法器/光滑子类型
   * Arnoldi, Lanczos, KrylovSchur, LOBPCG 等等.
   *    非负数表明使用默认求解方法, 负数表明用户自行提供方法.
   */
  PASE_INT eigen_smoother;      // 光滑.
  PASE_INT eigen_solver;        // 求解.
  PASE_INT eigen_smoother_pre;  // 前光滑. 如无特别指定, 则默认与 eigen_smoother 相同
  PASE_INT eigen_smoother_post; // 后光滑. 如无特别指定, 则默认与 eigen_smoother 相同
} PASE_PARAMETER_PRIVATE;

typedef PASE_PARAMETER_PRIVATE * PASE_PARAMETER;

/* 参数设置与获取 */
//void PASE_Param_set_linearSmoother(JPINT linear_smoother);
//PASE_INT PASE_Paramater_GetLinearSmoother(void);



#endif
