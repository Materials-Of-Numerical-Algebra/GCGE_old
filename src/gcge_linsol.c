/*
 * =====================================================================================
 *
 *       Filename:  gcge_cg.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时57分13秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gcge_linsol.h"

/**
 * @brief CG迭代求解Matrix * x = b
 *
 * 本函数不直接修改b的值，但如果利用
 * GCGE_Default_LinearSolverSetUp
 * 设定工作空间时，可以将b设为linear_solver_workspace->p or w;
 * 不能设为r，因为在一开始需要用b和r求初始残量
 *
 * @param Matrix  求解的矩阵(GCGE_MATRIX)
 * @param b       右端项向量(GCGE_VEC)
 * @param x       解向量    (GCGE_VEC)
 * @param ops
 */
void GCGE_LinearSolver_PCG(void *Matrix, void *b, void *x, GCGE_OPS *ops)
{
   GCGE_PCGSolver *pcg = (GCGE_PCGSolver*)ops->linear_solver_workspace;
   void        *r, *p, *w;
   GCGE_INT    niter = 0;
   GCGE_INT    max_it = pcg->max_it, type = 1;
   GCGE_DOUBLE rate = pcg->rate, tol = pcg->tol;
   GCGE_DOUBLE alpha, beta, rho1, rho2, init_error, last_error, pTw, wTw;

   /* CG迭代中用到的临时向量 */
   r = pcg->r; //记录残差向量
   p = pcg->p; //记录下降方向
   w = pcg->w; //记录tmp=A*p
   if (r==NULL || p==NULL || w==NULL)
   {
      GCGE_Printf("Default linear solver is used. Please set r p w in ops->linear_solver_workspace.\n");
      exit;
   }

   ops->MatDotVec(Matrix, x, r, ops);
   ops->VecAxpby(1.0, b, -1.0, r, ops); //r = b-A*x 
   ops->VecInnerProd(r, r, &rho2, ops);//用残量的模来判断误差
   init_error = sqrt(rho2);
   niter = 1;
   last_error = init_error;
   /* 当last_error< rate*init_error时停止迭代, 即最后的误差下降到初始误差的rate倍时停止 */
   while( (last_error >= rate*init_error)&&(last_error>tol)&&(niter<max_it) )
   {
      if(niter == 1)
      {
         //set the direction as r
         ops->VecAxpby(1.0, r, 0.0, p, ops); //p = r;
      }
      else
      {
         //compute the value of beta
         beta = rho2/rho1; 
         //compute the new direction: p = r + beta * p
         ops->VecAxpby(1.0, r, beta, p, ops);
      }//end for if(niter == 1)
      //compute the vector w = A*p
      ops->MatDotVec(Matrix, p, w, ops);
      if(type == 1)
      {
         //compute the value pTw = p^T * w 
         ops->VecInnerProd(p, w, &pTw, ops);
         //compute the value of alpha
         //printf("pTw=%e\n",pTw);
         alpha = rho2/pTw; 
         //compute the new solution x = alpha * p + x
         ops->VecAxpby(alpha, p, 1.0, x, ops);
         //compute the new residual: r = - alpha*w + r
         ops->VecAxpby(-alpha, w, 1.0, r, ops);
         //set rho1 as rho2
         rho1 = rho2;
         //compute the new rho2
         ops->VecInnerProd(r, r, &rho2, ops); 
      }
      else
      {    
         //这里我们把两个向量内积放在一起计算，这样可以减少一次的向量内积的全局通讯，提高可扩展性
         //compute the value pTw = p^T * w, wTw = w^T*w 
         ops->VecInnerProd(p, w, &pTw, ops);
         ops->VecInnerProd(w, w, &wTw, ops);

         alpha = rho2/pTw; 
         //compute the new solution x = alpha * p + x
         ops->VecAxpby(alpha, p, 1.0, x, ops);
         //compute the new residual: r = - alpha*w + r
         ops->VecAxpby(-alpha, w, 1.0, r, ops);
         //set rho1 as rho2
         rho1 = rho2;
         //compute the new rho2
         rho2 = rho1 - 2.0*alpha*pTw + alpha*alpha*wTw;
      } 
      last_error = sqrt(rho2);      
      //printf("  niter= %d,  The current residual: %10.5f\n",niter, last_error);
      //printf("  error=%10.5f, last_error=%10.5f,  last_error/error= %10.5f\n",error, last_error, last_error/error);
      //update the iteration time
      niter++;   
   }//end while((last_error/error >= rate)&&(niter<max_it))
}//end for the CG program

/**
 * @brief 在调用GCGE默认的LinearSolver之前需要设置LinearSolver
 *        再次调用LinearSolver时，如果参数和临时空间不变，无需再次调用
 *
 * @param pc
 * @param max_it
 * @param tol
 * @param r
 * @param p
 * @param w
 * @param ops
 */
void GCGE_LinearSolver_PCG_SetUp(GCGE_INT max_it, GCGE_DOUBLE rate, GCGE_DOUBLE tol, 
      void *r, void *p, void *w, void *pc, GCGE_OPS *ops)
{
   /* 只初始化一次，且全局可见 */
   static GCGE_PCGSolver pcg_static = {
      .max_it = 50, .rate = 1e-2, .tol=1e-12, 
      .r = NULL, .p = NULL, .w = NULL, 
      .pc = NULL};
   ops->linear_solver_workspace = (void *)&pcg_static;

   GCGE_PCGSolver *pcg = (GCGE_PCGSolver*)ops->linear_solver_workspace;
   pcg->max_it = max_it;
   pcg->rate   = rate;
   pcg->tol    = tol;
   pcg->r      = r;
   pcg->p      = p;
   pcg->w      = w;
   pcg->pc     = pc;
}

/**
 * @brief 
 *
 * 本函数不直接修改RHS值，但如果利用
 * GCGE_Default_MultiLinearSolverSetUp
 * 设定工作空间时，可以将RHS设为multi_linear_solver_workspace->p or w;
 * 不能设为r，因为在一开始需要用b和r求初始残量
 *
 * @param Matrix 矩阵
 * @param RHS    右端项
 * @param V      解向量
 * @param start  start[0] RHS开始的指标  start[1] V开始的指标
 * @param end    end[0]   RHS结束的指标  end[1]   V结束的指标
 * @param ops
 */
void GCGE_MultiLinearSolver_BPCG(void *Matrix, void **RHS, void **V, 
      GCGE_INT *start, GCGE_INT *end, GCGE_OPS *ops) 
{
   GCGE_BPCGSolver *bpcg = (GCGE_BPCGSolver*)ops->multi_linear_solver_workspace;
   GCGE_INT max_it = bpcg->max_it;
   GCGE_DOUBLE tol = bpcg->tol, rate = bpcg->rate;
   void   **CG_B = RHS, **CG_X = V;  
   void   **CG_R = bpcg->r, **CG_P = bpcg->p, **CG_W = bpcg->w; 
   GCGE_INT *start_rpw = bpcg->start_rpw, *end_rpw = bpcg->end_rpw;
   //V中用到的向量个数
   GCGE_INT    x_length = end[1] - start[1];
   if (CG_R==NULL || CG_P==NULL || CG_W==NULL)
   {
      GCGE_Printf("Default multi linear solver is used. Please set r p w in ops->multi_linear_solver_workspace.\n");
      exit;
   }
   if (end_rpw[0]-start_rpw[0] < x_length || 
       end_rpw[1]-start_rpw[1] < x_length ||
       end_rpw[2]-start_rpw[2] < x_length)
   {
      GCGE_Printf("Default multi linear solver is used. Please the length of r p w in ops->multi_linear_solver_workspace is not enough.\n");
      exit;
   }

   GCGE_INT    if_shift = bpcg->if_shift;
   GCGE_DOUBLE shift    = bpcg->shift;
   void        *B       = bpcg->B;
   void        *Bx;
   if((if_shift == 1)&&(B != NULL))
   {
      ops->VecCreateByMat(&Bx, Matrix, ops);
   }
   GCGE_DOUBLE alpha, beta;
   GCGE_DOUBLE *rho1       = bpcg->dtmp, 
	       *rho2       = rho1       + x_length, 
	       *pTw        = rho2       + x_length,
	       *init_error = pTw        + x_length, 
	       *last_error = init_error + x_length;
   GCGE_INT *unlock = bpcg->itmp;

   void   *b, *x, *r, *p, *w;
   GCGE_INT    id, idx, niter = 0;
   GCGE_INT num_unlock = 0, old_num_unlock = 0; 
   
   //计算初始的残差向量和残差
   for(idx=0; idx<x_length; idx++)
   {
      ops->GetVecFromMultiVec(CG_X, start[1]+idx, &x, ops);   //取出初始值 x      
      ops->GetVecFromMultiVec(CG_R, start_rpw[0]+idx, &r, ops); //取出暂存残差r的变量
      ops->MatDotVec(Matrix,x,r, ops); //r = A*x
      if(if_shift == 1)
      {
	 if(B == NULL)
	 {
	    //计算r=Ax+shift*x
	    ops->VecAxpby(shift, x, 1.0, r, ops);
	 }
	 else
	 {
	    //计算r=Ax+shift*Bx
	    ops->MatDotVec(B,x,Bx, ops); 
	    ops->VecAxpby(shift, Bx, 1.0, r, ops);
	 }
      }
      ops->RestoreVecForMultiVec(CG_X, start[1]+idx, &x, ops);
      // r = b - A*x;
      ops->GetVecFromMultiVec(CG_B, start_rpw[0]+idx, &b, ops);    //取出右端项 b
      ops->VecAxpby(1.0, b, -1.0, r, ops);  
      ops->VecLocalInnerProd(r, r, rho2+idx, ops);    //用残量的模来判断误差    
      ops->RestoreVecForMultiVec(CG_B, start[0]+idx, &b, ops); //把b相应的位置设为残差
      ops->RestoreVecForMultiVec(CG_R, start_rpw[0]+idx, &r, ops); //把r返回
   }//算完初始的残差向量和残差
   //统一进行数据传输  
#if GCGE_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, rho2, x_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif 
   //最后计算每个向量的残差(初始残差，用来进行收敛性)
   num_unlock = 0;
   for(idx=0;idx<x_length;idx++)
   {
      init_error[idx] = sqrt(rho2[idx]);
      last_error[idx] = init_error[idx];
      //printf("the initial residual: %e\n",error[idx]);
      if(init_error[idx] > tol)
      {
	 unlock[num_unlock] = idx;
	 num_unlock ++;
      }//end if(last_error[idx]/init_error[idx] >= rate)       
   }

   niter = 1;
   while( (num_unlock > 0)&&(niter < max_it) )
   {
      //printf("num_unlock: %d, niter: %d, max_it: %d, rate: %e, init_error: %e, %e\n", num_unlock, niter, max_it, rate, last_error[0], last_error[1]);
      //对每一个未收敛的向量进行处理
      for(id = 0; id < num_unlock; id++)
      {
	 idx = unlock[id];
	 ops->GetVecFromMultiVec(CG_R, start_rpw[0]+idx, &r, ops);   //取出右端项 r
	 ops->GetVecFromMultiVec(CG_P, start_rpw[1]+idx, &p, ops);   //取出向量 p
	 if(niter == 1)
	 {
	    //set the direction as r: p = r
	    ops->VecAxpby(1.0, r, 0.0, p, ops);
	 }
	 else
	 { 
	    //compute the value of beta
	    beta = rho2[idx]/rho1[idx]; 
	    //compute the new direction: p = r + beta * p
	    ops->VecAxpby(1.0, r, beta, p, ops);          
	 }////end for if(niter == 1)
	 ops->RestoreVecForMultiVec(CG_R, start_rpw[0]+idx, &r, ops);
	 //compute the vector w = A*p和p^Tw = pTw
	 ops->GetVecFromMultiVec(CG_W, start_rpw[2]+idx, &w, ops);   //取出 w
	 ops->MatDotVec(Matrix,p,w,ops);  //w = A*p
	 if(if_shift == 1)
	 {
	    if(B == NULL)
	    {
	       //计算w=Ap+shift*p
	       ops->VecAxpby(shift, p, 1.0, w, ops);
	    }
	    else
	    {
	       //计算w=Ap+shift*Bp
	       ops->MatDotVec(B,p,Bx, ops); 
	       ops->VecAxpby(shift, Bx, 1.0, w, ops);
	    }
	 }
	 //做局部内积（在每个进程内部做局部的内积）
	 ops->VecLocalInnerProd(p, w, pTw+id, ops);
	 ops->RestoreVecForMultiVec(CG_P, start_rpw[1]+idx, &p, ops);
	 ops->RestoreVecForMultiVec(CG_W, start_rpw[2]+idx, &w, ops);       
      }//end for id
      //compute the value pTw = p^T * w
#if GCGE_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, pTw, num_unlock, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif     
      //现在来计算: alpha = rho2/pTw, x = alpha*p +x , r = -alpha*w + r, 并且同时计算r的局部内积
      for(id=0;id<num_unlock;id++)
      { 
	 idx = unlock[id];
	 //计算alpha
	 alpha = rho2[id]/pTw[id];
	 //compute the new solution x = alpha * p + x
	 ops->GetVecFromMultiVec(CG_P, start_rpw[1]+idx, &p, ops);   //取出 p
	 ops->GetVecFromMultiVec(CG_X, start[1]+idx, &x, ops); //取出初始值 x 
	 //  x = alpha*p +x 
	 ops->VecAxpby(alpha, p, 1.0, x, ops);
	 ops->RestoreVecForMultiVec(CG_P, start_rpw[1]+idx, &p, ops); 
	 ops->RestoreVecForMultiVec(CG_X, start[1]+idx, &x, ops);
	 //compute the new r and residual
	 ops->GetVecFromMultiVec(CG_R, start_rpw[0]+idx, &r, ops);   //取出 r
	 ops->GetVecFromMultiVec(CG_W, start_rpw[2]+idx, &w, ops);   //取出 w
	 //r = -alpha*w + r
	 ops->VecAxpby(-alpha, w, 1.0, r, ops);
	 //set rho1 as rho2
	 rho1[id] = rho2[id];
	 //计算新的r的局部内积
	 ops->VecLocalInnerProd(r, r, rho2+id, ops);       
	 ops->RestoreVecForMultiVec(CG_R, start[0]+idx, &r, ops); 
	 ops->RestoreVecForMultiVec(CG_W, start[3]+idx, &w, ops);
      }//end for idx
      //统一进行数据传输  
#if GCGE_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, rho2, num_unlock, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      for(id=0;id<num_unlock;id++)
      {
	 idx = unlock[id];
	 //error是要每个向量都存储的，只是我们之需要更新未收敛向量的误差就可以
	 last_error[idx] = sqrt(rho2[id]);  
      }
      //下面进行收敛性的判断
      old_num_unlock = num_unlock;
      num_unlock = 0;
      for(id=0;id<old_num_unlock;id++)
      {   
	 idx = unlock[id];
	 if((last_error[idx] >= rate*init_error[idx]) && (last_error[idx] > tol))
	 {
	    unlock[num_unlock] = idx;
	    num_unlock ++;
	 }//end if(last_error[idx]/init_error[idx] >= rate)       
      }//end for id        
      //update the iteration time
      niter++;       
   }//end while((last_error/init_error >= rate)&&(niter<max_it))
   if((if_shift == 1)&&(B != NULL))
   {
      ops->VecDestroy(&Bx, ops);
   }
}//end for this subprogram

/**
 * @brief 在调用GCGE默认的MultiLinearSolver之前需要设置MultiLinearSolver
 *        再次调用MultiLinearSolver时，如果参数和临时空间不变，无需再次调用
 *
 * @param pc
 * @param max_it
 * @param tol
 * @param r
 * @param p
 * @param w
 * @param start_rpw  长度为3,            start_rpw[0] r可用的起始
 * @param end_rpw    长度为3,            end_rpw[0] r可用的结束
 * @param dtmp       5*count(MultiVec)   double型工作空间 
 * @param itmp       1*count(MultiVec)   int型工作空间   
 * @param ops
 */
void GCGE_MultiLinearSolver_BPCG_SetUp(GCGE_INT max_it, GCGE_DOUBLE rate, GCGE_DOUBLE tol, 
      void **r, void **p, void **w, GCGE_INT *start_rpw, GCGE_INT *end_rpw, 
      GCGE_DOUBLE *dtmp, GCGE_INT *itmp, void *pc, 
      GCGE_INT if_shift, GCGE_DOUBLE shift, void *B, 
      GCGE_OPS *ops)
{
   /* 只初始化一次，且全局可见 */
   static GCGE_BPCGSolver bpcg_static = {
      .max_it = 50, .rate = 1e-2, .tol=1e-12, 
      .r = NULL, .p = NULL, .w = NULL, 
      .start_rpw = NULL, .end_rpw = NULL, 
      .dtmp = NULL, .itmp = NULL, 
      .pc = NULL, 
      .if_shift = 0, .shift = 0.0, .B = NULL};
   ops->multi_linear_solver_workspace = (void *)&bpcg_static;

   GCGE_BPCGSolver *bpcg = (GCGE_BPCGSolver*)ops->multi_linear_solver_workspace;
   bpcg->max_it    = max_it;
   bpcg->rate      = rate;
   bpcg->tol       = tol;
   bpcg->r         = r;
   bpcg->p         = p;
   bpcg->w         = w;
   bpcg->start_rpw = start_rpw;
   bpcg->end_rpw   = end_rpw;
   bpcg->dtmp      = dtmp;
   bpcg->itmp      = itmp;
   bpcg->pc        = pc;

   bpcg->if_shift  = if_shift;
   bpcg->shift     = shift;
   bpcg->B         = B;
}

/* GCGE_BMG 算法过程
 * 递归求解，给定要求解的层号current_level, 该层右端项rhs, 解sol
 *
 * 如果current_level是最细层, 直接求解(暂时用GCGE_BCG多迭代几次)
 * 否则 coarse_level = current_level + 1
 *
 * 1. 前光滑, 使用GCGE_BCG迭代几次, 
 *    需要两组工作空间，使用mg->u_tmp与mg->u_tmp_1
 *    (即为 solver->u_tmp_1 与 solver->u_tmp_2)
 *
 * 2. 计算当前细层的残差residual = rhs - A * sol
 *    residual 使用 mg->u_tmp 的工作空间
 *
 * 3. 将当前层的残差residual投影到粗一层空间上
 *    coarse_residual = R * residual
 *    coarse_residual 使用 mg->rhs 的工作空间(因为是用作下一层的rhs)
 *    (mg->rhs = solver->u_tmp)
 *
 * 4. 给 coarse_sol 赋初值为 0
 *    coarse_sol 使用 mg->u 的工作空间
 *
 * 5. 递归调用 GCGE_BMG, 层号为 coarse_level
 *
 * 6. 将粗层的解 coarse_sol 插值到current_level层, 并加到sol上
 *    residual = P * coarse_sol
 *    sol += residual
 *
 * 7. 后光滑, 使用GCGE_BCG迭代几次
 *
 * 工作空间总结：
 *    递归调用过程中, 
 *    解sol所用空间一直是mg->u(即 solver->u)
 *    右端项rhs所用空间一直是mg->rhs(即 solver->u_tmp)
 *    残差residual所用空间一直是mg->u_tmp(即 solver->u_tmp_1)
 *    GCGE_BCG使用空间为 mg->u_tmp, mg->u_tmp_1(solver->u_tmp_1, 2)
 */
/*
 * mg : 包含需要的工作空间
 * current_level: 从哪个细层开始进行多重网格计算，都会算到mg中的最粗层
 * offset: 0 是最细层还是最粗层,默认最细层
 * rhs: 多个右端项
 * sol: 多个解向量
 * start,end: rhs与sol分别计算第几个到第几个向量
 * tol: 收敛准则 TODO 目前这个参数没用到
 * rate: 前后光滑中CG迭代的精度提高比例
 * nsmooth: 各细层CG迭代次数
 * max_coarsest_smooth : 最粗层最大迭代次数
 */
static void BlockAlgebraicMultiGridSolver( GCGE_INT current_level, 
	       void **rhs, void **sol, 
               GCGE_INT *start, GCGE_INT *end,
	       GCGE_OPS *ops )
{
   //printf("current_level: %d, coarsest_level: %d, rate: %e\n", current_level, mg->coarsest_level, rate);
   GCGE_BAMGSolver *bamg = (GCGE_BAMGSolver *)ops->multi_linear_solver_workspace;
   //默认0层为最细层
   // obtain the coarsest level
   GCGE_INT coarsest_level = bamg->num_levels;
   GCGE_INT mv_s[2];
   GCGE_INT mv_e[2];
   void *A;

   /* --------------------------------------------------------------- */
   // obtain the 'enough' accurate solution on the coarest level
   //direct solving the linear equation
   A = bamg->A_array[current_level];
   GCGE_MultiLinearSolver_BPCG_SetUp( 
	 bamg->max_it[current_level], 
	 bamg->rate[current_level], 
	 bamg->tol[current_level], 
	 bamg->r_array[current_level], 
	 bamg->p_array[current_level], 
	 bamg->w_array[current_level], 
	 bamg->start_rpw, 
	 bamg->end_rpw, 
	 bamg->dtmp, bamg->itmp, NULL,
	 0, 0.0, NULL, 
	 ops);
   GCGE_MultiLinearSolver_BPCG(A, rhs, sol, start, end, ops);
   //GCGE_Printf("current_level: %d, after direct\n", current_level);
   //mg->gcge_ops->MultiVecPrint(sol, 1, mg->gcge_ops);
   if( current_level < coarsest_level )
   {   
      mv_s[0] = start[1];
      mv_e[0] = end[1];
      mv_s[1] = bamg->start_rpw[0];
      mv_e[1] = bamg->end_rpw[0];
      //计算residual = A*sol
      void **residual = bamg->r_array[current_level];
      ops->MatDotMultiVec(A, sol, residual, mv_s, mv_e, ops);
      //计算residual = rhs-A*sol
      mv_s[0] = start[0];
      mv_e[0] = end[0];
      mv_s[1] = bamg->start_rpw[0];
      mv_e[1] = bamg->end_rpw[0];
      ops->MultiVecAxpby(1.0, rhs, -1.0, residual, mv_s, mv_e, ops);

      //把residual投影到粗网格
      GCGE_INT coarse_level = current_level + 1;
      void **coarse_rhs = bamg->rhs_array[coarse_level];
      mv_s[0] = bamg->start_rpw[0];
      mv_e[0] = bamg->end_rpw[0];
      mv_s[1] = bamg->start_bx[0];
      mv_e[1] = bamg->end_bx[0];
      ops->MultiVecFromItoJ(bamg->P_array, current_level, coarse_level, 
	    residual, coarse_rhs, bamg->w_array, mv_s, mv_e, ops);

      //求粗网格解问题，利用递归
      void **coarse_sol = bamg->sol_array[coarse_level];
      mv_s[0] = bamg->start_bx[1];
      mv_e[0] = bamg->end_bx[1];
      mv_s[1] = bamg->start_bx[1];
      mv_e[1] = bamg->end_bx[1];
      //先给coarse_sol赋初值0
      ops->MultiVecAxpby(0.0, coarse_sol, 0.0, coarse_sol, mv_s, mv_e, ops);
      GCGE_MultiLinearSolver_BAMG_SetUp(
	 bamg->max_it,    bamg->rate,      bamg->tol, 
	 bamg->A_array,   bamg->P_array,
	 bamg->num_levels,
	 bamg->rhs_array, bamg->sol_array, 
	 bamg->start_bx,  bamg->end_bx, 
	 bamg->r_array,   bamg->p_array,   bamg->w_array, 
	 bamg->start_rpw, bamg->end_rpw, 
	 bamg->dtmp,      bamg->itmp,      bamg->pc, 
	 ops);
      BlockAlgebraicMultiGridSolver( coarse_level, 
	    coarse_rhs, coarse_sol, 
	    bamg->start_bx, bamg->end_bx,
	    ops );
      //GCGE_Printf("current_level: %d, after postsmoothing\n", current_level);
      //mg->gcge_ops->MultiVecPrint(sol, 1, mg->gcge_ops);

      // 把粗网格上的解插值到细网格，再加到前光滑得到的近似解上
      // 可以用 residual 代替
      mv_s[0] = bamg->start_rpw[0];
      mv_e[0] = bamg->end_rpw[0];
      mv_s[1] = bamg->start_bx[1];
      mv_e[1] = bamg->end_bx[1];
      ops->MultiVecFromItoJ(bamg->P_array, coarse_level, current_level, 
	    coarse_sol, residual, bamg->w_array, mv_s, mv_e, ops);
      //计算residual = rhs-A*sol
      mv_s[0] = bamg->start_rpw[0];
      mv_e[0] = bamg->end_rpw[0];
      mv_s[1] = start[1];
      mv_e[1] = end[1];
      ops->MultiVecAxpby(1.0, residual, 1.0, sol, mv_s, mv_e, ops);

      //后光滑
      GCGE_MultiLinearSolver_BPCG_SetUp( 
	    bamg->max_it[current_level], 
	    bamg->rate[current_level], 
	    bamg->tol[current_level], 
	    bamg->r_array[current_level], 
	    bamg->p_array[current_level], 
	    bamg->w_array[current_level], 
	    bamg->start_rpw, 
	    bamg->end_rpw, 
	    bamg->dtmp, bamg->itmp, NULL,
	    0, 0.0, NULL, 
	    ops);
      GCGE_MultiLinearSolver_BPCG(A, rhs, sol, start, end, ops);
   }//end for (if current_level)
}

void GCGE_MultiLinearSolver_BAMG(void *Matrix, void **RHS, void **V, 
      GCGE_INT *start, GCGE_INT *end, GCGE_OPS *ops) 
{
   BlockAlgebraicMultiGridSolver(0, RHS, V, start, end, ops);
}

void GCGE_MultiLinearSolver_BAMG_SetUp(
      GCGE_INT    *max_it,      GCGE_DOUBLE *rate,        GCGE_DOUBLE *tol, 
      void        **A_array,    void        **P_array, 
      GCGE_INT    num_levels,
      void        ***rhs_array, void        ***sol_array, 
      GCGE_INT    *start_bx,    GCGE_INT    *end_bx,
      void        ***r_array,   void        ***p_array,   void        ***w_array,
      GCGE_INT    *start_rpw,   GCGE_INT    *end_rpw, 
      GCGE_DOUBLE *dtmp,        GCGE_INT    *itmp, 
      void        *pc,
      GCGE_OPS *ops)
{
   static GCGE_BAMGSolver bamg_static = {
      .max_it     = NULL, .rate      = NULL, .tol     = NULL, 
      .A_array    = NULL, .P_array   = NULL, 
      .num_levels = 0,
      .rhs_array  = NULL, .sol_array = NULL,
      .r_array    = NULL, .p_array   = NULL, .w_array = NULL, 
      .start_rpw  = NULL, .end_rpw   = NULL, 
      .dtmp       = NULL, .itmp      = NULL, 
      .pc         = NULL};
   ops->multi_linear_solver_workspace = (void *)&bamg_static;

   GCGE_BAMGSolver *bamg = (GCGE_BAMGSolver*)ops->multi_linear_solver_workspace;
 
   bamg->max_it  = max_it;    
   bamg->tol     = tol; 
   bamg->A_array    = A_array;   
   bamg->P_array    = P_array;
   bamg->num_levels = num_levels;
   bamg->rhs_array  = rhs_array;       
   bamg->sol_array  = sol_array; 
   bamg->start_bx   = start_bx;
   bamg->end_bx     = end_bx; 
   bamg->r_array    = r_array;
   bamg->p_array    = p_array;
   bamg->w_array    = w_array;
   bamg->start_rpw  = start_rpw; 
   bamg->end_rpw    = end_rpw; 
   bamg->dtmp       = dtmp;
   bamg->itmp       = itmp;
   bamg->pc         = pc;
}




