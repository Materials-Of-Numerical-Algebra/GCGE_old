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
void GCGE_Default_LinearSolver(void *Matrix, void *b, void *x, GCGE_OPS *ops)
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
      ops->MatDotVec(Matrix,p,w, ops);
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
void GCGE_Default_LinearSolverSetUp(void *pc, GCGE_INT max_it, GCGE_DOUBLE tol, 
      void *r, void *p, void *w, GCGE_OPS *ops)
{
   GCGE_PCGSolver *pcg = (GCGE_PCGSolver*)ops->linear_solver_workspace;
   pcg->pc     = pc;
   pcg->max_it = max_it;
   pcg->tol    = tol;
   pcg->r      = r;
   pcg->p      = p;
   pcg->w      = w;
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
void GCGE_Default_MultiLinearSolver(void *Matrix, void **RHS, void **V, 
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
void GCGE_Default_MultiLinearSolverSetUp(void *pc, GCGE_INT max_it, GCGE_DOUBLE tol, 
      void **r, void **p, void **w, GCGE_INT *start_rpw, GCGE_INT *end_rpw, 
      GCGE_DOUBLE *dtmp, GCGE_INT *itmp, 
      GCGE_INT if_shift, GCGE_DOUBLE shift, void *B, 
      GCGE_OPS *ops)
{
   GCGE_BPCGSolver *bpcg = (GCGE_BPCGSolver*)ops->multi_linear_solver_workspace;
   bpcg->pc        = pc;
   bpcg->max_it    = max_it;
   bpcg->tol       = tol;
   bpcg->r         = r;
   bpcg->p         = p;
   bpcg->w         = w;
   bpcg->start_rpw = start_rpw;
   bpcg->end_rpw   = end_rpw;
   bpcg->dtmp      = dtmp;
   bpcg->itmp      = itmp;

   bpcg->if_shift  = if_shift;
   bpcg->shift     = shift;
   bpcg->B         = B;
}
