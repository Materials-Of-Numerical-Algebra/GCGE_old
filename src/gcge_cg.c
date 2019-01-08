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

#include <math.h>

#include "gcge_cg.h"

//CG迭代求解Matrix * x = b
//Matrix是用于求解的矩阵(GCGE_MATRIX),b是右端项向量(GCGE_VEC),x是解向量(GCGE_VEC)
//用时，V_tmp+1
void GCGE_CG(void *Matrix, void *b, void *x, GCGE_OPS *ops, GCGE_PARA *para, 
      void **V_tmp1, void **V_tmp2)
{
   //临时向量
   void        *r, *p, *w;
   GCGE_INT    niter = 0;
   GCGE_INT    max_it = para->cg_max_it, type = para->cg_type;
   GCGE_DOUBLE rate = para->cg_rate;
   GCGE_DOUBLE tmp1, tmp2, alpha, beta, rho1, rho2, error, last_error, pTw, wTw;

   //CG迭代中用到的临时向量
   ops->GetVecFromMultiVec(V_tmp1, 1, &r, ops); //记录残差向量
   ops->GetVecFromMultiVec(V_tmp2, 0, &p, ops); //记录下降方向
   ops->GetVecFromMultiVec(V_tmp2, 1, &w, ops); //记录tmp=A*p
   ops->MatDotVec(Matrix,x,r, ops); //tmp = A*x
   ops->VecAxpby(1.0, b, -1.0, r, ops);  
   ops->VecInnerProd(r, r, &rho2, ops);//用残量的模来判断误差
   error = sqrt(rho2);
   /* TODO */
   //这里应该判断以下如果error充分小就直接返回!!!
   //printf("the initial residual: %e\n",error);
   //ops->VecAxpby(1.0, r, 0.0, p);
   niter = 1;
   last_error = error;
   //max_it = 40;
   //rate = 1e-5;
   //printf("Rate = %e,  max_it = %d\n",rate,max_it);
   while((last_error/error >= rate)&&(niter<max_it))
   {
      if(niter == 1)
      {
         //set the direction as r
         ops->VecAxpby(1.0, r, 0.0, p, ops);
         //p = r;
      }
      else
      {
         //printf("come here!\n");
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
   ops->RestoreVecForMultiVec(V_tmp1, 1, &r, ops); 
   ops->RestoreVecForMultiVec(V_tmp2, 0, &p, ops);
   ops->RestoreVecForMultiVec(V_tmp2, 1, &w, ops);
}//end for the CG program




//统一进行块形式的CG迭代
//CG迭代求解 A * x =  RHS 
//W存储在 V(:,w_start:w_start+w_length), RHS存储在V_tmp(:,0:w_length)
// V_tmp = [r, p, w]的方式来组织
/**
 *  Matrix:    input,        A*x = RHS中的矩阵A
 *  RHS:       input,        前x_length个向量为要求解的x_length个方程组的右端项
 *  V:         input|output, 第x_start到x_start+x_length个向量为要求解的向量
 *                           input  为求解的初值
 *                           output 为得到的解
 *  x_start:   input,        V中解向量的起始位置
 *  x_length:  input,        要求解的方程组个数
 *  ops:       input,        操作
 *  para:      input,        参数
 *  V_tmp1:    input,        向量组工作空间
 *  V_tmp2:    input,        向量组工作空间
 *  subspace_dtmp: input,    double型工作空间
 *  subspace_itmp: input,    int型工作空间
 */
/**
 * TODO
 * 目前 BCG 存在的问题：
 *
 * 1. 测试时将rate取为1e-8, 第一个很容易就收敛了，
 *    后几个在达到1e-9或1e-7时就不再继续下降，保持不变，
 *    所以BCG过程也无法终止，第一个会在达到要求后继续提高精度
 *
 * 2. 当第一个解的精度很高时，如残差已达到1e-27, 继续算下去，解会变成nan
 * 
 * 3. 直接将精确解给它，会算错(解变成nan)
 *
 * 4. TODO 原因？不知
 *
 * 5. TODO 修改：
 *    如果确实是前面的收敛的快，当前面有的收敛了以后，只对后面进行操作
 *
 */
void GCGE_BCG(void *Matrix, void **RHS, void**V, GCGE_INT x_start, 
      GCGE_INT x_length, GCGE_INT max_it, GCGE_DOUBLE rate,
      GCGE_OPS *ops, void *V_tmp1, void **V_tmp2, void **V_tmp3,
      GCGE_DOUBLE *subspace_dtmp, GCGE_INT *subspace_itmp)
{
   //临时向量
   void        *x,  *r, *p, *w,  *b;
   //void      **CG_R = RHS, **CG_P = workspace->CG_p, **CG_W = workspace->evec, 
   void        **CG_R = RHS, **CG_P = V_tmp1, **CG_W = V_tmp2, 
               **CG_X = V;
   if(V_tmp3 != NULL)
   {
       //使用V_tmp3作为运行中使用的r
      CG_R = V_tmp3;
      GCGE_INT mv_s[2];
      GCGE_INT mv_e[2];
      mv_s[0] = 0;
      mv_e[0] = x_length;
      mv_s[1] = 0;
      mv_e[1] = x_length;
      ops->MultiVecAxpby(1.0, RHS, 0.0, V_tmp3, mv_s, mv_e, ops); 
   }
   GCGE_INT    id, idx, niter = 0;
   GCGE_DOUBLE tmp1, tmp2, alpha, beta;
   //GCGE_DOUBLE *rho1 = workspace->subspace_dtmp, 
   GCGE_DOUBLE *rho1  = subspace_dtmp, 
               *rho2  = rho1 + x_length, 
               *error = rho2 + x_length, 
               *last_error = error +x_length,
               *ptw = last_error + x_length,
               *rtr = ptw + x_length;
   //GCGE_INT *unlock = workspace->subspace_itmp, num_unlock = 0; 
   GCGE_INT *unlock = subspace_itmp, num_unlock = 0, old_num_unlock = 0; 

   //计算初始的残差向量和残差
   for(idx=0; idx<x_length; idx++)
   {
      ops->GetVecFromMultiVec(CG_X, x_start+idx, &x, ops);   //取出初始值 x      
      ops->GetVecFromMultiVec(CG_W, idx, &r, ops); //取出暂存残差r的变量
      ops->MatDotVec(Matrix,x,r, ops); //r = A*x
      ops->RestoreVecForMultiVec(CG_X, x_start+idx, &x, ops);
      // b = b - r;
      ops->GetVecFromMultiVec(CG_R, idx, &b, ops);    //取出右端项 b
      ops->VecAxpby(-1.0, r, 1.0, b, ops);  
      ops->VecLocalInnerProd(b, b, rho2+idx, ops);    //用残量的模来判断误差    
      ops->RestoreVecForMultiVec(CG_R, idx, &b, ops); //把b相应的位置设为残差
      ops->RestoreVecForMultiVec(CG_W, idx, &r, ops); //把r返回

      unlock[idx] = idx;
      num_unlock ++;
   }//算完初始的残差向量和残差
   //统一进行数据传输  
#if GCGE_USE_MPI
   MPI_Allreduce(MPI_IN_PLACE, rho2, x_length, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif 
    //最后计算每个向量的残差(初始残差，用来进行收敛性)	
   for(idx=0;idx<x_length;idx++)
      error[idx] = sqrt(rho2[idx]);
   /**
    *ops->MatDotVec(Matrix,x,r); //tmp = A*x
    ops->VecAxpby(1.0, b, -1.0, r);  
    ops->VecInnerProd(r, r, &rho2);//用残量的模来判断误差
    error = sqrt(rho2); */
   /* TODO */
   //这里应该判断以下如果error充分小就直接返回!!!
   //printf("the initial residual: %e\n",error);
   //ops->VecAxpby(1.0, r, 0.0, p);
   niter = 1;
   //last_error = error; 
   //max_it = 40;
   //rate = 1e-5;
   //printf("Rate = %e,  max_it = %d\n",rate,max_it);
   while((num_unlock > 0)&&(niter < max_it))
   {
     //对每一个为收敛的向量进行处理
     for(id = 0; id < num_unlock; id++)
     {
       idx = unlock[id];
       ops->GetVecFromMultiVec(CG_P, idx, &p, ops);   //取出向量 p
       ops->GetVecFromMultiVec(CG_R, idx, &r, ops);   //取出右端项 r
       
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
	  //p = r + beta * p
	  ops->VecAxpby(1.0, r, beta, p, ops);	  
	 }////end for if(niter == 1)
	 ops->RestoreVecForMultiVec(CG_R, idx, &r, ops);
	 //compute the vector w = A*p和p^Tw = ptw
	 ops->GetVecFromMultiVec(CG_W, idx, &w, ops);   //取出 w
	 ops->MatDotVec(Matrix,p,w,ops);  //w = A*p
	 //做局部内积（在每个进程内部做局部的内积）
	 ops->VecLocalInnerProd(p,w,ptw+id, ops);
	 ops->RestoreVecForMultiVec(CG_P, idx, &p, ops);
	 ops->RestoreVecForMultiVec(CG_W, idx, &w, ops);       
     }//end for id
     //这里我们把两个向量内积放在一起计算，这样可以减少一次的向量内积的全局通讯，提高可扩展性
     //compute the value pTw = p^T * w
     //应该要写一个专门的子程序来做这个统一的内积
     //与张宁商量一下如何实现
#if GCGE_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, ptw, num_unlock, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif     
      //现在来计算: alpha = rho2/ptw, x = alpha*p +x , r = -alpha*w + r, 并且同时计算r的局部内积
      for(id=0;id<num_unlock;id++)
      { 
         idx = unlock[id];
         //计算alpha
         alpha = rho2[id]/ptw[id];
         //compute the new solution x = alpha * p + x
         ops->GetVecFromMultiVec(CG_P, idx, &p, ops);   //取出 p
         ops->GetVecFromMultiVec(CG_X, x_start+idx, &x, ops); //取出初始值 x 
         //  x = alpha*p +x 
         ops->VecAxpby(alpha, p, 1.0, x, ops);
         ops->RestoreVecForMultiVec(CG_P, idx, &p, ops); 
         ops->RestoreVecForMultiVec(CG_X, x_start+idx, &x, ops);

         //compute the new r and residual
         ops->GetVecFromMultiVec(CG_R, idx, &r, ops);   //取出 r
         ops->GetVecFromMultiVec(CG_W, idx, &w, ops);   //取出 w
         //r = -alpha*w + r
         ops->VecAxpby(-alpha, w, 1.0, r, ops);
         //set rho1 as rho2
         rho1[id] = rho2[id];
	  //计算新的r的局部内积
         ops->VecLocalInnerProd(r, r, rho2+id, ops);       
         ops->RestoreVecForMultiVec(CG_R, idx, &r, ops); 
         ops->RestoreVecForMultiVec(CG_W, idx, &w, ops);
      }//end for idx
      //统一进行数据传输  
#if GCGE_USE_MPI
      MPI_Allreduce(MPI_IN_PLACE, rho2, num_unlock, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

      for(id=0;id<num_unlock;id++)
      {
	idx = unlock[id];
	//error是要每个向量都存储的，只是我们之需要更新未收敛向量的误差就可以
	last_error[idx] = sqrt(fabs(rho2[id]));  
      }
      //下面进行收敛性的判断
      old_num_unlock = num_unlock;
      num_unlock = 0;
      for(id=0;id< old_num_unlock;id ++)
      {   
          idx = unlock[id];
         if(last_error[idx]/error[idx] >= rate)
         {
            unlock[num_unlock] = idx;
            num_unlock ++;
         }//end if(last_error[idx]/error[idx] >= rate)       
         //GCGE_Printf("niter: %d, i: %d, error=%10.5e, last_error=%10.5e,  last_error/error= %10.5e, rate: %e\n",
         //niter, idx, error[idx], last_error[idx], last_error[idx]/error[idx], rate);
      }//end for id        
      //printf("  niter= %d,  The current residual: %10.5f\n",niter, last_error);
      //update the iteration time
      niter++;       
      //printf("\n");   
   }//end while((last_error/error >= rate)&&(niter<max_it))
}//end for this subprogram
