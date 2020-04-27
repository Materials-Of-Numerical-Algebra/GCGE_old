/*
 * =====================================================================================
 *
 *       Filename:  gcge_eigsol.c
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
#include <string.h>
#include <math.h>

#include "gcge_eigsol.h"

/**
 * @brief 计算 A * evec = eval * B * evec 
 *    
 *    TODO 考虑X正交化时，如果有相关性，重新选随机数,至少要报错（初始化失败）
 *
 *    Initialization: 
 *       对evec设定随机初值,把evec复制给X
 *       对X正交化,在X张成的子空间中计算RR问题
 *       子空间矩阵subspace_matrix, 子空间特征向量subspace_evec, 
 *       特征值workspace->eval
 *       检查收敛性，得到unlock,num_unlock
 *
 *       对未收敛的X计算W
 *       对W进行正交化
 *       在[X,W]张成的子空间中计算RR问题
 *       检查收敛性，得到unlock,num_unlock
 *       
 *    Loop: 
 *       对未收敛的X计算P
 *       对未收敛的X计算W
 *       对W进行正交化
 *       在[X,W]张成的子空间中计算RR问题
 *       检查特征值重数，调整X的长度
 *       检查收敛性，得到unlock,num_unlock
 *
 *    Finalization:
 *       把X复制给evec
 *       把workspace->eval复制给eval
 *
 * @param A
 * @param B
 * @param eval
 * @param evec
 * @param para
 * @param ops
 * @param workspace
 */
//求解特征值调用GCGE_Eigen函数
//A,B表示要求解特征值的矩阵GCGE_MATRIX，evec是要求解的特征向量GCGE_Vec
void GCGE_Solve(void *A, void *B, GCGE_DOUBLE *eval, void **evec, 
        GCGE_PARA *para, GCGE_OPS *ops, GCGE_WORKSPACE *workspace)
{
    //para->shift = 1.5;
    //para->if_shift = 1;
    /*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
     * 注释中小括号的内容为针对这个参数的解释 
     *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
    GCGE_PrintParaInfo(para);
    GCGE_INT    nev       = para->nev;        //nev表示要求解的特征值个数，用户需在GCGE_SOLVER_Setup前给定
    GCGE_INT    ev_max_it = para->ev_max_it;  //ev_max_it表示GCGE算法的最大迭代次数
    GCGE_INT    max_dim_x = workspace->max_dim_x; //max_dim_x表示判断特征值重数的最大取值空间
    GCGE_INT    num_init_evec = para->num_init_evec; //初始给定的特征向量的个数
    void        **V       = workspace->V;  //V表示大规模的工作空间，V=[X,P,W]
    //void        **RitzVec = workspace->RitzVec; //RitzVec在计算中临时存储Ritz向量（大规模近似特征向量）
    GCGE_DOUBLE *subspace_matrix = workspace->subspace_matrix; //subspace_matrix表示Rayleigh-Ritz过程中的子空间矩阵
    GCGE_DOUBLE *subspace_evec   = workspace->subspace_evec;   //subspace_evec表示Rayleigh-Ritz过程中的子空间向量
    GCGE_DOUBLE *subspace_dtmp   = workspace->subspace_dtmp;  
    void        *Orth_mat; //Orth_mat正交化时计算内积所用的矩阵: B 或者 A，通常是B
    void        *RayleighRitz_mat; //RayleighRitz_mat计算子空间矩阵时所用矩阵: A或者B,通常用A
    
    GCGE_DOUBLE value_1 = 0.0;
    GCGE_DOUBLE value_2 = 0.0;
    GCGE_DOUBLE max_ip = 0.0;
    GCGE_INT    max_i = 0;
    GCGE_INT    max_j = 0;

    workspace->evec = evec;

    GCGE_STATISTIC_PARA *stat_para = para->stat_para;
    //t1,t2用于统计时间
    GCGE_DOUBLE t1 = 0.0;
    GCGE_DOUBLE t2 = 0.0;

    GCGE_INT    i = 0;
    GCGE_INT    j = 0;
    //因为计算中对矩阵乘向量和向量内积数乘等线性代数操作，
    //都尽量采用向量组操作,mv_s与mv_e表示计算中需要对向量组中的
    //第mv_s到mv_e个向量进行操作,一次操作中可能用到两个向量组，所以是二维数组
    //mv_s:matrix-vector operation start index
    //mv_e:matrix-vector operation end index + 1
    GCGE_INT    mv_s[2] = { 0, 0 };
    GCGE_INT    mv_e[2]   = { 0, 0 };
    //如果使用B内积，那么用矩阵B进行正交化，用矩阵A计算子空间矩阵
    //如果使用A内积，那么用矩阵A进行正交化，用矩阵B计算子空间矩阵
    //strcmp函数：当两个字符串相同时返回0, 如果orth_type==B, 返回0,运行else
    if(strcmp(para->orth_type, "B"))
    {
        //这个是表示不是用矩阵B做内积的情况， 因为 strcmp(para->orth_type, "B")不为零的时候表示的
        //orth_type不是B
        Orth_mat = A;
        RayleighRitz_mat = B;
    }
    else
    {   //表示的使用矩阵B做内积的情况
        Orth_mat = B;
        RayleighRitz_mat = A;
    }
    //这几个参数用于计算矩阵A的Omega范数(如果收敛准则为"O")
    double init_norm = 0.0;
    double omega_F_norm = 0.0;
    double A_Omega_F_norm = 0.0;
    double A_Omega_norm = 0.0;
    double F_norm = 0.0;
    void  *omega_vec;
    //如果用户不给初值，这里给随机初值，用户需要提供给向量设随机值的函数
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    if(num_init_evec < nev)
    {
        do{
            //调用方式: MultiVecSetRandomValue(void** V, start, num_vecs, ops)
            //其中V是存储向量的对象， start：表示是从V中的start开始形成num_vecs个随机向量
            //V(:,start:start + num_vecs) 进行随机化
            ops->MultiVecSetRandomValue(evec, num_init_evec, nev - num_init_evec, ops);
            if(strcmp(para->conv_type, "O") == 0)
            {
                if(num_init_evec == 0)
                {
                    omega_F_norm = 0.0;
                    for(i=0; i<nev; i++)
                    {
                        ops->GetVecFromMultiVec(evec, i, &omega_vec, ops);
                        ops->VecInnerProd(omega_vec, omega_vec, &init_norm, ops);
                        ops->VecAxpby(0.0, omega_vec, 1.0/sqrt(init_norm), omega_vec, ops);
                        ops->RestoreVecForMultiVec(evec, i, &omega_vec, ops);
                    }
                    for(i=0; i<nev; i++)
                    {
                        ops->GetVecFromMultiVec(evec, i, &omega_vec, ops);
                        ops->VecInnerProd(omega_vec, omega_vec, &init_norm, ops);
                        ops->RestoreVecForMultiVec(evec, i, &omega_vec, ops);
                        omega_F_norm += init_norm;
                    }
 
                    mv_s[0] = 0;
                    mv_e[0] = nev;
                    mv_s[1] = 0;
                    mv_e[1] = nev;
                    ops->MatDotMultiVec(A, evec, V, mv_s, mv_e, ops);
                    //此时，Omega = V(:,0:nev-1), A*Omega = evec(:,0:nev-1)
                    A_Omega_F_norm = 0.0;
                    for(i=0; i<nev; i++)
                    {
                        ops->GetVecFromMultiVec(V, i, &omega_vec, ops);
                        ops->VecInnerProd(omega_vec, omega_vec, &F_norm, ops);
                        ops->RestoreVecForMultiVec(V, i, &omega_vec, ops);
                        A_Omega_F_norm += F_norm;
                    }
                    A_Omega_norm = sqrt(A_Omega_F_norm/omega_F_norm);
                    //GCGE_Printf("A_Omega_norm: %e\n", A_Omega_norm);
                    para->conv_omega_norm = A_Omega_norm;
                }
            }
 
            if(para->dirichlet_boundary == 1)
            {
                //理边界条件，把Dirichlet边界条件的位置强制设置为 0
                ops->SetDirichletBoundary(evec,nev,A,B);    
            }//end for SetDirichletBoundary
            //进行正交化, 并返回正交化向量的个数
            //对初始近似特征向量做B正交化, 
            //正交化这里修改了, 此时 dim_x = x_end
            //x_end: 表示X的终止位置
            //dim_x(x_end)表示 X 中的向量个数，初始为nev(参数dim_x需要被替换为x_end, x_start始终为0)
            workspace->dim_x = nev;
            // 对 V(:,0:dim_x)进行正交化,正交矩阵 Orth_mat,V_tmp:用来做正交的辅助向量,一般只用V_tmp[0]
            //subspace_dtmp: 用来记录不同的两个向量之间的内积 (是一个数组)
            if(strcmp(para->x_orth_type, "multi") == 0)
            {
                GCGE_StableMultiOrthonormalization(evec, num_init_evec, 
                      &(workspace->dim_x), Orth_mat, ops, para, workspace);
            }
            else if(strcmp(para->x_orth_type, "scbgs") == 0)
            {
                GCGE_SCBOrthonormalization(evec, num_init_evec, 
		      &(workspace->dim_x), Orth_mat, ops, para, workspace);
            }
            else if(strcmp(para->x_orth_type, "cbgs") == 0)
            {
                GCGE_CBOrthonormalization(evec, num_init_evec, 
                      &(workspace->dim_x), Orth_mat, ops, para, workspace);
            }
            else
            {
                GCGE_Orthonormalization(evec, num_init_evec, &(workspace->dim_x), Orth_mat, 
                      ops, para, workspace->V_tmp, workspace->subspace_dtmp);
            }
 
            num_init_evec = workspace->dim_x;
            //GCGE_Printf("num_init_evec = %d\n",num_init_evec);
        }while(num_init_evec < nev);
    }//end for Initialization for the eigenvectors   
    else
    {
        if(strcmp(para->x_orth_type, "multi") == 0)
        {
            GCGE_StableMultiOrthonormalization(evec, para->num_conv, 
                  &(workspace->dim_x), Orth_mat, ops, para, workspace);
        }
        else if(strcmp(para->x_orth_type, "scbgs") == 0)
        {
            GCGE_SCBOrthonormalization(evec, para->num_conv, 
	          &(workspace->dim_x), Orth_mat, ops, para, workspace);
        }
        else if(strcmp(para->x_orth_type, "cbgs") == 0)
        {
            GCGE_CBOrthonormalization(evec, para->num_conv, 
                  &(workspace->dim_x), Orth_mat, ops, para, workspace);
        }
        else
        {
            GCGE_Orthonormalization(evec, para->num_conv, 
                  &(workspace->dim_x), Orth_mat, ops, para, 
                  workspace->V_tmp, workspace->subspace_dtmp);
        }
    }
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->x_orth_time = t2-t1;
    stat_para->part_time_total->x_orth_time += t2-t1;
#endif

    //把用户提供的evec初值copy给V, 从evec的0到nev-1拷贝到V的0到nev-1
    mv_s[0] = 0;
    mv_e[0] = nev;
    mv_s[1] = 0;
    mv_e[1] = nev;
    //把初始特征向量evec赋值给V的前nev个向量组: V(:,1:nev) = evec
    //即: V(:,mv_s[1]:mv_e[1]) = evec(:,mv_s[0]:mv_e[0]);
    ops->MultiVecAxpby(1.0, evec, 0.0, V, mv_s, mv_e, ops);

    /**
    //对初始近似特征向量做B正交化, 
    //正交化这里修改了, 此时 dim_x = x_end
    //x_end: 表示X的终止位置
    //dim_x(x_end)表示 X 中的向量个数，初始为nev(参数dim_x需要被替换为x_end, x_start始终为0)
    workspace->dim_x = nev;
    // 对 V(:,0:dim_x)进行正交化,正交矩阵 Orth_mat,V_tmp:用来做正交的辅助向量,一般只用V_tmp[0]
    //subspace_dtmp: 用来记录不同的两个向量之间的内积 (是一个数组)
    GCGE_Orthonormalization(V, 0, &(workspace->dim_x), Orth_mat, ops, para->orth_para, 
    workspace->V_tmp, workspace->subspace_dtmp);
    //默认初始近似特征向量是线性无关的，否则报错
    if(workspace->dim_x < nev)
    {
    printf("The initial eigenvectors is linearly dependent!\n");
    ops->MultiVecSetRandomValue(evec, nev, ops);
    exit(0);
    }
     **/
    //dim_xpw 表示V=[X,P,W]的总向量个数，此时V中只有X, 此时应该有 dim_xpw = dim_x. 
    // ??应被替换为 w_end??
    workspace->dim_xpw = workspace->dim_x;
    //printf("Before start: dim_x=%d\n",workspace->dim_x);
    //last_dim_x表示上次迭代中X向量的个数，用于计算P时确定拷贝的起始位置
    //last_dim_x需要被替换为xx_nrows,表示下面矩阵中XX矩阵的行数
    //计算P时： XX   O       XX   O      
    //          XP   O  ==>  XP   XP 
    //          XW   O       XW   XW 
    //last_dim_x 这个表示什么意思？不是很好的一个名字？？？？？
    //last_dim_x: 上一次迭代X中的向量个数
    workspace->last_dim_x = workspace->dim_x;
    //printf("Before start: last_ dim_x=%d\n",workspace->last_dim_x);
    //dim_xp表示[X,P]的向量个数
    //也表示计算子空间矩阵时大规模操作的起始位置（目前是表示这个意思）
    //dim_xp需要被修改为三个参数：p_end与rrls_start(Rayleigh-Ritz large scale)
    //以及w_start(W向量组在V中存放的起始位置）
    //这里dim_xp赋值为0表示rrls_start,即计算子空间矩阵时大规模操作的起始位置
    workspace->dim_xp = 0;
    //计算得到子空间矩阵subspace_matrix=V^TAV
    //这个函数修改了workspace->subspace_matrix,没有修改参数
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_ComputeSubspaceMatrix(RayleighRitz_mat, V, ops, para, workspace);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->rr_mat_time = t2-t1;
    stat_para->part_time_total->rr_mat_time += t2-t1;
#endif
    //计算子空间矩阵的特征对
    GCGE_ComputeSubspaceEigenpairs(subspace_matrix, workspace->eval, subspace_evec, 
            ops, para, workspace);
    //基向量 V 线性组合得到dim_x(x_end)个向量 X(RitzVec), 线性组合系数为 subspace_evec
    //RitzVec = V * subspace_evec
    //由于线性组合计算时不能覆盖基向量组V，所以需要一个工作空间RitzVec
    //是因为计算P的时候需要用到，这个时候是不是 X 部分可以改变
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_ComputeX(V, subspace_evec, evec, nev, workspace->dim_xpw, ops, workspace);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->x_axpy_time = t2-t1;
    stat_para->part_time_total->x_axpy_time += t2-t1;
#endif
    //检查收敛性
    //检查X(RitzVec)向量的收敛性，这个函数会修改 unlock(未收敛的特征对编号）
    //unconv_bs(当前批次中未收敛的特征对个数),num_unlock(总的未收敛个数）
    //此函数会修改 num_iter (加1)
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_CheckConvergence(A, B, workspace->eval, evec, ops, para, workspace);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->conv_time = t2-t1;
    stat_para->part_time_total->conv_time += t2-t1;
#endif
    GCGE_PrintIterationInfo(workspace->eval, para);
    //printf("after: sum_res=%e\n",para->sum_res);

    if(para->num_conv_continuous < nev)
    {
    //Ritz向量放到V中作为下次迭代中基向量的一部分,即为[X,P,W]中的X
    mv_s[0] = workspace->num_soft_locked_in_last_iter;
    mv_s[1] = workspace->num_soft_locked_in_last_iter;
    mv_e[0] = nev;
    mv_e[1] = nev;
    //基底向量组V已经没用，将RitzVec中的X向量与V进行指针交换，V中存储新的基向量
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    ops->MultiVecSwap(V, evec, mv_s, mv_e, ops);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->x_axpy_time = t2-t1;
    stat_para->part_time_total->x_axpy_time += t2-t1;
#endif
    //这里修改dim_xp用于表示 W 向量存放的起始位置为dim_x，此时 V=[X,W]
    workspace->dim_xp = workspace->dim_x;
    //用未收敛的特征向量作为 X 求解线性方程组 A W = \lambda B X
    //需要参数unlock及w_length(unconv_bs),w_start,
    //修改向量组V中w_start至w_end列，不会修改参数
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_ComputeW(A, B, V, workspace->eval, ops, para, workspace);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->w_line_time = t2-t1;
    stat_para->part_time_total->w_line_time += t2-t1;
#endif
    //last_dim_xpw表示计算子空间矩阵时，小规模操作时使用的上次子空间矩阵的维数
    //dim_xpw会在正交化时被修改，所以这里先更新last_dim_xpw
    //last_dim_xpw需要修改为rr_last_ldm(Rayleigh-Ritz中上次迭代子空间矩阵的leading dimension)
    workspace->last_dim_xpw = workspace->dim_xpw;
    //dim_xpw(即w_end)
    workspace->dim_xpw = workspace->dim_xp + workspace->unconv_bs;
    //对基向量组V的第dim_x列到dim_xpw列进行正交化
    //即为对W部分进行正交化,dim_x这里表示w_start,dim_xpw这里表示w_end
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    if(strcmp(para->w_orth_type, "multi") == 0)
    {
        GCGE_StableMultiOrthonormalization(V, workspace->dim_x, 
              &(workspace->dim_xpw), Orth_mat, ops, para, workspace);
    }
    else if(strcmp(para->w_orth_type, "cbgs") == 0)
    {
        GCGE_CBOrthonormalization(V, workspace->dim_x, 
	      &(workspace->dim_xpw), Orth_mat, ops, para, workspace);
    }
    else if(strcmp(para->w_orth_type, "scbgs") == 0)
    {
        GCGE_SCBOrthonormalization(V, workspace->dim_x, 
	      &(workspace->dim_xpw), Orth_mat, ops, para, workspace);
    }
    else
    {
        GCGE_Orthonormalization(V, workspace->dim_x, &(workspace->dim_xpw), 
	      Orth_mat, ops, para, workspace->V_tmp, workspace->subspace_dtmp);
    }
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->w_orth_time = t2-t1;
    stat_para->part_time_total->w_orth_time += t2-t1;
#endif
    //这里dim_xp表示计算子空间矩阵时大规模操作的起始位置(即rrls_start=0)
    workspace->dim_xp = nev;
    //workspace->dim_xp = 0;
    //计算子空间矩阵subspace_matrix = V^T*A*V
    //这里修改了workspace->subspace_matrix,没有修改参数
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_ComputeSubspaceMatrix(RayleighRitz_mat, V, ops, para, workspace);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->rr_mat_time = t2-t1;
    stat_para->part_time_total->rr_mat_time += t2-t1;
#endif
    //计算子空间矩阵的特征对
    GCGE_ComputeSubspaceEigenpairs(subspace_matrix, workspace->eval, subspace_evec, 
            ops, para, workspace);
    //基向量 V 线性组合得到dim_x个向量 X(RitzVec), 线性组合系数为 subspace_evec
    //RitzVec = V * subspace_evec
    //由于线性组合计算时不能覆盖基向量组V，所以需要一个工作空间RitzVec
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_ComputeX(V, subspace_evec, evec, nev, workspace->dim_xpw, ops, workspace);
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->x_axpy_time = t2-t1;
    stat_para->part_time_total->x_axpy_time += t2-t1;
#endif
    //检查收敛性
    //检查X(RitzVec)向量的收敛性，这个函数会修改 unlock(未收敛的特征对编号）
    //unconv_bs(当前批次中未收敛的特征对个数),num_unlock(总的未收敛个数）
    //并将num_iter加1
    //这里为什么用unconv_bs???
#if GET_PART_TIME
    t1 = GCGE_GetTime();
#endif
    GCGE_CheckConvergence(A, B, workspace->eval, evec, ops, para, workspace);
    //para->shift = 1.5;
    //para->if_shift = 1;
#if GET_PART_TIME
    t2 = GCGE_GetTime();
    stat_para->part_time_one_iter->conv_time = t2-t1;
    stat_para->part_time_total->conv_time += t2-t1;
#endif
    //做完第一步[X,W]之后得到的特征值
    GCGE_PrintIterationInfo(workspace->eval, para);
    }
    //--------------------开始循环--------------------------------------------
    while((para->num_conv_continuous < nev)&&(para->num_iter < ev_max_it))
    {
        //brief: 计算P的子空间系数，并进行小规模正交化,再线性组合得到P
        //       XX  O  拷贝  XX  O   正交化  XX  PX  线性组合         PX
        //       XP  O  ===>  XP  XP  =====>  XP  PP  =======> P = V * PP
        //       XW  O        XW  XW          XW  PW                   PW
        //out:   输出时大规模P向量存储在V的dim_x(p_start)到dim_xp(p_end)列
        //会修改dim_xp即p_end
        //GCGE_Printf("before orth, dim_x: %d, dim_xp: %d, dim_xpw: %d\n", 
	//      workspace->dim_x, workspace->dim_x+workspace->unconv_bs, 
	//      workspace->dim_x+2*workspace->unconv_bs);
        GCGE_ComputeP(subspace_evec, V, ops, para, workspace);

        //Ritz向量放到V中作为下次迭代中基向量的一部分
        mv_s[0] = workspace->num_soft_locked_in_last_iter;
        mv_s[1] = workspace->num_soft_locked_in_last_iter;
        mv_e[0] = nev;
        mv_e[1] = nev;
        //计算完P之后，基向量组V就没用了,将X放到V中,操作方法为：
        //交换V的前dim_x(x_end)列与X(RitzVec)的前dim_x(x_end)列的指针
        //把V中与X相对应的向量组更新
#if GET_PART_TIME
        t1 = GCGE_GetTime();
#endif
        ops->MultiVecSwap(V, evec, mv_s, mv_e, ops);
#if GET_PART_TIME
        t2 = GCGE_GetTime();
        stat_para->part_time_one_iter->x_axpy_time = t2-t1;
        stat_para->part_time_total->x_axpy_time += t2-t1;
#endif

        //用未收敛的特征向量作为X求解线性方程组 A W = \lambda B X
        //需要参数unlock及w_length(unconv_bs),w_start,
        //修改向量组V中w_start至w_end列，不会修改参数
#if GET_PART_TIME
        t1 = GCGE_GetTime();
#endif
        GCGE_ComputeW(A, B, V, workspace->eval, ops, para, workspace);
#if GET_PART_TIME
        t2 = GCGE_GetTime();
        stat_para->part_time_one_iter->w_line_time = t2-t1;
        stat_para->part_time_total->w_line_time += t2-t1;
#endif
        //last_dim_xpw表示计算子空间矩阵时，小规模操作时使用的上次子空间矩阵的维数
        //dim_xpw会在正交化时被修改，所以这里先更新last_dim_xpw
        //last_dim_xpw需要修改为rr_last_ldm(Rayleigh-Ritz中上次迭代子空间矩阵的leading dimension)
        workspace->last_dim_xpw = workspace->dim_xpw;
        //dim_xpw(即w_end)
        workspace->dim_xpw = workspace->dim_xp + workspace->unconv_bs;
        //对基向量组V的第dim_xp列到dim_xpw列进行正交化
        //即为对W部分进行正交化,dim_xp这里表示w_start,dim_xpw这里表示w_end
        //printf("before: sum_res=%e\n",para->sum_res);
#if GET_PART_TIME
        t1 = GCGE_GetTime();
#endif
        if(strcmp(para->w_orth_type, "bgs") == 0)
        {
            GCGE_BOrthonormalization(V, workspace->dim_xp, &(workspace->dim_xpw), Orth_mat, ops, para,
                workspace);
        }
        else if(strcmp(para->w_orth_type, "cbgs") == 0)
        {
            GCGE_CBOrthonormalization(V, workspace->dim_xp, &(workspace->dim_xpw), Orth_mat, ops, para,
                workspace);
        }
        else if(strcmp(para->w_orth_type, "scbgs") == 0)
        {
            GCGE_SCBOrthonormalization(V, workspace->dim_xp, &(workspace->dim_xpw), Orth_mat, ops, para,
                workspace);
        }
        else if(strcmp(para->w_orth_type, "multi") == 0)
        {
            GCGE_StableMultiOrthonormalization(V, workspace->dim_xp, 
                  &(workspace->dim_xpw), Orth_mat, ops, para, workspace);
        }
        else
        {
            //GCGE_Orthonormalization(V, 0, &(workspace->dim_xpw), Orth_mat, ops, para,
            GCGE_Orthonormalization(V, workspace->dim_xp, &(workspace->dim_xpw), Orth_mat, ops, para,
                workspace->V_tmp, workspace->subspace_dtmp);
        }
#if GET_PART_TIME
        t2 = GCGE_GetTime();
        stat_para->part_time_one_iter->w_orth_time = t2-t1;
        stat_para->part_time_total->w_orth_time += t2-t1;
#endif
        //计算子空间矩阵subspace_matrix = V^T*A*V
        //这里修改了workspace->subspace_matrix,没有修改参数
#if GET_PART_TIME
        t1 = GCGE_GetTime();
#endif
        //GCGE_Printf("after Worth, dim_x: %d, dim_xp: %d, dim_xpw: %d\n", 
	//      workspace->dim_x, workspace->dim_xp, workspace->dim_xpw);
        GCGE_ComputeSubspaceMatrix(RayleighRitz_mat, V, ops, para, workspace);

#if GET_PART_TIME
        t2 = GCGE_GetTime();
        stat_para->part_time_one_iter->rr_mat_time = t2-t1;
        stat_para->part_time_total->rr_mat_time += t2-t1;
#endif
        //计算子空间矩阵的特征对
        GCGE_ComputeSubspaceEigenpairs(subspace_matrix, workspace->eval, subspace_evec, 
                ops, para, workspace);

        workspace->last_dim_x = workspace->dim_x;
        //检查特征值重数,确定新的dim_x(即x_end)
        //这里是用什么方法进行检测的？？？
	workspace->dim_x = max_dim_x;
        //GCGE_CheckEvalMultiplicity(nev, nev, max_dim_x, &(workspace->dim_x), workspace->eval);
        //para->num_unlock = workspace->dim_x;
        //基向量 V 线性组合得到dim_x个向量 X(RitzVec), 线性组合系数为 subspace_evec
        //RitzVec = V * subspace_evec
        //由于线性组合计算时不能覆盖基向量组V，所以需要一个工作空间RitzVec
#if GET_PART_TIME
        t1 = GCGE_GetTime();
#endif

        GCGE_ComputeX(V, subspace_evec, evec, nev, workspace->dim_xpw, ops, workspace);

#if 0
	//检查正交性
        mv_s[0] = 0;
        mv_s[1] = 0;
        mv_e[0] = para->num_conv_continuous;
        mv_e[1] = para->num_conv_continuous;
        ops->MultiVecInnerProd(V, V, subspace_dtmp, "sym", mv_s, mv_e, para->num_conv_continuous, 0, ops);
	max_ip = 0.0;
	max_i = 0;
	max_j = 0;
	for(i=0; i<para->num_conv_continuous; i++)
	{
	    for(j=0; j<para->num_conv_continuous; j++)
	    {
	        if(i!=j)
		{
		if(max_ip < subspace_dtmp[i*para->num_conv_continuous+j])
		{
		    max_ip = subspace_dtmp[i*para->num_conv_continuous+j];
			max_i = i;
			max_j = j;
		}
		}
#if 0
	        if((i!=j)&&(subspace_dtmp[i*nev+j] > 1e-14))
		{
		    GCGE_Printf("inner_prod(%d, %d) = %e;\n", i+1, j+1, subspace_dtmp[i*nev+j]);
		}
#endif
	    }
	}
        GCGE_Printf("max_ip: %e, max_i: %d, j: %d\n", max_ip, max_i, max_j);
#if 0
	if(max_ip > 5e-14)
	{
	    para->if_shift = 0;
	}
#endif
#endif
#if GET_PART_TIME
        t2 = GCGE_GetTime();
        stat_para->part_time_one_iter->x_axpy_time = t2-t1;
        stat_para->part_time_total->x_axpy_time += t2-t1;
#endif
        //检查收敛性
        //检查X(RitzVec)向量的收敛性，这个函数会修改 unlock(未收敛的特征对编号）
        //unconv_bs(当前批次中未收敛的特征对个数),num_unlock(总的未收敛个数）
        //并将num_iter加1
#if GET_PART_TIME
        t1 = GCGE_GetTime();
#endif
        GCGE_CheckConvergence(A, B, workspace->eval, evec, ops, para, workspace);
#if GET_PART_TIME
        t2 = GCGE_GetTime();
        stat_para->part_time_one_iter->conv_time = t2-t1;
        stat_para->part_time_total->conv_time += t2-t1;
#endif
        GCGE_PrintIterationInfo(workspace->eval, para);

    }//GCG算法迭代结束


    //把计算得到的近似特征对拷贝给eval,evec输出
    memcpy(eval, workspace->eval, nev*sizeof(GCGE_DOUBLE));

    mv_s[0] = 0;
    mv_e[0] = workspace->num_soft_locked_in_last_iter;
    mv_s[1] = 0;
    mv_e[1] = workspace->num_soft_locked_in_last_iter;
    ops->MultiVecAxpby(1.0, V, 0.0, evec, mv_s, mv_e, ops);

    GCGE_PrintFinalInfo(eval, para);

}

/*  brief: 检查(eval, X)的收敛性,返回收敛信息num_unlock,unconv_bs,unlock
 *     
 *     对unlock中的num_unlock个编号的特征向量计算残差，判断收敛性
 *     允许收敛的特征对不连续
 *     一旦出现连续10个特征对未收敛，就默认后面的都未收敛，不全部检测
 *      
 *     需要的参数：unlock (in \ out) : 未收敛的编号
 *                 num_unlock (in \ out) : 未收敛的特征对个数
 *                 unconv_bs (out) : 本批次中未收敛的特征对个数
 *                 res (out) : 记录残差
 *
 */
//计算残差，检查收敛性，并获取未收敛的特征对编号及个数
//A,B是用来计算残差的矩阵(GCGE_MATRIX), evec是特征向量(GCGE_Vec)
//unlock中所包含的未收敛的特征值有可能是为了保证收敛速度而额外算得，所以要跟nev的关系弄清楚
void GCGE_CheckConvergence(void *A, void *B, GCGE_DOUBLE *eval, void **evec, 
        GCGE_OPS *ops, GCGE_PARA *para, 
        GCGE_WORKSPACE *workspace)
{
    GCGE_INT    i = 0, idx = 0; //idx表示当前正在计算的特征对编号
    GCGE_INT    *unlock = workspace->unlock;//存储未收敛的编号
    GCGE_INT    max_conv_idx = unlock[0]-1;//已收敛特征值的最大编号
    GCGE_INT    start = unlock[0]; //从第start个开始计算残差
    GCGE_INT    dim_x = workspace->dim_x; //表示X中的特征向量个数
    GCGE_INT    flag = 0; //flag==0表示前面都是连续收敛的
    GCGE_INT    min_multi_idx = 0;//记录当前idx的重特征值的最小编号
    GCGE_INT    num_unlock = 0;//unlock的特征对个数
    GCGE_INT    num_conv_continuous = unlock[0]; //连续收敛的特征对个数
    GCGE_INT    max_ind = 0, min_ind = 0;
    GCGE_DOUBLE multi_error = 0.0; //表示当前特征值与前一个特征值的误差
    GCGE_DOUBLE multi_tol = para->multi_tol_for_lock; //判定重根的阈值
    GCGE_DOUBLE res_norm, evec_norm, residual;//用于计算残差
    GCGE_DOUBLE max_res = 0.0, min_res = 0.0, sum_res = 0.0;
    GCGE_DOUBLE ev_tol = para->ev_tol; //判定是否收敛的阈值
    GCGE_DOUBLE *res = para->res; //存储残差
    char        *conv_type = para->conv_type;
    void        *tmp1, *tmp2, *ev;
    ops->GetVecFromMultiVec(workspace->V_tmp, 0, &tmp1, ops);
    ops->GetVecFromMultiVec(workspace->V_tmp, 1, &tmp2, ops);
    unlock[0] = -1;//unlock[0]初始化为-1备用
    GCGE_DOUBLE abs_residual_tmp = 0.0;
    for(idx=start; idx<para->nev; idx++ )
    {
        ops->GetVecFromMultiVec(evec, idx, &ev, ops);
        //计算残量
        ops->MatDotVec(A, ev, tmp1, ops);
        if(B == NULL)
        {
            ops->VecAxpby(-eval[idx], ev, 1.0, tmp1, ops);
        }
        else
        {
            ops->MatDotVec(B, ev, tmp2, ops);
            ops->VecAxpby(-eval[idx], tmp2, 1.0, tmp1, ops);
        }
        //tmp1是残差向量
        ops->VecInnerProd(tmp1, tmp1, &res_norm, ops);
        res_norm = sqrt(res_norm);
        ops->VecInnerProd(ev, ev, &evec_norm, ops);
        evec_norm = sqrt(evec_norm);
        ops->RestoreVecForMultiVec(evec, idx, &ev, ops);
        //绝对残差
        residual = res_norm/evec_norm;
	abs_residual_tmp = residual;
        if(strcmp(conv_type, "R") == 0)
        {
            //相对残差
            residual /= (1.0>fabs(eval[idx])?1.0:fabs(eval[idx]));
        }
        else if(strcmp(conv_type, "O") == 0)
        {
            //残差的Omega范数
            residual /= (para->conv_omega_norm + fabs(eval[idx]));
        }
        res[idx] = residual;

        if((residual < ev_tol)&&(abs_residual_tmp < 1e-2))
        {
            //该特征对收敛
            if(idx>0)
            {
                if(res[idx-1] < ev_tol)
                {
                    //前面一个也收敛, 那么本特征对收敛
                    max_conv_idx = idx;
                    if(flag == 0)
                    {
                        //更新连续收敛的特征值个数
                        num_conv_continuous = idx+1;
                    }
                }//end for if(res[idx-1] < ev_tol)
                else
                {
                    //前面一个未收敛, 那么检查是否重根
                    multi_error = fabs(eval[idx]-eval[idx-1])/fabs(eval[idx-1]);
                    if(multi_error < multi_tol)
                    {
                        //是重根，本特征对也不收敛
                        unlock[num_unlock++] = idx;
                    }//end for if(multi_error < multi_tol)
                    else
                    {
                        //与前面不是重根，本特征对收敛
                        max_conv_idx = idx;
                    }//end for else(if(multi_error < multi_tol))
                }//end for else(if(res[idx-1] < ev_tol))
            }//end for if(idx>0)
            else
            {
                //idx==0
                max_conv_idx = 0;
                num_conv_continuous = 1;
            }//end for else(if(idx>0))
        }//end for if(residual < ev_tol)
        else
        {
            //本特征对未收敛
            flag = 1;
            if(idx>0 && res[idx-1] < ev_tol)
            {
                //如果前面一个收敛，检查重根情况
                //min_multi_idx为上一个未收敛的之后的一个编号
                min_multi_idx = idx; 
                for(i=idx; i>0; i--)
                {
                    multi_error = fabs(eval[i]-eval[i-1])/fabs(eval[i-1]);
                    if(multi_error < multi_tol)
                    {
                        //那么从i-1开始都不是idx的重根
                        min_multi_idx = i-1;
                        //更新已收敛的最后一个编号
                        //max_conv_idx = i-1;
                    }
		    else
		    {
                        break;
		    }
                }
                for(i=min_multi_idx; i<=idx; i++)
                {
                    unlock[num_unlock++] = i;
                }
            }//end for if(res[idx-1] < ev_tol)
            else
            {
                //如果前面一个也不收敛
                unlock[num_unlock++] = idx;
            }//end for else(if(res[idx-1] < ev_tol))
        }//end for else(if(residual < ev_tol))
        sum_res += residual;
        if(idx == start)
        {
            max_res = residual;
            min_res = residual;
        }
        else
        {
            if(residual > max_res)
            {
                max_res = residual;
                max_ind = idx;
            }
            if(residual < min_res)
            {
                min_res = residual;
                min_ind = idx;
            }
        }//end if(j==0)
        //如果已有连续超过10个未收敛, 不再继续计算残差
        if(idx-max_conv_idx > 9)
        {
            break;
        }
    }//end for j
    ops->RestoreVecForMultiVec(workspace->V_tmp, 0, &tmp1, ops);
    ops->RestoreVecForMultiVec(workspace->V_tmp, 1, &tmp2, ops);

    //整个0到nev没有收敛特征对的个数
    //根据重数重新确定unconv_bs
    if(para->use_shift == 1)
    {
        para->if_shift = 1;
        para->shift = (eval[dim_x-1]-100.0*eval[0])/99.0;
    }
    //GCGE_Printf("shift: %f, eval[0]: %e\n", para->shift, eval[0]);
    //para->shift = 1.5;

#if 1
    if((num_conv_continuous<para->nev)&&(num_unlock<para->block_size))
    {
        if(unlock[num_unlock-1] < para->nev-1)
        {
            for(i=num_unlock; i<para->block_size; i++)
            {
                unlock[i] = unlock[i-1]+1;
                num_unlock += 1;
		//最大到了nev-1就要break
                if(unlock[i] > para->nev-2)
                {
                    break;
                }
            }
        }
    }
#endif

    if(num_conv_continuous < para->nev)
    {
        //如果还未全部收敛，且第一个未收敛的编号小于连续收敛的个数
        if(unlock[0] < num_conv_continuous)
	{
	    num_conv_continuous = unlock[0];
	}
    }

    //GCGE_Printf("num_unlock: %d, num_conv_continuous: %d\n", num_unlock, num_conv_continuous);
    //对已收敛的lock的特征值检查重数
    if(num_unlock > para->block_size)
    {
        num_unlock = para->block_size;
    }
#if 0
    for(i=0; i<num_unlock; i++)
    {
        GCGE_Printf("unlock[%d] = %d\n", i, unlock[i]);
    }
#endif
    para->num_unlock = num_unlock;
    para->num_conv_continuous = num_conv_continuous;
     //当前的BLOCK中没有收敛的个数
    workspace->unconv_bs = num_unlock;

    para->num_iter += 1;
    para->max_res = max_res;
    para->max_ind = max_ind;
    para->min_res = min_res;
    para->min_ind = min_ind;
    para->sum_res = sum_res;

}//end for this subprogram

/* brief:  从第nev个特征值开始检查重数，如果后一个特征值与前一个的比值在0.8到1.2之间，
 *         就认为可能是重特征值
 *
 *  需要的参数：需要检查的特征值的起始和终点位置
 *              start (in)
 *              end   (in \ out)
 */
//检查特征值重数，更新dim_x
void GCGE_CheckEvalMultiplicity(GCGE_INT start,GCGE_INT nev, GCGE_INT end, 
                                GCGE_INT *dim_x, GCGE_DOUBLE *eval)
{
    GCGE_INT tmp, i;
    tmp = start;
     if(start < nev)
     {
        printf("The parameter setting is wrong!\n");
        exit(0);
      }  
     
    //dsygv求出的特征值已经排序是ascending,从小到大
    //检查特征值的数值确定下次要进行计算的特征值个数
    for( i=start; i<end; i++ )
    {
        if((fabs(fabs(eval[tmp]/eval[nev])-1))<0.2)
            tmp += 1;
        else
            break;
    }
    //更新新的X的向量个数
    *dim_x = tmp;
    //*dim_x = end;
}

