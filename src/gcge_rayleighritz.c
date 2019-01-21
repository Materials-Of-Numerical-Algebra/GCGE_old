/*
 * =====================================================================================
 *
 *       Filename:  gcge_rayleighritz.c
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

#include "gcge_rayleighritz.h"

/* brief: 计算Rayleigh-Ritz过程中的子空间矩阵V^T*A*V中的大规模操作
 *        subspace_matrix = [X,P,W]^T * A * [W] = XAW
 *                                                PAW
 *                                                WAW
 *
 *   需要 submat, ldm, W_start, W_end, V_tmp(临时存储A*W)
 *
 */
//计算subspace_matrix = V^T*A*W
void GCGE_ComputeSubspaceMatrixVTAW(void **V, void *A, GCGE_INT start_W, GCGE_INT end_V, GCGE_DOUBLE *subspace_mat, GCGE_OPS *ops, void **workspace)
{
    //start[0],end[0]表示V中[X,P]的起始与终点位置
    GCGE_INT w_length = end_V - start_W;
    GCGE_INT xp_length = end_V - w_length;

    GCGE_INT mv_s[2];
    GCGE_INT mv_e[2];

    mv_s[0] = start_W;
    mv_e[0] = end_V;
    mv_s[1] = 0;
    mv_e[1] = w_length;
    //mv_s,mv_e表示每个多向量操作函数中，两个向量组中操作向量的起始和终点位置
    
    //比如下面表示对V的mv_s[0]到mv_e[0]-1位置的向量进行操作
    //    对workspace的mv_s[1]到mv_e[1]-1位置的向量进行操作
    
    //计算 workspace(0:w_length) = A * W(start[1]:end[1])
    //这里workspace 为大规模向量组临时空间
    ops->MatDotMultiVec(A, V, workspace, mv_s, mv_e, ops);

    mv_s[0] = 0;
    mv_e[0] = xp_length;
    mv_s[1] = 0;
    mv_e[1] = w_length;

    //计算 subspace_mat = V * workspace
    //这里计算 X(AW) 部分
    //         P(AW)
    //需要 subspace_mat 的 leading dimension, 这里为end_V
    ops->MultiVecInnerProd(V, workspace, subspace_mat, "nonsym", mv_s, mv_e, end_V, 0, ops);

    //double *vTAw = subspace_mat;

    mv_s[0] = start_W;
    mv_e[0] = end_V;
    mv_s[1] = 0;
    mv_e[1] = w_length;

    //计算 subspace_mat = W * workspace, 且已知 subspace_mat 是对称矩阵
    //这里计算 W(AW) 部分
    //需要 subspace_mat 的 leading dimension, 这里为end_V
    //     以及子空间矩阵存放的起始位置 subspace_mat+xp_length
    ops->MultiVecInnerProd(V, workspace, subspace_mat + xp_length, "sym", mv_s, mv_e, end_V, 0, ops);

}


//计算子空间矩阵subspace_matrix=V^TAV
//A是计算子空间矩阵时用到的矩阵Ａ(GCGE_MATRIX), V是用到的向量组(GCGE_Vec)
//void GCGE_GetSubspaceMatrix(void *A, void **V, GCGE_DOUBLE *subspace_matrix, 
/* brief: 计算Rayleigh-Ritz过程中的子空间矩阵V^T*A*V
 *        subspace_matrix = [X,P,W]^T * A * [X,P,W] = XAX  XAP  XAW
 *                                                    PAX  PAP  PAW
 *                                                    WAX  WAP  WAW
 *   1. 小规模操作,需要 coef, ldc, AA_last, p_end, 
 *                      ldm(新矩阵subspace_matrix的规模leading dimension)
 *
 *        XAX  XAP  = XX  PX ^T * AA_last *  XX  PX
 *        PAX  PAP    XP  PP                 XP  PP
 *                    XW  PW                 XW  PW
 *
 *      进行优化计算如下：
 *
 *        XAX = \Lambda
 *        XAP = 0
 *        PAX = 0
 *        PAP = PX ^T * AA_last *  PX
 *              PP                 PP
 *              PW                 PW
 *
 *   2. 大规模操作，计算 XAW , 需要 submat, ldm, W_start, W_length,
 *                       PAW        V_tmp(临时存储A*W)
 *                       WAW
 *
 */
/* TODO need change input */
void GCGE_ComputeSubspaceMatrix(void *A, void **V, 
        GCGE_OPS *ops, GCGE_PARA *para, GCGE_WORKSPACE *workspace)
{
    //ldm: laplack的风格名字：表示subspace_matrix的维数
    GCGE_INT    i, dim_xp = workspace->dim_xp, ldm = workspace->dim_xpw;
    GCGE_DOUBLE *subspace_dtmp   = workspace->subspace_dtmp; 
    GCGE_DOUBLE *subspace_matrix = workspace->subspace_matrix;
    GCGE_DOUBLE *subspace_evec   = workspace->subspace_evec;
#if 0
    //对子空间矩阵的 XX,XP,PP 部分全部使用稠密矩阵相乘来计算
    if(dim_xp != 0)
    {
        //lde表示AA_last的行数(leading dimension of subspace_evec)
        //AA_last: 上一次迭代的AA矩阵的行数
        GCGE_INT    lde = workspace->last_dim_xpw; 
        GCGE_DOUBLE alpha = 1.0, beta = 0.0;
        // 这里只是取了一个临时空间用于存储下式
        // AP表示 AA_last *  XX  PX 的起始位置, 
        //                   XP  PP
        //                   XW  PW
        // AA_last此时存储在subspace_matrix中
        // dsymm 计算AP,因为AA_last是对称矩阵，所以使用dsymm计算对称矩阵乘矩阵
        // 需要的参数有 AA_last 的行数 lde，XX PX 的总列数 dim_xp
        //DenseMatDotDenseMat的三个矩阵中是否有可覆盖的可能（为了减少内存开销）
        ops->DenseSymMatDotDenseMat("L", "U", &lde, &dim_xp, &alpha, subspace_matrix, 
                &lde, workspace->subspace_evec, &lde, &beta, subspace_dtmp, &lde);
        //dgemm 计算下式
        //  XAX  XAP  = XX  PX ^T * AP
        //  PAX  PAP    XP  PP    
        //              XW  PW    
        // 需要的参数有 左矩阵的行数 lde, 右矩阵的列数 dim_xp, 
        // 左矩阵的列数(右行) lde (这里需要添加几个参数命名 TODO)
        memset(subspace_matrix, 0.0, ldm*ldm*sizeof(GCGE_DOUBLE));
        ops->DenseMatDotDenseMat("T", "N", &dim_xp, &dim_xp, &lde, &alpha, 
                workspace->subspace_evec, &lde, subspace_dtmp, 
                &lde, &beta, subspace_matrix, &ldm);

    }
#else
    //应用子空间矩阵的结构,XX部分为对角阵,XP部分为0,PP部分使用稠密矩阵相乘来计算
    if(dim_xp != 0)
    {
        GCGE_INT    lde   = workspace->last_dim_xpw; //lde表示subspace_evec的行数
        GCGE_INT    dim_x = workspace->dim_x; //dim_x为当前X向量的个数
        GCGE_INT    dim_p = dim_xp - dim_x; //dim_p为当前P向量的个数
        GCGE_DOUBLE alpha = 1.0;
        GCGE_DOUBLE beta  = 0.0;
        GCGE_DOUBLE *pp   = subspace_evec+dim_x*lde; //pp为subspace_evec中P的线性组合系数
        GCGE_DOUBLE *AApp = subspace_dtmp; //AApp为子空间矩阵AA乘以P的线性组合系数pp
        GCGE_DOUBLE *PTAP = subspace_matrix+dim_x*ldm+dim_x; //PTAP为子空间矩阵P^TAP的相应部分
        /*  PTAP 表示下式中 PAP 的起始位置
         *
         *  XAX  XAP  XAW     XX  PX  WX ^T              XX  PX  WX
         *  PAX  PAP  PAW  =  XP  PP  WP    * AA_last *  XP  PP  WP
         *  WAX  WAP  WAW     XW  PW  WW                 XW  PW  WW
         *
         *  AApp 表示下式的起始位置
         *
         *                   PX
         *  AApp = AA_last * PP
         *                   PW
         *
         *  pp 表示下式的起始位置
         *
         *       PX
         *  pp = PP
         *       PW
         */
        //调用 BLAS-3 的 dsymm 计算 AApp = subspace_matrix * pp
        ops->DenseSymMatDotDenseMat("L", "U", &lde, &dim_p, &alpha, subspace_matrix, 
                &lde, pp, &lde, &beta, AApp, &lde);
        //将subspace_matrix赋值为0
        memset(subspace_matrix, 0.0, ldm*ldm*sizeof(GCGE_DOUBLE));
        //调用 BLAS-3 的 dgemm 计算 PTAP = pp^T * AApp
        ops->DenseMatDotDenseMat("T", "N", &dim_p, &dim_p, &lde, &alpha, 
                pp, &lde, subspace_dtmp, 
                &lde, &beta, PTAP, &ldm);
        //将XTAX部分赋值为特征值对角阵
        for(i=0; i<dim_x; i++)
        {
            subspace_matrix[i*ldm+i] = workspace->eval[i];
        }

    }
#endif
    GCGE_ComputeSubspaceMatrixVTAW(V, A, dim_xp, ldm, subspace_matrix+dim_xp*ldm, 
      ops, workspace->evec);
}

/* brief: 计算subspace_matrix的特征对
 *    
 *    需要的参数：subspace_matrix的规模: ldm
 *                要计算特征值的范围：il,iu(计算第il到第iu个特征对)
 *
 *    这里 ops->DenseMatEigenSolver 的形式参数给了 dsyevx和dsyevr 用到的
 *    所有参数，工作空间给了两个函数需要的最大空间，
 *    默认选择dsyevx，可以在使用时设置使用哪一种求解方式
 */
#if 0
void GCGE_ComputeSubspaceEigenpairs(GCGE_DOUBLE *subspace_matrix, 
        GCGE_DOUBLE *eval, GCGE_DOUBLE *subspace_evec, 
        GCGE_OPS *ops, GCGE_PARA *para, GCGE_WORKSPACE *workspace)
{
#if GET_PART_TIME
    GCGE_DOUBLE t1 = GCGE_GetTime();
#endif
    GCGE_INT    ldm = workspace->dim_xpw;
    GCGE_INT    info;
    GCGE_DOUBLE *temp_matrix = workspace->subspace_dtmp,
                *dwork_space = temp_matrix + ldm*ldm,
                vl = 0.0, vu = 0.0, abstol = 0.0;
    //abstol = 2*dlamch_("S");
    GCGE_INT    max_dim_x = workspace->max_dim_x;
    //计算subspace_matrix的前rr_eigen_start+1到iu个特征值
    GCGE_INT    i = 0;
    GCGE_INT    rr_eigen_start = workspace->unlock[0];
    if(para->opt_rr_eig_partly == 0)
        rr_eigen_start = 0;
    GCGE_INT    il = rr_eigen_start; 
    //从unlock[0]开始往前判断eval[unlock[0]]这个特征值的重数
    for(i=rr_eigen_start-1; i>-1; i--)
    {
        if(fabs(eval[i] - eval[rr_eigen_start])/eval[rr_eigen_start] < para->multi_tol_for_lock)
        {
            il -= 1;
        }
    }
    rr_eigen_start = il;
    il += 1;
    workspace->num_soft_locked_in_last_iter = rr_eigen_start;
    //rr_eigen_start = 0;
    GCGE_INT    iu = (max_dim_x < ldm) ? max_dim_x : ldm;
    GCGE_INT    m = iu-il+1;
    GCGE_INT    lwork = 8*ldm;
    GCGE_INT    liwork = 5*ldm;
    GCGE_INT    *subspace_itmp = workspace->subspace_itmp;
    //subspace_itmp[liwork]在dsyevr时存储liwork
    GCGE_INT    *isuppz = subspace_itmp + 10*ldm; 
    GCGE_INT    *ifail  = subspace_itmp + 5*ldm; 
    for(i=0; i<(workspace->dim_xpw - workspace->dim_xp); i++)
	memset(subspace_matrix+(workspace->dim_xp+i)*ldm, 0.0, rr_eigen_start*sizeof(GCGE_DOUBLE));
    memcpy(temp_matrix, subspace_matrix, ldm*ldm*sizeof(GCGE_DOUBLE));
    
    GCGE_INT    nrows = ldm, ncols = ldm;
    //以下N代表ldm
    //work给足够大的空间: max(26*N, 1+6*N+2*N**2)
    //lwork: dsyevr(26*N),dsyevx(8*N),dsyev(3*N),dsyevd(1+6*N+2*N**2)
    //workspace_dtmp空间需要；N*N+lwork
    //iwork足够大的空间：10*N
    //liwork: dysevr(10*N),dsyevx(5*N,没有这个参数),
    //        dsyev(没有这个参数，且不需要iwork),dsyevd(3+5*N)
    //isuppz: dsyevr(2*N的空间)
    //ifail: dsyevx(N的空间）
    //特征值与特征向量的存储位置都向后移rr_eigen_start
    
    GCGE_INT rank = 0;
    if(para->opt_bcast == 1)
    {
        if(para->use_mpi_bcast)
        {
#if GCGE_USE_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        }
    }
    //此时, 如果使用 MPI 且要用 MPI_Bcast, 那么 非0号 进程 rank != 0, 
    //此时 0号 进程 rank==0, 如果 不用MPI_Bcast 也是 rank != 0
    //那么, 如果 rank==0, 就计算特征值问题, 否则不用计算, 等待广播
    if(rank == 0)
    {
#if 0
        ops->DenseMatEigenSolver("V", "I", "U", &nrows, temp_matrix, &ncols, 
                &vl, &vu, &il, &iu, &abstol, 
                &m, eval+rr_eigen_start, subspace_evec+rr_eigen_start*ldm, &ldm, 
                isuppz, dwork_space, &lwork,
                subspace_itmp, &liwork, ifail, &info);
#else
	nrows -= rr_eigen_start;
	il -= rr_eigen_start;
	iu -= rr_eigen_start;
        ops->DenseMatEigenSolver("V", "I", "U", &nrows, 
	        temp_matrix+rr_eigen_start*ncols+rr_eigen_start, &ncols, 
                &vl, &vu, &il, &iu, &abstol, 
                &m, eval+rr_eigen_start, 
		subspace_evec+rr_eigen_start*ldm+rr_eigen_start, &ldm, 
                isuppz, dwork_space, &lwork,
                subspace_itmp, &liwork, ifail, &info);
	iu += rr_eigen_start;
#endif
    }
    if(para->use_mpi_bcast)
    {
#if GCGE_USE_MPI
        memcpy(subspace_evec+iu*ldm, eval+rr_eigen_start, m*sizeof(GCGE_DOUBLE));
        GCGE_Printf("use MPI_Bcast\n");
        MPI_Bcast(subspace_evec+rr_eigen_start*ldm, m*ldm+m, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
        memcpy(eval+rr_eigen_start, subspace_evec+iu*ldm, m*sizeof(GCGE_DOUBLE));
#endif
    }
#if GET_PART_TIME
    GCGE_DOUBLE t2 = GCGE_GetTime();
    para->stat_para->part_time_one_iter->rr_eigen_time = t2-t1;
    para->stat_para->part_time_total->rr_eigen_time += t2-t1;
#endif
#if 0
    if(rr_eigen_start > 0)
    {
        //sub_end表示这里要进行正交化的x个数
        GCGE_INT sub_end = max_dim_x - rr_eigen_start;

        memset(subspace_evec, 0.0, rr_eigen_start*ldm*sizeof(GCGE_DOUBLE));
        for(i=0; i<rr_eigen_start; i++)
            subspace_evec[i*ldm+i] = 1.0;
        for(i=rr_eigen_start; i<max_dim_x; i++)
            memset(subspace_evec+i*ldm, 0.0, rr_eigen_start*sizeof(GCGE_DOUBLE));
        if(strcmp(para->x_orth_type, "scbgs") == 0)
        {
            GCGE_BlockOrthonormalizationInSubspace(subspace_evec + rr_eigen_start*ldm +
                        rr_eigen_start, ldm, ldm - rr_eigen_start, &sub_end, 
                        para->orth_para->x_orth_block_size,
                        ops, para, workspace->subspace_dtmp);
        }
        else if(strcmp(para->x_orth_type, "bgs") == 0)
        {
            GCGE_BOrthonormalizationInSubspace(subspace_evec+rr_eigen_start*ldm+rr_eigen_start, 
                        ldm, ldm - rr_eigen_start, 0, &sub_end, 
                        NULL, -1, para->orth_para, workspace->subspace_dtmp, ops);
        }
        else
        {
            GCGE_OrthonormalizationInSubspace(subspace_evec+rr_eigen_start*ldm+rr_eigen_start, 
                        ldm, ldm - rr_eigen_start, 0, &sub_end, 
                        NULL, -1, para->orth_para, ops);
        }
    }
#endif
    //用lapack_syev计算得到的特征值是按从小到大排列,
    //如果用A内积，需要把特征值取倒数后再按从小到大排列
    //由于后面需要用到的只有前dim_x个特征值，所以只把前dim_x个拿到前面来
    if(strcmp(para->orth_type, "A") == 0)
    {
        GCGE_SortEigenpairs(eval, subspace_evec, iu, ldm, workspace->subspace_dtmp);
    }
#if GET_PART_TIME
    GCGE_DOUBLE t3 = GCGE_GetTime();
    para->stat_para->part_time_one_iter->x_orth_time = t3-t2;
    para->stat_para->part_time_total->x_orth_time += t3-t2;
#endif
}
#else
void GCGE_ComputeSubspaceEigenpairs(GCGE_DOUBLE *subspace_matrix, 
        GCGE_DOUBLE *eval, GCGE_DOUBLE *subspace_evec, 
        GCGE_OPS *ops, GCGE_PARA *para, GCGE_WORKSPACE *workspace)
{
#if GET_PART_TIME
    GCGE_DOUBLE t1 = GCGE_GetTime();
#endif
    GCGE_INT    ldm = workspace->dim_xpw;
    GCGE_INT    info;
    GCGE_DOUBLE *temp_matrix = workspace->subspace_dtmp,
                *dwork_space = temp_matrix + ldm*ldm,
                vl = 0.0, vu = 0.0, abstol = 0.0;
    //abstol = 2*dlamch_("S");
    GCGE_INT    max_dim_x = workspace->max_dim_x;
    //计算subspace_matrix的前rr_eigen_start+1到iu个特征值
    GCGE_INT    i = 0;
    GCGE_INT    rr_eigen_start = workspace->unlock[0];
    if(para->opt_rr_eig_partly == 0)
        rr_eigen_start = 0;
    GCGE_INT    il = rr_eigen_start; 
    //从unlock[0]开始往前判断eval[unlock[0]]这个特征值的重数
    for(i=rr_eigen_start-1; i>-1; i--)
    {
        if(fabs(eval[i] - eval[rr_eigen_start])/eval[rr_eigen_start] < para->multi_tol_for_lock)
        {
            il -= 1;
        }
    }
    rr_eigen_start = il;
    il += 1;
    workspace->num_soft_locked_in_last_iter = rr_eigen_start;
    //rr_eigen_start = 0;
    GCGE_INT    iu = (max_dim_x < ldm) ? max_dim_x : ldm;
    if(strcmp(para->eval_type, "sm") == 0)
    {
        //如果要求的是模最小的特征值，那么要求到最大的一个特征值
	iu = ldm;
    }
    GCGE_INT    m = iu-il+1;
    GCGE_INT    lwork = 8*ldm;
    GCGE_INT    liwork = 5*ldm;
    GCGE_INT    *subspace_itmp = workspace->subspace_itmp;
    //subspace_itmp[liwork]在dsyevr时存储liwork
    GCGE_INT    *isuppz = subspace_itmp + 10*ldm; 
    GCGE_INT    *ifail  = subspace_itmp + 5*ldm; 
    for(i=0; i<(workspace->dim_xpw - workspace->dim_xp); i++)
        memset(subspace_matrix+(workspace->dim_xp+i)*ldm, 0.0, rr_eigen_start*sizeof(GCGE_DOUBLE));
    memcpy(temp_matrix, subspace_matrix, ldm*ldm*sizeof(GCGE_DOUBLE));
    
    GCGE_INT    nrows = ldm, ncols = ldm;
    //以下N代表ldm
    //work给足够大的空间: max(26*N, 1+6*N+2*N**2)
    //lwork: dsyevr(26*N),dsyevx(8*N),dsyev(3*N),dsyevd(1+6*N+2*N**2)
    //workspace_dtmp空间需要；N*N+lwork
    //iwork足够大的空间：10*N
    //liwork: dysevr(10*N),dsyevx(5*N,没有这个参数),
    //        dsyev(没有这个参数，且不需要iwork),dsyevd(3+5*N)
    //isuppz: dsyevr(2*N的空间)
    //ifail: dsyevx(N的空间）
    //特征值与特征向量的存储位置都向后移rr_eigen_start
    
    GCGE_INT rank = 0;
    GCGE_INT n_proc = 1;
    GCGE_INT j = 0;

    if(para->opt_allgatherv == 1)
    {
        if(para->opt_bcast == 1)
        {
            if(para->use_mpi_bcast)
            {
#if GCGE_USE_MPI
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
#endif
            }
        }
        //实际要计算的部分矩阵的行数以及要计算该矩阵的第几个特征值
        nrows -= rr_eigen_start;
        il -= rr_eigen_start;
        iu -= rr_eigen_start;
       
        //平均每个进程要计算几个特征值, 先取上整数
        GCGE_INT eigen_num_per_rank = (m+n_proc-1)/n_proc;
        //取下整数的话，多出来几个特征值（也就是前几个进程要比下整数多算一个）
        GCGE_INT num_more_eig = m - n_proc*(eigen_num_per_rank-1);
        //old_m存储实际要计算的特征值的总个数
        GCGE_INT old_m = m;
        //实际每个进程要计算的特征值个数，以及第几个
        if(rank < num_more_eig)
        {
            il += rank*eigen_num_per_rank;
            iu = il+eigen_num_per_rank-1;
            m = eigen_num_per_rank;
        }
        else
        {
            il += rank*(eigen_num_per_rank-1)+num_more_eig;
            iu = il+eigen_num_per_rank-2;
            m = eigen_num_per_rank-1;
        }
        //本进程要计算的第一个特征值，实际是大矩阵的第几个特征值
        GCGE_INT real_eigen_start = il-1 + rr_eigen_start;
        //正常计算该进程需要计算的特征对，放在特征对应该在的位置
        //如果rr_eigen_start==0, 将特征值和特征向量分两次进行消息传输
        if(m > 0)
        {
            ops->DenseMatEigenSolver(
                    temp_matrix+rr_eigen_start*ncols+rr_eigen_start, 
                    ncols, nrows, 
                    eval+real_eigen_start, 
                    subspace_evec+real_eigen_start*ldm+rr_eigen_start, ldm, 
                    il, iu, subspace_itmp, dwork_space);
        }
#if GCGE_USE_MPI
        memcpy(temp_matrix, eval+real_eigen_start, m*sizeof(GCGE_DOUBLE));
        for(i=0; i<m; i++)
        {
            memcpy(temp_matrix+m+i*nrows, subspace_evec+real_eigen_start*ldm+i*ldm+rr_eigen_start, nrows*sizeof(GCGE_DOUBLE));
        }
        //rr_eigen_start至少是1
        //发送数据的指针，发送数据量，发送数据类型
        //接收数据的指针，接收数据量，在存储位置的偏移
        //接收数据类型，通信子
        GCGE_INT recv_counts[n_proc];
        GCGE_INT recv_displs[n_proc];
        //发送特征向量时, 每个进程要发送的个数以及应该放置的位置
        for(i=0; i<num_more_eig; i++)
        {
            recv_counts[i] = eigen_num_per_rank*(nrows+1);
        }
        for(i=num_more_eig; i<n_proc; i++)
        {
            recv_counts[i] = (eigen_num_per_rank-1)*(nrows+1);
        }
        recv_displs[0] = 0;
        for(i=1; i<n_proc; i++)
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
        GCGE_DOUBLE *recv_workspace;
        //subspace_dtmp空间放大到2*ldm*ldm
        recv_workspace = temp_matrix+recv_displs[1];
        //printf("recv_displs: %d\n", recv_displs[1]+recv_displs[n_proc-1]+recv_counts[n_proc-1]);
        //全收集特征向量
        //GCGE_Printf("use MPI_Allgatherv\n");
        MPI_Allgatherv(temp_matrix, recv_counts[rank], MPI_DOUBLE, 
              recv_workspace, recv_counts, recv_displs, MPI_DOUBLE, MPI_COMM_WORLD);
        //现在表示每个进程计算的特征对个数
        GCGE_INT recv_nevec[n_proc];
        for(i=0; i<n_proc; i++)
        {
            recv_counts[i] = recv_counts[i]/(nrows+1);
            recv_nevec[i]  = recv_displs[i]/(nrows+1);
        }
        //0进程内容, 拷贝特征值
        memcpy(eval+rr_eigen_start, recv_workspace, recv_counts[0]*sizeof(GCGE_DOUBLE));
        //0进程内容, 拷贝特征向量
        for(i=0; i<recv_counts[0]; i++)
        {
            memcpy(subspace_evec+(rr_eigen_start+i)*ldm+rr_eigen_start, 
                  recv_workspace+recv_counts[0]+i*nrows, nrows*sizeof(GCGE_DOUBLE));
        }
        for(i=1; i<n_proc; i++)
        {
            //先拷贝特征值
            memcpy(eval+rr_eigen_start+recv_nevec[i], recv_workspace+recv_displs[i], recv_counts[i]*sizeof(GCGE_DOUBLE));
            //拷贝特征向量
            for(j=0; j<recv_counts[i]; j++)
            {
                memcpy(subspace_evec+(rr_eigen_start+recv_nevec[i]+j)*ldm+rr_eigen_start, 
                      recv_workspace+recv_displs[i]+recv_counts[i]+j*nrows, nrows*sizeof(GCGE_DOUBLE));
            }
        }
#endif
    }
    else
    {
        if(para->opt_bcast == 1)
        {
            if(para->use_mpi_bcast)
            {
#if GCGE_USE_MPI
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
            }
        }
        //此时, 如果使用 MPI 且要用 MPI_Bcast, 那么 非0号 进程 rank != 0, 
        //此时 0号 进程 rank==0, 如果 不用MPI_Bcast 也是 rank != 0
        //那么, 如果 rank==0, 就计算特征值问题, 否则不用计算, 等待广播
        nrows -= rr_eigen_start;
        if(rank == 0)
        {
            il -= rr_eigen_start;
            iu -= rr_eigen_start;
            ops->DenseMatEigenSolver( temp_matrix+rr_eigen_start*ncols+rr_eigen_start, 
                    ncols, nrows, eval+rr_eigen_start, 
                    subspace_evec+rr_eigen_start*ldm+rr_eigen_start, ldm, 
                    il, iu, subspace_itmp, dwork_space);
            iu += rr_eigen_start;
        }
        if(para->use_mpi_bcast)
        {
#if GCGE_USE_MPI
            memcpy(subspace_evec+iu*ldm, eval+rr_eigen_start, m*sizeof(GCGE_DOUBLE));
            MPI_Bcast(subspace_evec+rr_eigen_start*ldm, m*ldm+m, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
            memcpy(eval+rr_eigen_start, subspace_evec+iu*ldm, m*sizeof(GCGE_DOUBLE));
#endif
        }
    }
    if(strcmp(para->eval_type, "sm") == 0)
    {
        //如果要求的是模最小的特征值，求解所有特征对后, 
	//将模最小的(总的)rr_eigen_start到dim_x个特征对排到前面
	//首先将eval(rr_eigen_start:ldm-1)中模最小的(max_dim_x-rr_eigen_start)个取出相应的编号
	//将这部分eval进行按模从小到大排序, 并将对应的编号存入subspace_itmp
	//然后将模最小的特征值对应的特征向量取出来放到dwork_space, 再统一拷贝回给subspace_evec
        GCGE_INT     eval_length = ((max_dim_x < ldm) ? max_dim_x : ldm) - rr_eigen_start;
	GCGE_DOUBLE *evec_start = subspace_evec+rr_eigen_start*ldm+rr_eigen_start;

	for(i=0; i<nrows; i++)
	{
	    subspace_itmp[i] = i;
	}
	GCGE_SortByMagnitude(eval+rr_eigen_start, subspace_itmp, 0, nrows-1);
	//实际取出来的evec是一个方阵, eval的个数也是这么多
	for(i=0; i<eval_length; i++)
	{
	    memcpy(temp_matrix+i*ldm, evec_start+subspace_itmp[i]*ldm, nrows*sizeof(GCGE_DOUBLE));
	}
	memcpy(evec_start, temp_matrix, ((eval_length-1)*ldm+nrows)*sizeof(GCGE_DOUBLE));
    }
#if GET_PART_TIME
    GCGE_DOUBLE t2 = GCGE_GetTime();
    para->stat_para->part_time_one_iter->rr_eigen_time = t2-t1;
    para->stat_para->part_time_total->rr_eigen_time += t2-t1;
#endif
    //用lapack_syev计算得到的特征值是按从小到大排列,
    //如果用A内积，需要把特征值取倒数后再按从小到大排列
    //由于后面需要用到的只有前dim_x个特征值，所以只把前dim_x个拿到前面来
    if(strcmp(para->orth_type, "A") == 0)
    {
        GCGE_SortEigenpairs(eval, subspace_evec, iu, ldm, workspace->subspace_dtmp);
    }
#if GET_PART_TIME
    GCGE_DOUBLE t3 = GCGE_GetTime();
    para->stat_para->part_time_one_iter->x_orth_time = t3-t2;
    para->stat_para->part_time_total->x_orth_time += t3-t2;
#endif
}
#endif

//将这部分a进行按模从小到大排序, 并将对应的编号存入idx
//即按a的大小也对int型的idx进行排序
void GCGE_SortByMagnitude(GCGE_DOUBLE *a, GCGE_INT *idx, GCGE_INT left, GCGE_INT right)
{
    GCGE_INT i = left; 
    GCGE_INT j = right;
    GCGE_INT    idx_temp = idx[left];
    GCGE_DOUBLE temp   = a[left];
    GCGE_DOUBLE m_temp = fabs(a[left]);
    if(left > right)
        return;
    while(i != j)
    {
        //如果模有这样的大小关系，就交换值
        while((fabs(a[j]) >= m_temp)&&(j > i))
            j--;
        if(j > i)
	{
	    idx[i] = idx[j];
            a[i++] = a[j];
	}
        while((fabs(a[i]) <= m_temp)&&(j > i))
            i++;
        if(j > i)
	{
	    idx[j] = idx[i];
            a[j--] = a[i];
	}
    }
    a[i] = temp;
    idx[i] = idx_temp;
    GCGE_SortByMagnitude(a, idx, left, i-1);
    GCGE_SortByMagnitude(a, idx, i+1, right);
    return;
}


//用A内积的话，特征值从小到大，就是原问题特征值倒数的从小到大，
//所以顺序反向，同时特征值取倒数
void GCGE_SortEigenpairs(GCGE_DOUBLE *eval, GCGE_DOUBLE *evec, GCGE_INT nev, 
    				GCGE_INT ldv, GCGE_DOUBLE *work)
{
    GCGE_INT head = 0, tail = ldv-1;
    GCGE_DOUBLE tmp;
    for( head=0; head<nev; head++ )
    {
        tail = ldv-1-head;
        if(head < tail)
        {
            memcpy(work, evec+head*ldv, ldv*sizeof(GCGE_DOUBLE));
            memcpy(evec+head*ldv, evec+tail*ldv, ldv*sizeof(GCGE_DOUBLE));
            memcpy(evec+tail*ldv, work, ldv*sizeof(GCGE_DOUBLE));
            tmp = eval[head];
            eval[head] = 1.0/eval[tail];
            eval[tail] = 1.0/tmp;
        }
        else
        {
            break;
        }
    }
}

/*
GCGE_DOUBLE GCGE_ModuleMaxDouble(GCGE_DOUBLE *a, GCGE_INT n)
{
    GCGE_INT i = 0;
    GCGE_DOUBLE max = a[0];
    for(i=1; i<n; i++)
    {
        if(fabs(max+ 1e-16) < fabs(a[i]))
            max = a[i];
    }
    return max;
}

*/
