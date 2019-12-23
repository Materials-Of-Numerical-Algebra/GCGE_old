#include "pase_amg.h" 

/* PASE_BMG 算法过程
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
 * 5. 递归调用 PASE_BMG, 层号为 coarse_level
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
void PASE_BMG( PASE_MULTIGRID mg, 
               PASE_INT current_level, 
               void **rhs, void **sol, 
               PASE_INT *start, PASE_INT *end,
               PASE_REAL tol, PASE_REAL rate, 
               PASE_INT nsmooth, PASE_INT max_coarsest_nsmooth)
{
    PASE_INT nlevel = mg->num_levels;
    //默认0层为最细层
    PASE_INT indicator = 1;
    // obtain the coarsest level
    PASE_INT coarsest_level = mg->coarsest_level;
    //if( indicator > 0 )
    //    coarsest_level = nlevel-1;
    //else
    //    coarsest_level = 0;
    //设置最粗层上精确求解的精度
    PASE_REAL coarest_rate = rate * 1e-10;
    void *A;
    PASE_INT mv_s[2];
    PASE_INT mv_e[2];
    void **residual = mg->cg_p[current_level];
    // obtain the 'enough' accurate solution on the coarest level
    //direct solving the linear equation
    if( current_level == coarsest_level )
    {
        A = mg->A_array[coarsest_level];
        GCGE_BCG(A, rhs, sol, start, end,  
                max_coarsest_nsmooth, coarest_rate, mg->gcge_ops, 
                mg->cg_p[coarsest_level], mg->cg_w[coarsest_level], 
                mg->cg_res[coarsest_level], 
                mg->cg_double_tmp, mg->cg_int_tmp);
	//GCGE_Printf("current_level: %d, after direct\n", current_level);
	//mg->gcge_ops->MultiVecPrint(sol, 1, mg->gcge_ops);
    }
    else
    {   
        A = mg->A_array[current_level];
        GCGE_BCG(A, rhs, sol, start, end, 
                nsmooth, rate, mg->gcge_ops, 
                mg->cg_p[current_level], mg->cg_w[current_level], 
		mg->cg_res[current_level], 
                mg->cg_double_tmp, mg->cg_int_tmp);
	//GCGE_Printf("current_level: %d, after presmoothing\n", current_level);
	//mg->gcge_ops->MultiVecPrint(sol, 1, mg->gcge_ops);

        mv_s[0] = start[1];
        mv_e[0] = end[1];
        mv_s[1] = start[2];
        mv_e[1] = end[2];
        //计算residual = A*sol
        mg->gcge_ops->MatDotMultiVec(A, sol, residual, mv_s, mv_e, mg->gcge_ops);
        //计算residual = rhs-A*sol
        mv_s[0] = start[0];
        mv_e[0] = end[0];
        mv_s[1] = start[2];
        mv_e[1] = end[2];
        mg->gcge_ops->MultiVecAxpby(1.0, rhs, -1.0, residual, mv_s, mv_e, mg->gcge_ops);

        // 把 residual 投影到粗网格
        PASE_INT coarse_level = current_level + indicator;
        void **coarse_residual = mg->rhs[coarse_level];
        mv_s[0] = start[2];
        mv_e[0] = end[2];
        mv_s[1] = start[0];
        mv_e[1] = end[0];
        PASE_INT error = PASE_MULTIGRID_FromItoJ(mg, current_level, coarse_level, 
                mv_s, mv_e, residual, coarse_residual);

        //求粗网格解问题，利用递归
        void **coarse_sol = mg->sol[coarse_level];
        mv_s[0] = start[1];
        mv_e[0] = end[1];
        mv_s[1] = start[1];
        mv_e[1] = end[1];
	//先给coarse_sol赋初值0
	mg->gcge_ops->MultiVecAxpby(0.0, coarse_sol, 0.0, coarse_sol, 
	        mv_s, mv_e, mg->gcge_ops);
        PASE_BMG(mg, coarse_level, coarse_residual, coarse_sol, 
                start, end, tol, rate, nsmooth, max_coarsest_nsmooth);
	//GCGE_Printf("current_level: %d, after postsmoothing\n", current_level);
	//mg->gcge_ops->MultiVecPrint(sol, 1, mg->gcge_ops);
        
        // 把粗网格上的解插值到细网格，再加到前光滑得到的近似解上
        // 可以用 residual 代替
        mv_s[0] = start[1];
        mv_e[0] = end[1];
        mv_s[1] = start[2];
        mv_e[1] = end[2];
        error = PASE_MULTIGRID_FromItoJ(mg, coarse_level, current_level, 
                mv_s, mv_e, coarse_sol, residual);
        //计算residual = rhs-A*sol
        mv_s[0] = start[2];
        mv_e[0] = end[2];
        mv_s[1] = start[1];
        mv_e[1] = end[1];
        mg->gcge_ops->MultiVecAxpby(1.0, residual, 1.0, sol, mv_s, mv_e, mg->gcge_ops);
        
	//后光滑
        GCGE_BCG(A, rhs, sol, start, end, 
                nsmooth, rate, mg->gcge_ops, 
                mg->cg_p[current_level], mg->cg_w[current_level], 
		mg->cg_res[current_level], 
                mg->cg_double_tmp, mg->cg_int_tmp);
    }//end for (if current_level)
}
