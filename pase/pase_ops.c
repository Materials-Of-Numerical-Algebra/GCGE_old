/*
 * =====================================================================================
 *
 *       Filename:  gcge_ops.c
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
#include <time.h>

#include "pase_ops.h"

/* if use mpi, multivec inner prod will be improved by MPI_Typre_vector and MPI_Op_create */
#if GCGE_USE_MPI
GCGE_INT SIZE_B, SIZE_E, LDA;
void user_fn_submatrix_sum_pase_ops(double *in,  double *inout,  int *len,  MPI_Datatype* dptr)
{
    int i, j;
    double *b,  *a;
    double one = 1.0;
    int    inc = 1;
    for (i = 0; i < *len; ++i)
    {
        for (j = 0; j < SIZE_B; ++j)
        {
            b = inout+j*LDA;
            a = in+j*LDA;
            daxpy_(&SIZE_E, &one, a, &inc, b, &inc);
            /*
               for (k = 0; k < SIZE_E; ++k)
               {
               b[k] += a[k];
               }
               */
        }
    }
}  
#endif

//给单向量设随机初值, seed为随机数种子
void PASE_DefaultVecSetRandomValue(void *vec, PASE_INT seed, struct PASE_OPS_ *ops)
{
    void      *b_H        = ((PASE_Vector)vec)->b_H;
    PASE_REAL *aux_h      = ((PASE_Vector)vec)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_Vector)vec)->num_aux_vec;
    //b_H部分调用GCGE_OPS赋随机初值
    ops->gcge_ops->VecSetRandomValue(b_H, ops->gcge_ops);
    //aux_h部分直接赋随机初值
    PASE_INT  i = 0;
    srand(seed);
    for(i=0; i<num_aux_vec; i++)
    {
        aux_h[i] = ((double)rand())/((double)RAND_MAX+1);
    }
}


//PASE矩阵乘向量 r = Matrix * x
void PASE_DefaultMatDotVec(void *Matrix, void *x, void *r, struct PASE_OPS_ *ops)
{
    void      *A_H     = ((PASE_Matrix)Matrix)->A_H;
    void      **aux_Hh = ((PASE_Matrix)Matrix)->aux_Hh;
    void      **aux_hH = ((PASE_Matrix)Matrix)->aux_hH;
    PASE_REAL *aux_hh  = ((PASE_Matrix)Matrix)->aux_hh;

    void      *x_b_H      = ((PASE_Vector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_Vector)x)->aux_h;
    void      *r_b_H      = ((PASE_Vector)r)->b_H;
    PASE_REAL *r_aux_h    = ((PASE_Vector)r)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_Vector)r)->num_aux_vec;
    //PASE_REAL *r_aux_h_tmp= ((PASE_Vector)r)->aux_h_tmp;
    PASE_REAL *r_aux_h_tmp= (PASE_REAL*)calloc(num_aux_vec, sizeof(PASE_REAL));

    PASE_INT mv_s[2];
    PASE_INT mv_e[2];
 
    //计算 r_b_H = A_H * x_b_H + aux_Hh * aux_h
    //计算 r_b_H = A_H * x_b_H
    ops->gcge_ops->MatDotVec(A_H, x_b_H, r_b_H, ops->gcge_ops);

    //计算 r->aux_h = aux_Hh^T * x->b_H + aux_hh^T * x->aux_h
    //计算 r->aux_h = aux_Hh^T * x->b_H
#if GCGE_USE_MPI
    MPI_Request request;
    MPI_Status  status;
#endif
    PASE_INT i=0;
    if(PASE_NO == ((PASE_Matrix)Matrix)->is_diag) {
        mv_s[0] = 0;
        mv_e[0] = num_aux_vec;
        mv_s[1] = 0;
        mv_e[1] = 1;
        //如果A_H矩阵不是单位阵，要计算下面的向量组与单向量内积
        ops->gcge_ops->MultiVecInnerProdLocal(aux_Hh, (void**)(&x_b_H), r_aux_h,
                "nonsym", mv_s, mv_e, num_aux_vec, 1, ops->gcge_ops);

#if GCGE_USE_MPI
        MPI_Iallreduce(MPI_IN_PLACE, r_aux_h, num_aux_vec, MPI_DOUBLE, 
	      MPI_SUM, MPI_COMM_WORLD, &request);
        //MPI_Allreduce(MPI_IN_PLACE, r_aux_h, num_aux_vec, MPI_DOUBLE, 
	//      MPI_SUM, MPI_COMM_WORLD);
#endif

    } else {
        //如果A_H矩阵是单位阵，那么aux_Hh为0
        memset(r_aux_h, 0.0, num_aux_vec*sizeof(PASE_SCALAR));
    }

    //计算 r_b_H += aux_Hh * aux_h
    mv_s[0] = 0;
    mv_e[0] = num_aux_vec;
    mv_s[1] = 0;
    mv_e[1] = 1;
    PASE_REAL alpha = 1.0;
    PASE_REAL beta  = 1.0;
    //多向量线性组合得到单向量
    ops->gcge_ops->MultiVecLinearComb(aux_Hh, (void**)(&r_b_H), mv_s, mv_e, 
            x_aux_h, num_aux_vec, 1, alpha, beta, ops->gcge_ops);

    //计算 r->aux_h += aux_hh^T * x->aux_h
    PASE_INT  ncols = 1;
    alpha = 1.0;
    beta  = 0.0;
    memset(r_aux_h_tmp, 0.0, num_aux_vec*sizeof(PASE_REAL));
    ops->gcge_ops->DenseMatDotDenseMat("T", "N", &num_aux_vec, &ncols, 
            &num_aux_vec, &alpha, aux_hh, &num_aux_vec, 
            x_aux_h, &num_aux_vec, &beta, r_aux_h_tmp, &num_aux_vec);

    if(PASE_NO == ((PASE_Matrix)Matrix)->is_diag) {
      MPI_Wait(&request, &status);
    }

    ops->gcge_ops->ArrayAXPBY(1.0, r_aux_h_tmp, 1.0, r_aux_h, num_aux_vec);
    free(r_aux_h_tmp); r_aux_h_tmp = NULL;
}


//单向量线性组合 y = a*x + b*y
void PASE_DefaultVecAxpby(PASE_REAL a, void *x, PASE_REAL b, void *y,
        struct PASE_OPS_ *ops)
{
    void      *x_b_H      = ((PASE_Vector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_Vector)x)->aux_h;
    void      *y_b_H      = ((PASE_Vector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_Vector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_Vector)y)->num_aux_vec;
    //y->b_H = a * x->b_H + b * y->b_H
    ops->gcge_ops->VecAxpby(a, x_b_H, b, y_b_H, ops->gcge_ops);
    //y->aux_h = a * x->aux_h + b * y->aux_h
    ops->gcge_ops->ArrayAXPBY(a, x_aux_h, b, y_aux_h, num_aux_vec);
}

//单向量内积 value_ip = x^T * y
void PASE_DefaultVecInnerProd(void *x, void *y, PASE_REAL *value_ip, struct PASE_OPS_ *ops)
{
    void      *x_b_H      = ((PASE_Vector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_Vector)x)->aux_h;
    void      *y_b_H      = ((PASE_Vector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_Vector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_Vector)y)->num_aux_vec;
    //value_ip = x->b_H * y->b_H + x->aux_h * y->aux_h
    //value_ip = x->b_H * y->b_H
    ops->gcge_ops->VecInnerProd(x_b_H, y_b_H, value_ip, ops->gcge_ops);
    //value_ip += x->aux_h * y->aux_h
    *value_ip += ops->gcge_ops->ArrayDotArray(x_aux_h, y_aux_h, num_aux_vec);
}
    
//单向量局部内积 value_ip = x^T * y
//如果是0进程，计算 aux 部分
void PASE_DefaultVecLocalInnerProd(void *x, void *y, PASE_REAL *value_ip, struct PASE_OPS_ *ops)
{
    void      *x_b_H      = ((PASE_Vector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_Vector)x)->aux_h;
    void      *y_b_H      = ((PASE_Vector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_Vector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_Vector)y)->num_aux_vec;
    //value_ip = x->b_H * y->b_H + x->aux_h * y->aux_h
    //value_ip = x->b_H * y->b_H
    ops->gcge_ops->VecLocalInnerProd(x_b_H, y_b_H, value_ip, ops->gcge_ops);
    //value_ip += x->aux_h * y->aux_h
    //0号进行进行计算
    PASE_INT rank = 0;
#if GCGE_USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if(rank == 0)
    {
        *value_ip += ops->gcge_ops->ArrayDotArray(x_aux_h, y_aux_h, num_aux_vec);
    }
}
    
//由src_vec单向量创建des_vec单向量
void PASE_DefaultVecCreateByVec(void **des_vec, void *src_vec, struct PASE_OPS_ *ops)
{
    PASE_Vector vec         = (PASE_Vector)malloc(sizeof(pase_Vector));
    void        *b_H        = ((PASE_Vector)src_vec)->b_H;
    PASE_REAL   *aux_h      = ((PASE_Vector)src_vec)->aux_h;
    PASE_INT    num_aux_vec = ((PASE_Vector)src_vec)->num_aux_vec;

    //des_vec->b_H 通过gcge_ops由 src_vec->b_H创建
    vec->num_aux_vec = num_aux_vec;
    ops->gcge_ops->VecCreateByVec(&(vec->b_H), b_H, ops->gcge_ops);
    //des_vec->aux_h 直接分配空间
    vec->aux_h = (PASE_REAL*)calloc(num_aux_vec, sizeof(PASE_REAL));
    //if_test_ops, 如果测试ops，要把下面这个注释关掉
    vec->aux_h_tmp = (PASE_REAL*)calloc(num_aux_vec, sizeof(PASE_REAL));
    *des_vec = (void*)vec;
}

//由矩阵mat创建des_vec单向量
void PASE_DefaultVecCreateByMat(void **vec, void *mat, struct PASE_OPS_ *ops)
{
    PASE_Vector v           = (PASE_Vector)malloc(sizeof(pase_Vector));
    void        *A_H        = ((PASE_Matrix)mat)->A_H;
    PASE_INT    num_aux_vec = ((PASE_Matrix)mat)->num_aux_vec;

    //des_vec->b_H 通过gcge_ops由 mat->A_H创建
    v->num_aux_vec = num_aux_vec;
    ops->gcge_ops->VecCreateByMat(&(v->b_H), A_H, ops->gcge_ops);
    //des_vec->aux_h 直接分配空间
    v->aux_h = (PASE_REAL*)calloc(num_aux_vec, sizeof(PASE_REAL));
    //if_test_ops, 如果测试ops，要把下面这个注释关掉
    v->aux_h_tmp = (PASE_REAL*)calloc(num_aux_vec, sizeof(PASE_REAL));
    *vec = (void*)v;
}

//销毁单向量 vec 空间
void PASE_DefaultVecDestroy(void **vec, struct PASE_OPS_ *ops)
{
    //通过gcge_ops释放vec->b_H空间
    ops->gcge_ops->VecDestroy(&(((PASE_Vector)(*vec))->b_H), ops->gcge_ops);
    //释放vec->aux_h空间
    free(((PASE_Vector)(*vec))->aux_h); 
    free(((PASE_Vector)(*vec))->aux_h_tmp); 
    ((PASE_Vector)(*vec))->aux_h = NULL; 
    free((PASE_Vector)(*vec)); 
    *vec = NULL;
}

//由单向量vec创建向量组multi_vec,其中multi_vec中有n_vec个向量
void PASE_DefaultMultiVecCreateByVec(void ***multi_vec, PASE_INT n_vec, 
        void *vec, struct PASE_OPS_ *ops)
{
    void      *b_H        = ((PASE_Vector)vec)->b_H;
    PASE_INT  num_aux_vec = ((PASE_Vector)vec)->num_aux_vec;

    //分配空间
    PASE_MultiVector vecs = (PASE_MultiVector)malloc(sizeof(pase_MultiVector));
    //给aux部分size赋值
    vecs->num_vec     = n_vec;
    vecs->num_aux_vec = num_aux_vec;
    //通过gcge_ops创建普通向量组部分
    ops->gcge_ops->MultiVecCreateByVec(&(vecs->b_H), n_vec, b_H, ops->gcge_ops);
    //创建aux部分
    vecs->aux_h = (PASE_REAL*)calloc(n_vec*num_aux_vec, sizeof(PASE_REAL));
    vecs->aux_h_tmp = (PASE_REAL*)calloc(n_vec*num_aux_vec, sizeof(PASE_REAL));
    *multi_vec  = (void**)vecs;
}

//由矩阵mat创建向量组multi_vec,其中multi_vec中有n_vec个向量
void PASE_DefaultMultiVecCreateByMat(void ***multi_vec, PASE_INT n_vec, 
        void *mat, struct PASE_OPS_ *ops)
{
    void      *A_H        = ((PASE_Matrix)mat)->A_H;
    PASE_INT  num_aux_vec = ((PASE_Matrix)mat)->num_aux_vec;

    //分配空间
    PASE_MultiVector vecs = (PASE_MultiVector)malloc(sizeof(pase_MultiVector));
    //给aux部分size赋值
    vecs->num_vec     = n_vec;
    vecs->num_aux_vec = num_aux_vec;
    //通过gcge_ops创建普通向量组部分
    ops->gcge_ops->MultiVecCreateByMat(&(vecs->b_H), n_vec, A_H, ops->gcge_ops);
    //if_test_ops, 如果测试ops，要把下面这个注释关掉
    vecs->aux_h_tmp = (PASE_REAL*)calloc(n_vec*num_aux_vec, sizeof(PASE_REAL));
    //创建aux部分
    vecs->aux_h = (PASE_REAL*)calloc(n_vec*num_aux_vec, sizeof(PASE_REAL));
    *multi_vec  = (void**)vecs;
}

//由向量组init_vec创建向量组multi_vec,其中multi_vec中有n_vec个向量
void PASE_DefaultMultiVecCreateByMultiVec(void **init_vec, void ***multi_vec,
        PASE_INT n_vec, struct PASE_OPS_ *ops)
{
    void      **b_H       = ((PASE_MultiVector)init_vec)->b_H;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)init_vec)->num_aux_vec;

    //分配空间
    PASE_MultiVector vecs = (PASE_MultiVector)malloc(sizeof(pase_MultiVector));
    //给aux部分size赋值
    vecs->num_vec     = n_vec;
    vecs->num_aux_vec = num_aux_vec;
    //通过gcge_ops创建普通向量组部分
    ops->gcge_ops->MultiVecCreateByMultiVec(&(vecs->b_H), n_vec, b_H, ops->gcge_ops);
    //创建aux部分
    vecs->aux_h = (PASE_REAL*)calloc(n_vec*num_aux_vec, sizeof(PASE_REAL));
    vecs->aux_h_tmp = (PASE_REAL*)calloc(n_vec*num_aux_vec, sizeof(PASE_REAL));
    *multi_vec  = (void**)vecs;
}

//销毁多向量组空间，其中MultiVec中有n_vec个向量(n_vec这个参数其实没用，为了与GCGE_OPS保持一致才加的)
void PASE_DefaultMultiVecDestroy(void ***MultiVec, PASE_INT n_vec, struct PASE_OPS_ *ops)
{
    //通过gcge_ops释放b_H部分的空间
    ops->gcge_ops->MultiVecDestroy(&(((PASE_MultiVector)(*MultiVec))->b_H),
            ((PASE_MultiVector)(*MultiVec))->num_vec, ops->gcge_ops);
    //释放aux_h部分的空间
    free(((PASE_MultiVector)(*MultiVec))->aux_h);
    //if_test_ops, 如果测试ops，要把下面这个注释关掉
    free(((PASE_MultiVector)(*MultiVec))->aux_h_tmp);
    ((PASE_MultiVector)(*MultiVec))->aux_h = NULL;
    //if_test_ops, 如果测试ops，要把下面这个注释关掉
    ((PASE_MultiVector)(*MultiVec))->aux_h_tmp = NULL;
    free((PASE_MultiVector)(*MultiVec));
    *MultiVec = NULL;
}

//给向量组赋初值，从multi_vec中第start个向量开始，给n_vec个向量赋随机初值
void PASE_DefaultMultiVecSetRandomValue(void **multi_vec, PASE_INT start, 
        PASE_INT n_vec, struct PASE_OPS_ *ops)
{
    //通过gcge_ops给b_H部分赋随机初值
    ops->gcge_ops->MultiVecSetRandomValue(((PASE_MultiVector)multi_vec)->b_H, start, n_vec, ops->gcge_ops);
    PASE_INT  i = 0;
    //aux部分用srand，这里time(NULL)需要再include time头文件
    srand(time(NULL));
    PASE_INT  num_aux_vec = ((PASE_MultiVector)multi_vec)->num_aux_vec;
    PASE_INT  num_vec     = ((PASE_MultiVector)multi_vec)->num_vec;
    PASE_REAL *aux_h      = ((PASE_MultiVector)multi_vec)->aux_h;
    for(i=0; i<num_aux_vec*num_vec; i++)
    {
        aux_h[i] = ((double)rand())/((double)RAND_MAX+1);
    }
}

//矩阵乘多向量组，y[:,start[1]:end[1]] = mat * x[:,start[0]:end[0]]
void PASE_DefaultMatDotMultiVec(void *mat, void **x, void **y, 
        PASE_INT *start, PASE_INT *end, struct PASE_OPS_ *ops)
{
    void      *A_H     = ((PASE_Matrix)mat)->A_H;
    void      **aux_Hh = ((PASE_Matrix)mat)->aux_Hh;
    void      **aux_hH = ((PASE_Matrix)mat)->aux_hH;
    PASE_REAL *aux_hh  = ((PASE_Matrix)mat)->aux_hh;

    void      **x_b_H     = ((PASE_MultiVector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_MultiVector)x)->aux_h;
    void      **y_b_H     = ((PASE_MultiVector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_MultiVector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)y)->num_aux_vec;
    //PASE_REAL *y_aux_h_tmp = ((PASE_MultiVector)y)->aux_h_tmp;
    PASE_REAL *y_aux_h_tmp = (PASE_REAL*)calloc((end[1]-start[1])*num_aux_vec, 
	  sizeof(PASE_REAL));

    PASE_INT  mv_s[2];
    PASE_INT  mv_e[2];
    PASE_REAL alpha = 1.0;
    PASE_REAL beta  = 1.0;
    PASE_INT  i = 0;
 
    //计算 r_b_H = A_H * x_b_H + aux_Hh * aux_h
    //计算 r_b_H = A_H * x_b_H
    ops->gcge_ops->MatDotMultiVec(A_H, x_b_H, y_b_H, start, end, ops->gcge_ops);

    //计算 r->aux_h = aux_Hh^T * x->b_H + aux_hh^T * x->aux_h
    //计算 r->aux_h = aux_Hh^T * x->b_H
#if GCGE_USE_MPI
    SIZE_B = end[1]-start[1];
    //SIZE_E = end[0]-start[0];
    SIZE_E = num_aux_vec;
    LDA    = num_aux_vec;
    MPI_Request request;
    MPI_Status  status;
    MPI_Datatype SUBMATRIX;
    MPI_Op SUBMATRIX_SUM;
#endif
    if(PASE_NO == ((PASE_Matrix)mat)->is_diag) {
        mv_s[0] = 0;
        mv_e[0] = num_aux_vec;
        mv_s[1] = start[0];
        mv_e[1] = end[0];
        ops->gcge_ops->MultiVecInnerProdLocal(aux_Hh, x_b_H, 
                y_aux_h+start[1]*num_aux_vec, "nonsym", 
                mv_s, mv_e, num_aux_vec, 0, ops->gcge_ops);
        //因为稠密矩阵y_aux_h可能只用到一些行，即要消息传输的部分并不连续
	//所以用下面的方式进行消息传输
#if GCGE_USE_MPI

        MPI_Type_vector(SIZE_B, SIZE_E, LDA, MPI_DOUBLE, &SUBMATRIX);
        MPI_Type_commit(&SUBMATRIX);
        
        MPI_Op_create((MPI_User_function*)user_fn_submatrix_sum_pase_ops, 1, &SUBMATRIX_SUM);
        MPI_Iallreduce(MPI_IN_PLACE, y_aux_h+start[1]*num_aux_vec, 1, SUBMATRIX, SUBMATRIX_SUM, MPI_COMM_WORLD, &request);
        //MPI_Allreduce(MPI_IN_PLACE, y_aux_h+start[1]*num_aux_vec, 1, SUBMATRIX, SUBMATRIX_SUM, MPI_COMM_WORLD);

	//for(i=0; i<SIZE_B; i++) {
        //  MPI_Allreduce(MPI_IN_PLACE, y_aux_h+(start[1]+i)*num_aux_vec, num_aux_vec, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	//}
#endif

    } else {
        memset(y_aux_h+start[1]*num_aux_vec, 0.0, 
                (end[1]-start[1])*num_aux_vec*sizeof(PASE_SCALAR));
    }

    //计算 r_b_H += aux_Hh * aux_h
    alpha = 1.0;
    beta  = 1.0;
    mv_s[0] = 0;
    mv_e[0] = num_aux_vec;
    mv_s[1] = start[1];
    mv_e[1] = end[1];
    ops->gcge_ops->MultiVecLinearComb(aux_Hh, y_b_H, 
            mv_s, mv_e, x_aux_h+start[0]*num_aux_vec+start[1], num_aux_vec, 
            0, alpha, beta,  ops->gcge_ops);

    //计算 y_aux_h_tmp = aux_hh^T * x->aux_h
    PASE_INT  ncols = end[1] - start[1];
    alpha = 1.0;
    beta  = 0.0;
    memset(y_aux_h_tmp, 0.0, ncols*num_aux_vec*sizeof(PASE_REAL));
    ops->gcge_ops->DenseMatDotDenseMat("T", "N", &num_aux_vec, &ncols, 
            &num_aux_vec, &alpha, aux_hh, &num_aux_vec, 
            x_aux_h+start[0]*num_aux_vec, &num_aux_vec, &beta, 
            y_aux_h_tmp, &num_aux_vec);
            //y_aux_h+start[1]*num_aux_vec, &num_aux_vec);
#if GCGE_USE_MPI
    if(PASE_NO == ((PASE_Matrix)mat)->is_diag) {
      MPI_Wait(&request, &status);
      MPI_Op_free(&SUBMATRIX_SUM);
      MPI_Type_free(&SUBMATRIX);
    }
#endif
    //计算 r->aux_h += y_aux_h_tmp
    ops->gcge_ops->ArrayAXPBY(1.0, y_aux_h_tmp, 1.0, y_aux_h+start[1]*num_aux_vec, 
	  ncols*num_aux_vec);
    free(y_aux_h_tmp); y_aux_h_tmp = NULL;
}

//多向量的Axpby, yy = a * xx + b * yy
//其中 yy = y[:,start[1]:end[1]], xx = x[:,start[0]:end[0]]
void PASE_DefaultMultiVecAxpby(PASE_REAL a, void **x, PASE_REAL b, 
        void **y, PASE_INT *start, PASE_INT *end, struct PASE_OPS_ *ops)
{
    void      **x_b_H     = ((PASE_MultiVector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_MultiVector)x)->aux_h;
    void      **y_b_H     = ((PASE_MultiVector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_MultiVector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)y)->num_aux_vec;
    //y->b_H = a * x->b_H + b * y->b_H
    ops->gcge_ops->MultiVecAxpby(a, x_b_H, b, y_b_H, start, end, ops->gcge_ops);
    //y->aux_h = a * x->aux_h + b * y->aux_h
    ops->gcge_ops->ArrayAXPBY(a, x_aux_h+start[0]*num_aux_vec, 
            b, y_aux_h+start[1]*num_aux_vec, (end[1]-start[1])*num_aux_vec);
}

//向量组中取出一列计算Axpby, yy = a * xx + b * yy
//其中 yy = y[col_y], xx = x[col_x]
void PASE_DefaultMultiVecAxpbyColumn(PASE_REAL a, void **x, PASE_INT col_x, 
        PASE_REAL b, void **y, PASE_INT col_y, struct PASE_OPS_ *ops)
{
    void      **x_b_H     = ((PASE_MultiVector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_MultiVector)x)->aux_h;
    void      **y_b_H     = ((PASE_MultiVector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_MultiVector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)y)->num_aux_vec;
    //y->b_H = a * x->b_H + b * y->b_H
    ops->gcge_ops->MultiVecAxpbyColumn(a, x_b_H, col_x, b, y_b_H, col_y, 
            ops->gcge_ops);
    //y->aux_h = a * x->aux_h + b * y->aux_h
    ops->gcge_ops->ArrayAXPBY(a, x_aux_h+col_x*num_aux_vec, 
            b, y_aux_h+col_y*num_aux_vec, num_aux_vec);
}

//多向量组线性组合 yy = alpha * xx * a + beta * yy
//其中 yy = y[:,start[1]:end[1]], xx = x[:,start[0]:end[0]]
//lda为a的leading dimension
//如果if_Vec是1，那么*yy是一个单向量的结构
void PASE_DefaultMultiVecLinearComb(void **x, void **y, PASE_INT *start, 
        PASE_INT *end, PASE_REAL *a, PASE_INT lda, PASE_INT if_Vec, 
        PASE_REAL alpha, PASE_REAL beta, struct PASE_OPS_ *ops)
{
    void      **x_b_H     = ((PASE_MultiVector)x)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_MultiVector)x)->aux_h;
    void      **y_b_H     = ((PASE_MultiVector)y)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_MultiVector)y)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)y)->num_aux_vec;
    //通过gcge_ops计算b_H部分的线性组合
    ops->gcge_ops->MultiVecLinearComb(x_b_H, y_b_H, start, end, a, lda,
            if_Vec, alpha, beta, ops->gcge_ops);
    //aux部分的线性组合，计算 y->aux_h = x_aux_h^T * a
    PASE_INT  nrows = num_aux_vec;
    PASE_INT  mid   = end[0]-start[0];
    PASE_INT  ncols = end[1]-start[1];
    //这里不做转置
    ops->gcge_ops->DenseMatDotDenseMat("N", "N", &nrows, &ncols, 
            &mid, &alpha, x_aux_h+start[0]*num_aux_vec, &num_aux_vec, 
            a, &lda, &beta, y_aux_h+start[1]*num_aux_vec, &num_aux_vec);
}

//计算向量组内积, a = VV^T * WW
//其中 VV = V[:,start[0]:end[0]], WW = W[:,start[0]:end[0]]
//lda为a的leading dimension
//如果if_Vec是1，那么*yy是一个单向量的结构
void PASE_DefaultMultiVecInnerProd(void **V, void **W, PASE_REAL *a, 
        char *is_sym, PASE_INT *start, PASE_INT *end, PASE_INT lda, 
        PASE_INT if_Vec, struct PASE_OPS_ *ops)
{
    void      **x_b_H     = ((PASE_MultiVector)V)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_MultiVector)V)->aux_h;
    PASE_REAL *a_tmp = (PASE_REAL*)calloc(lda*(end[1]-start[1]), 
    	    sizeof(PASE_REAL));
    //PASE_REAL *a_tmp      = ((PASE_MultiVector)V)->aux_h_tmp;
    void      **y_b_H;
    PASE_REAL *y_aux_h;  
    PASE_INT  num_aux_vec;
    if(if_Vec == 0)
    {
        y_b_H       = ((PASE_MultiVector)W)->b_H;
        y_aux_h     = ((PASE_MultiVector)W)->aux_h;
        num_aux_vec = ((PASE_MultiVector)W)->num_aux_vec;
    }
    else
    {
        y_b_H       = &(((PASE_Vector)(W[0]))->b_H);
        y_aux_h     = ((PASE_Vector)(W[0]))->aux_h;
        num_aux_vec = ((PASE_Vector)(W[0]))->num_aux_vec;
        //a_tmp       = ((PASE_Vector)(W[0]))->aux_h_tmp;
    }
    //if(lda*(end[1]-start[1]) > num_aux_vec*num_aux_vec) {
    //  a_tmp = (PASE_REAL*)calloc(lda*(end[1]-start[1]), 
    //	    sizeof(PASE_REAL));
    //}
    //a = x_b_H^T * y_b_H + x_aux_h * y_aux_h
    //a = x_b_H^T * y_b_H
#if GCGE_USE_MPI
    MPI_Request request;
    MPI_Status  status;
#endif
    ops->gcge_ops->MultiVecInnerProdLocal(x_b_H, y_b_H, a, is_sym, 
	  start, end, lda, if_Vec, ops->gcge_ops);
#if GCGE_USE_MPI

    SIZE_B = end[1]-start[1];
    SIZE_E = end[0]-start[0];
    LDA    = lda;
    
    MPI_Datatype SUBMATRIX;
    MPI_Type_vector(SIZE_B, SIZE_E, LDA, MPI_DOUBLE, &SUBMATRIX);
    MPI_Type_commit(&SUBMATRIX);
    
    MPI_Op SUBMATRIX_SUM;
    MPI_Op_create((MPI_User_function*)user_fn_submatrix_sum_pase_ops, 1, &SUBMATRIX_SUM);
    MPI_Iallreduce(MPI_IN_PLACE, a, 1, SUBMATRIX, SUBMATRIX_SUM, MPI_COMM_WORLD, &request);
    //MPI_Allreduce(MPI_IN_PLACE, a, 1, SUBMATRIX, SUBMATRIX_SUM, MPI_COMM_WORLD);
    //PASE_INT i = 0;
    //for(i=0; i<SIZE_B; i++)
    //{
    //MPI_Allreduce(MPI_IN_PLACE, a+i*lda, SIZE_E, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    //}

#endif
    //y_aux_h_tmp = x_aux_h * y_aux_h
    PASE_INT  mid   = num_aux_vec;
    PASE_INT  nrows = end[0]-start[0];
    PASE_INT  ncols = end[1]-start[1];
    PASE_REAL alpha = 1.0;
    PASE_REAL beta  = 1.0;
    //TODO nrows==ncols==1的情况
    memset(a_tmp, 0.0, ncols*lda*sizeof(PASE_REAL));
    ops->gcge_ops->DenseMatDotDenseMat("T", "N", &nrows, &ncols, 
            &mid, &alpha, x_aux_h+start[0]*num_aux_vec, &num_aux_vec, 
            y_aux_h+start[1]*num_aux_vec, &num_aux_vec, &beta, a_tmp, &lda);
            //y_aux_h+start[1]*num_aux_vec, &num_aux_vec, &beta, a, &lda);
#if GCGE_USE_MPI
    MPI_Wait(&request, &status);
    MPI_Op_free(&SUBMATRIX_SUM);
    MPI_Type_free(&SUBMATRIX);
#endif
    //计算 a += y_aux_h_tmp
    ops->gcge_ops->ArrayAXPBY(1.0, a_tmp, 1.0, a, ncols*lda);
    //if(lda*(end[1]-start[1]) > num_aux_vec*num_aux_vec) {
    free(a_tmp); a_tmp = NULL;
    //}
}

//多向量组交换，目前GCGE中用到的这个函数都是需要将V_2拷贝给V_1,
//所以目前只实现将V_2拷贝给V_1的功能(对aux部分)
void PASE_DefaultMultiVecSwap(void **V_1, void **V_2, PASE_INT *start, 
        PASE_INT *end, struct PASE_OPS_ *ops)
{
    void      **x_b_H     = ((PASE_MultiVector)V_1)->b_H;
    PASE_REAL *x_aux_h    = ((PASE_MultiVector)V_1)->aux_h;
    void      **y_b_H     = ((PASE_MultiVector)V_2)->b_H;
    PASE_REAL *y_aux_h    = ((PASE_MultiVector)V_2)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)V_2)->num_aux_vec;
    //Swap x_b_H and y_b_H
    ops->gcge_ops->MultiVecSwap(x_b_H, y_b_H, start, end, ops->gcge_ops);
    //Swap x_aux_h and y_aux_h
    //因为目前这个函数只用在GCGE中，且用处为将V_2复制给V_1, 
    //为了减少中间的内存分配，先只将y_aux_h拷贝给x_aux_h
    memcpy(x_aux_h+start[0]*num_aux_vec, y_aux_h+start[1]*num_aux_vec,
            (end[1]-start[1])*num_aux_vec*sizeof(PASE_SCALAR));
}

//打印多向量组
void PASE_DefaultMultiVecPrint(void **x, PASE_INT n, struct PASE_OPS_ *ops)
{
    void      **b_H       = ((PASE_MultiVector)x)->b_H;
    PASE_REAL *aux_h      = ((PASE_MultiVector)x)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)x)->num_aux_vec;
    //打印b_H部分
    ops->gcge_ops->MultiVecPrint(b_H, n, ops->gcge_ops);
    PASE_INT i = 0;
    PASE_INT j = 0;
    //打印aux_h部分
    for(i=0; i<n; i++)
    {
        for(j=0; j<num_aux_vec; j++)
        {
            GCGE_Printf("aux_h(%d,%d) = %18.15lf\n", i, j, 
                    aux_h[i*num_aux_vec+j]);
        }
    }
}

//从向量组中获取一个向量
void PASE_DefaultGetVecFromMultiVec(void **V, PASE_INT j, void **x, struct PASE_OPS_ *ops)
{
    void      **V_b_H     = ((PASE_MultiVector)V)->b_H;
    PASE_REAL *V_aux_h    = ((PASE_MultiVector)V)->aux_h;
    PASE_REAL *V_aux_h_tmp= ((PASE_MultiVector)V)->aux_h_tmp;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)V)->num_aux_vec;
    //首先创建*x向量
    PASE_Vector vec         = (PASE_Vector)malloc(sizeof(pase_Vector));
    vec->num_aux_vec = num_aux_vec;
    //从V_b_H中获取b_H
    ops->gcge_ops->GetVecFromMultiVec(V_b_H, j, &(vec->b_H), ops->gcge_ops);
    //将vec->aux_h的指针指向相应的位置
    vec->aux_h = V_aux_h+j*num_aux_vec;
    vec->aux_h_tmp = V_aux_h_tmp;
    *x = vec;
    //memcpy(((PASE_Vector)(*x))->aux_h, V_aux_h+j*num_aux_vec, num_aux_vec*sizeof(PASE_SCALAR));
}

//将取出的向量返回给向量组
void PASE_DefaultRestoreVecForMultiVec(void **V, PASE_INT j, void **x, struct PASE_OPS_ *ops)
{
    void      **V_b_H     = ((PASE_MultiVector)V)->b_H;
    PASE_REAL *V_aux_h    = ((PASE_MultiVector)V)->aux_h;
    PASE_INT  num_aux_vec = ((PASE_MultiVector)V)->num_aux_vec;
    //由b_H返回V_b_H
    ops->gcge_ops->RestoreVecForMultiVec(V_b_H, j, &(((PASE_Vector)(*x))->b_H), ops->gcge_ops);
    free((PASE_Vector)(*x));
    *x = NULL;
    //memcpy(V_aux_h+j*num_aux_vec, ((PASE_Vector)(*x))->aux_h, num_aux_vec*sizeof(PASE_SCALAR));
}

void PASE_OPS_Create(PASE_OPS **ops, GCGE_OPS *gcge_ops)
{
    *ops = (PASE_OPS*)malloc(sizeof(PASE_OPS));
    (*ops)->gcge_ops = gcge_ops;
    (*ops)->VecSetRandomValue        = PASE_DefaultVecSetRandomValue       ;
    (*ops)->MatDotVec                = PASE_DefaultMatDotVec               ;
    (*ops)->VecAxpby                 = PASE_DefaultVecAxpby                ;
    (*ops)->VecInnerProd             = PASE_DefaultVecInnerProd            ;
    (*ops)->VecLocalInnerProd        = PASE_DefaultVecLocalInnerProd       ;
    (*ops)->VecCreateByVec           = PASE_DefaultVecCreateByVec          ;
    (*ops)->VecCreateByMat           = PASE_DefaultVecCreateByMat          ;
    (*ops)->VecDestroy               = PASE_DefaultVecDestroy              ;
    (*ops)->MultiVecCreateByVec      = PASE_DefaultMultiVecCreateByVec     ;
    (*ops)->MultiVecCreateByMat      = PASE_DefaultMultiVecCreateByMat     ;
    (*ops)->MultiVecCreateByMultiVec = PASE_DefaultMultiVecCreateByMultiVec;
    (*ops)->MultiVecDestroy          = PASE_DefaultMultiVecDestroy         ;
    (*ops)->MultiVecSetRandomValue   = PASE_DefaultMultiVecSetRandomValue  ;
    (*ops)->MatDotMultiVec           = PASE_DefaultMatDotMultiVec          ;
    (*ops)->MultiVecAxpby            = PASE_DefaultMultiVecAxpby           ;
    (*ops)->MultiVecAxpbyColumn      = PASE_DefaultMultiVecAxpbyColumn     ;
    (*ops)->MultiVecLinearComb       = PASE_DefaultMultiVecLinearComb      ;
    (*ops)->MultiVecInnerProd        = PASE_DefaultMultiVecInnerProd       ;
    (*ops)->MultiVecSwap             = PASE_DefaultMultiVecSwap            ;
    (*ops)->GetVecFromMultiVec       = PASE_DefaultGetVecFromMultiVec      ;
    (*ops)->RestoreVecForMultiVec    = PASE_DefaultRestoreVecForMultiVec   ;
    (*ops)->MultiVecPrint            = PASE_DefaultMultiVecPrint           ;
}

void PASE_OPS_Free(PASE_OPS **ops)
{
    free(*ops); *ops = NULL;
}

void PASE_MatrixCreate( PASE_Matrix* pase_matrix,
                        PASE_INT num_aux_vec,
                        void *A_H, 
                        PASE_OPS *ops
                        )
{
    *pase_matrix = (PASE_Matrix)malloc(sizeof(pase_Matrix));
    (*pase_matrix)->num_aux_vec = num_aux_vec;
    (*pase_matrix)->A_H         = A_H;
    (*pase_matrix)->is_diag     = 0;
    ops->gcge_ops->MultiVecCreateByMat(&((*pase_matrix)->aux_Hh), num_aux_vec,
            A_H, ops->gcge_ops);
    (*pase_matrix)->aux_hh = (PASE_SCALAR*)calloc(num_aux_vec*num_aux_vec, 
            sizeof(PASE_SCALAR));
}

void PASE_MatrixDestroy( PASE_Matrix*  matrix, 
                         PASE_OPS *ops )
{
    PASE_INT num_aux_vec = (*matrix)->num_aux_vec;
    ops->gcge_ops->MultiVecDestroy(&((*matrix)->aux_Hh), num_aux_vec,
            ops->gcge_ops);
    free((*matrix)->aux_hh); (*matrix)->aux_hh = NULL;
    free(*matrix); *matrix = NULL;
}

