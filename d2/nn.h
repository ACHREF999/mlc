#ifndef NN_H_
#define NN_H_


#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#ifndef NN_MALLOC
#define NN_MALLOC (float* ) malloc
#endif // !NN_MALLOC


#ifndef NN_ASSERT
#define NN_ASSERT assert
#endif // !NN_ASSERT




typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
}Matrice;



#define MAT_PRINT(m) mat_print(m,#m)

#define MAT_AT(m,r,c) (m).es[(r)*(m).stride + (c)]

float rand_float(void);
float sigf(float x);




Matrice mat_alloc(size_t rows, size_t cols);
Matrice mat_row(Matrice m,size_t row);

void mat_rand(Matrice m,float low,float high);
void mat_dot(Matrice result , Matrice a , Matrice b);
void mat_sum(Matrice result , Matrice a , Matrice b);
void mat_print(Matrice m,const char* name);
void mat_fill(Matrice m ,float val);
void mat_sig(Matrice m);
void mat_copy(Matrice dst , Matrice src);



#endif // !NN_H_


float rand_float(void){

    return (float) rand()/(float)RAND_MAX;

}

float sigf(float x){
    return (float) 1.0f / (1.0f + expf(-x));
}


#ifdef NN_IMPLEMENTATION
Matrice mat_alloc(size_t rows, size_t cols){
    Matrice m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}




void mat_dot(Matrice result , Matrice a , Matrice b){
    NN_ASSERT(a.cols==b.rows);
    NN_ASSERT(result.rows == a.rows);
    NN_ASSERT(result.cols == b.cols);
    size_t n = a.cols;
    
    
    for (size_t i = 0; i < result.rows; i++)
    {
        for (size_t j = 0; j < result.cols; j++)
        {
            MAT_AT(result,i,j) = 0;
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(result,i,j) += MAT_AT(a,i,k)*MAT_AT(b,k,j); 
            }
        }
    }


}

void mat_sum(Matrice result , Matrice a , Matrice b){
    NN_ASSERT(a.rows==b.rows);
    NN_ASSERT(result.rows==b.rows);
    NN_ASSERT(a.cols==b.cols);
    NN_ASSERT(result.cols==b.cols);

    for (size_t i = 0; i < result.rows; i++)
    {
        for (size_t j = 0; j < result.cols; j++)
        {
            MAT_AT(result,i,j) = MAT_AT(a,i,j) + MAT_AT(b,i,j);
        }
    }
}

void mat_print(Matrice m,const char * name){
    printf("%s = [\n",name);
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("\t%f",MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("]\n");
}

void es_print(Matrice m){
    for (size_t i = 0; i < m.rows*m.cols; i++)
    {
        printf("%f ",m.es[i]);
    }
    printf("\n");
}

void mat_rand(Matrice m,float low,float high){

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = rand_float()*(high-low) +low;
        }
    }

}


void mat_fill(Matrice m,float val){

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m,i,j) = val;
        }
    }
}
void mat_sig(Matrice m){

    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols ; j++)
        {
            MAT_AT(m,i,j) = sigf(MAT_AT(m,i,j));
        }
    }
}


Matrice mat_row(Matrice m,size_t row){

    return (Matrice){
        .rows = 1,
        .cols=m.cols,
        .stride=m.stride,
        .es = &MAT_AT(m,row,0)
    };
}

void mat_copy(Matrice dst,Matrice src){
    
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; i++)
    {
        for (size_t j = 0; j < dst.cols; j++)
        {
            MAT_AT(dst,i,j) = MAT_AT(src,i,j);
        }
    }


}




#endif