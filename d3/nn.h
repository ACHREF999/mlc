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



#define MAT_PRINT(m) mat_print(m,#m,0)

#define MAT_AT(m,r,c) (m).es[(r)*(m).stride + (c)]

float rand_float(void);
float sigf(float x);




Matrice mat_alloc(size_t rows, size_t cols);
Matrice mat_row(Matrice m,size_t row);

void mat_rand(Matrice m,float low,float high);
void mat_dot(Matrice result , Matrice a , Matrice b);
void mat_sum(Matrice result , Matrice a , Matrice b);
void mat_print(Matrice m,const char* name,size_t padding);
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

void mat_print(Matrice m,const char * name,size_t padding){
    
    printf("%*s%s = [\n",(int) padding , "",name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s\t",(int) padding,"");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f\t",MAT_AT(m,i,j));
        }
        printf("\n");
    }
    printf("%*s]\n\n",(int) padding , "");
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


















//NN




#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) ((nn).as[(nn).count])


typedef struct {
    

    // This is for the number of layers
    size_t count;

    Matrice *ws;
    Matrice *bs;
    Matrice *as; //THE AMOUNT IS COUNT +1 BCUZ OF INPUT
    
} NN;



void nn_print(NN nn,const char* name);
NN nn_alloc(size_t* arch , size_t arch_count);
void nn_rand(NN nn, float low,float high );
void nn_forward(NN nn);
float nn_cost(NN nn ,Matrice ti , Matrice to);
void nn_finite_diff(NN n,NN g , float eps , Matrice ti , Matrice to);
void nn_learn(NN nn , NN g , float rate);
void nn_backprop(NN nn,NN g,Matrice ti , Matrice to);

//variadic functions 



NN nn_alloc(size_t* arch , size_t arch_count){
    
    NN_ASSERT(arch_count>0);
    // arch = {input_count , layer1_count , layer2_count ..}
    NN nn ;
    nn.count = arch_count -1;
    nn.ws = (Matrice*)NN_MALLOC(sizeof(*nn.ws)*nn.count);
    NN_ASSERT(nn.ws!= NULL);

    nn.bs = (Matrice*) NN_MALLOC(sizeof(*nn.bs)*nn.count);
    NN_ASSERT(nn.bs!=NULL);

    nn.as = (Matrice*) NN_MALLOC(sizeof(*nn.as)*(nn.count+1));
    NN_ASSERT(nn.as!=NULL);


    nn.as[0] = mat_alloc(1,arch[0]);

    for (size_t i = 0; i < nn.count; i++)
    {
        nn.ws[i] = mat_alloc(nn.as[i].cols,arch[i+1]);
        nn.bs[i] = mat_alloc(1,arch[i+1]);
        nn.as[i+1] = mat_alloc(1,arch[i+1]);
    }





    return nn;
}


#define NN_PRINT(nn) nn_print(nn,#nn)

void nn_print(NN nn,const char* name){
    printf("\n%s = [\n\n",name);
    
    char buffer[256];

    for (size_t i = 0; i < nn.count; i++)
    {
        snprintf(buffer,sizeof(buffer),"ws%zu : ",i);
        mat_print(nn.ws[i],buffer,4);
        snprintf(buffer,sizeof(buffer),"bs%zu : ",i);
        mat_print(nn.bs[i],buffer,4);


    }
    

    printf(" ] \n\n");


}

void nn_rand(NN nn, float low,float high ){
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_rand(nn.ws[i],low,high);
        mat_rand(nn.bs[i],low,high);
    }

}

void nn_forward(NN nn){

    for (size_t i = 0; i < nn.count; i++)
    {
        // printf("(%zu,%zu) * (%zu,%zu)\n",nn.as[i].rows,nn.as[i].cols,nn.ws[i].rows,nn.ws[i].cols);
        mat_dot(nn.as[i+1],nn.as[i],nn.ws[i]);
        mat_sum(nn.as[i+1],nn.as[i+1],nn.bs[i]);
        mat_sig(nn.as[i+1]);

        
    }


}


float nn_cost(NN nn ,Matrice ti , Matrice to){
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    float c = 0;

    for (size_t i = 0; i < n; i++)
    {
        Matrice x = mat_row(ti,i);
        Matrice y = mat_row(to,i);
        mat_copy(NN_INPUT(nn),x);
        nn_forward(nn);
        NN_OUTPUT(nn);
        size_t q = to.cols;
        for (size_t j = 0; j < q; j++)
        {
            float d = MAT_AT(NN_OUTPUT(nn),0,j) - MAT_AT(y,0,j);
            c += d*d;
        }   

    }

    return c/n;
}

void nn_finite_diff(NN nn,NN g , float eps , Matrice ti , Matrice to){

    float cost = nn_cost(nn,ti,to);

    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                MAT_AT(nn.ws[i],j,k) += eps; 
                MAT_AT(g.ws[i],j,k)  = (nn_cost(nn,ti,to) - cost )/eps;
                MAT_AT(nn.ws[i],j,k) -= eps;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                MAT_AT(nn.bs[i],j,k) += eps; 
                MAT_AT(g.bs[i],j,k)  = (nn_cost(nn,ti,to) - cost )/eps;
                MAT_AT(nn.bs[i],j,k) -= eps;
            }
        }



    }

}

void nn_zero(NN g ){

    for (size_t i = 0; i < g.count; i++)
    {
        
    

        mat_fill(g.ws[i],0);
        mat_fill(g.bs[i],0);
        mat_fill(g.as[i],0);

    }
    mat_fill(g.as[g.count],0);

} 

void nn_backprop(NN nn,NN g,Matrice ti , Matrice to){
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols==NN_OUTPUT(nn).cols);
    size_t n = ti.rows;
    // we need to clear the gradient after each epoch  we cant just accumm

    nn_zero(g);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    // we are itering each sample
    for (size_t i = 0; i < n; i++)
    {
        mat_copy(NN_INPUT(nn),mat_row(ti,i));
        nn_forward(nn);

        // we need to clear as from 
        for (size_t j = 0; j <=  g.count; j++)
        {
            mat_fill(g.as[j],0);
        }
        // we calc the `first` cost func
        for (size_t j = 0; j < to.cols; j++)
        {
            
        MAT_AT(NN_OUTPUT(g),0,j) = MAT_AT(NN_OUTPUT(nn),0,j) - MAT_AT(to,i,j) ;
        }

        // we are itering layers
        for (size_t l = nn.count; l > 0; l--)
        {
            // we are itering neurons in that layer
            for (size_t j = 0; j < nn.as[l].cols; j++)
            {
                // j - weight matrix col
                // k - weight matrix row

                float a = MAT_AT(nn.as[l],0,j);
                float da = MAT_AT(g.as[l],0,j);
                
                // we update the neurons bias gradient
                // we can , because the derivative with respect to bias is irrelevant to prev layer activation 
                MAT_AT(g.bs[l-1],0,j) +=  2*a*(da)*(1-a);
                
                // for the weights we need to iter them all (aka iter all neurons from prev layer )
                for (size_t k = 0; k < nn.as[l-1].cols; k++)
                {
                    float prev_actv = MAT_AT(nn.as[l-1],0,k);
                    MAT_AT(g.ws[l-1],k,j) += 2*da*a*(1-a)*prev_actv;
                    
                    // l-1 because l is for indexing layers which is +1 the weights idx

                    // MAT_AT(g.as[l-1],0,k) += 2*da*a*(1-a)*MAT_AT(nn.ws[l],k,j);
                    MAT_AT(g.as[l-1],0,k) += 2*da*a*(1-a)*MAT_AT(nn.ws[l-1],k,j);

                    // u can think of it as : 
                    // g.as[(l-1) - 1 due to as] = ....*..[l-1]
                }
            }
        }

    }


    for (size_t i = 0; i < g.count; i++)
    {
        for (size_t j = 0; j < g.ws[i].rows; j++)
        {
            for (size_t k = 0; k < g.ws[i].cols; k++)
            {
                MAT_AT(g.ws[i],j,k) /= n;
            }
        }

        for (size_t j = 0; j < g.bs[i].rows; j++)
        {
            for (size_t k = 0; k < g.bs[i].cols; k++)
            {
                MAT_AT(g.ws[i],j,k) /= n;
            }
        }


    }


}

void nn_learn(NN nn ,NN g,float rate){
    for (size_t i = 0; i < nn.count; i++)
    {
        for (size_t j = 0; j < nn.ws[i].rows; j++)
        {
            for (size_t k = 0; k < nn.ws[i].cols; k++)
            {
                MAT_AT(nn.ws[i],j,k) -= rate * MAT_AT(g.ws[i],j,k);
            }
        }   

        for (size_t j = 0; j < nn.bs[i].rows; j++)
        {
            for (size_t k = 0; k < nn.bs[i].cols; k++)
            {
                MAT_AT(nn.bs[i],j,k) -= rate * MAT_AT(g.bs[i],j,k);
            }
        }   

    }
}

#endif //NN_IMPLEMENTATION