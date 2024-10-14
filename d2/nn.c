#define NN_IMPLEMENTATION
#include "nn.h"
#include <time.h>

float td[]={
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0
    };

#if 0
typedef struct {

    Matrice a0;

    Matrice w1;
    Matrice b1;
    Matrice a1;

    Matrice w2;
    Matrice b2;
    Matrice a2;

}Xor;


Xor xor_alloc(void){
    Xor m ;
    
    m.a0 = mat_alloc(1,2);
    m.w1 = mat_alloc(2,2);
    m.b1 = mat_alloc(1,2);

    m.a1 = mat_alloc(1,2);
    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);

    m.a2 = mat_alloc(1,1);

    return m;

}








void forward_xor(Xor model){



    // First Layer
    mat_dot(model.a1,model.a0,model.w1);
    mat_sum(model.a1,model.a1,model.b1);
    mat_sig(model.a1);

    // Second Layer
    mat_dot(model.a2,model.a1,model.w2);
    mat_sum(model.a2,model.a2,model.b2);
    mat_sig(model.a2);

    // return model.a2.es[0];

}

float cost(Xor model , Matrice x, Matrice y){

    NN_ASSERT(x.rows==y.rows);
    NN_ASSERT(y.cols==model.a2.cols);

    size_t n = x.rows;
    float c = 0.f;


    for (size_t i = 0; i < n; i++)
    {
        Matrice expected = mat_row(y,i);
        Matrice input = mat_row(x,i);

        mat_copy(model.a0,input);
        forward_xor(model);


        for (size_t j = 0; j < expected.cols; j++)
        {
            float d = MAT_AT(model.a2,0,j) - MAT_AT(expected,0,j);
            c+= d*d;
        }
    }

    return c/n;

}

void finite_diff(Xor model,Xor gradient , float eps,Matrice ti,Matrice to){

    float c = cost(model,ti,to);


    // we wiggle W1 params (meaning FIRST LAYERS Ws)

    for (size_t i = 0; i < model.w1.rows; i++)
    {
        for (size_t j = 0; j < model.w1.cols; j++)
        {
            MAT_AT(model.w1,i,j) += eps;
            MAT_AT(gradient.w1,i,j) = (cost(model,ti,to)-c)/eps;
            MAT_AT(model.w1,i,j) -= eps;
        }
    }

    for (size_t i = 0; i < model.b1.rows; i++)
    {
        for (size_t j = 0; j < model.b1.cols; j++)
        {
            MAT_AT(model.b1,i,j) += eps;
            MAT_AT(gradient.b1,i,j) = (cost(model,ti,to)-c)/eps;
            MAT_AT(model.b1,i,j) -= eps;
        }
    }

    for (size_t i = 0; i < model.w2.rows; i++)
    {
        for (size_t j = 0; j < model.w2.cols; j++)
        {
            MAT_AT(model.w2,i,j) += eps;
            MAT_AT(gradient.w2,i,j) = (cost(model,ti,to)-c)/eps;
            MAT_AT(model.w2,i,j) -= eps;
        }
    }

    for (size_t i = 0; i < model.b2.rows; i++)
    {
        for (size_t j = 0; j < model.b2.cols; j++)
        {
            MAT_AT(model.b2,i,j) += eps;
            MAT_AT(gradient.b2,i,j) = (cost(model,ti,to)-c)/eps;
            MAT_AT(model.b2,i,j) -= eps;
        }
    }


    

}


void xor_learn(Xor model , Xor gradient , float rate){
    
    for (size_t i = 0; i < model.w1.rows; i++)
    {
        for (size_t j = 0; j < model.w1.cols; j++)
        {
            MAT_AT(model.w1,i,j) -= rate*MAT_AT(gradient.w1,i,j);
        }
    }

    for (size_t i = 0; i < model.b1.rows; i++)
    {
        for (size_t j = 0; j < model.b1.cols; j++)
        {
            MAT_AT(model.b1,i,j) -= rate*MAT_AT(gradient.b1,i,j);
        }
    }

    for (size_t i = 0; i < model.w2.rows; i++)
    {
        for (size_t j = 0; j < model.w2.cols; j++)
        {
            MAT_AT(model.w2,i,j) -= rate*MAT_AT(gradient.w2,i,j);
        }
    }

    for (size_t i = 0; i < model.b2.rows; i++)
    {
        for (size_t j = 0; j < model.b2.cols; j++)
        {
            MAT_AT(model.b2,i,j) -= rate*MAT_AT(gradient.b2,i,j);
        }
    }


}
#endif
int main(void){

    srand(time(0)/2);


    size_t stride = 3;

    size_t n = sizeof(td)/sizeof(td[0])/stride;

    Matrice ti = {
        .rows=n,
        .cols=2,
        .stride = stride,
        .es=td
    };


    Matrice to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = td+2
    };


    
    
    



    size_t epochs = 3*1e5;
    float eps = 1e-2;
    float rate = 1e-2;

#if 0
    for (size_t i = 0; i < epochs; i++)
    {
    printf("cost  %zu :  %f\n",i,cost(model,ti,to));

    finite_diff(model,gradient,eps,ti,to);
    xor_learn(model,gradient , rate);
        
    }

    printf("%f\n",cost(model,ti,to));


    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(model.a0,0,0) = i;
            MAT_AT(model.a0,0,1) = j;
            forward_xor(model);
            float y = model.a2.es[0];

            printf("%zu ^ %zu : %f\n",i,j,y);
        }
    }


#endif

    size_t arch[] = {2,2,1};
    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN g  = nn_alloc(arch,ARRAY_LEN(arch));
    nn_rand(nn,0,1);
    mat_copy(NN_INPUT(nn),mat_row(ti,1));

    // nn_forward(nn);


    // MAT_PRINT(NN_OUTPUT(nn));
    for (size_t i = 0; i < epochs/1; i++)
    {
        printf("%f\n",nn_cost(nn,ti,to));
        nn_finite_diff(nn,g,eps,ti,to);
        nn_learn(nn,g,rate);
        
    }

    printf("\n\n");

    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            MAT_AT(NN_INPUT(nn),0,0) = i;
            MAT_AT(NN_INPUT(nn),0,1) = j;
            nn_forward(nn);
            printf("%zu ^ %zu = %f\n",i,j,MAT_AT(NN_OUTPUT(nn),0,0));
        }
    }
    
    printf("\n\n");

    NN_PRINT(nn);

    return 0;
}

