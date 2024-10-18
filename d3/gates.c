#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef  float sample[3]  ;

sample nand_train[] = {
    {0,0,1},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};

sample or_train[]={
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1}
};

sample and_train[]={
    {0,0,0},
    {0,1,0},
    {1,0,0},
    {1,1,1}
};

sample xor_train[]={
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0}
};


sample *train = or_train;


// #define train_count (sizeof(train)/sizeof(train[0]))
#define train_count 4


float sigf(float x) {
    return 1.f/(1.f + expf(-x));
}



float loss(float w1 , float w2,float c){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigf(x1*w1 + x2*w2 + c);
        float d = y - train[i][2];
        result += d*d;
    }
    
    result /= train_count;
    return result;

}

void fdcost(float w1 , float w2, float c , float *dw1 , float *dw2 , float *dc,float eps){
        float curr_cost = loss(w1,w2,c);
        *dw1  = (loss(w1+eps,w2,c)-curr_cost)/eps;
        *dw2  = (loss(w1,w2+eps,c)-curr_cost)/eps;
        *dc  = (loss(w1,w2,c+eps)-curr_cost)/eps;
}

void gcost(float w1,float w2,float c ,float *dw1 , float *dw2 , float *dc){
    *dw1 = 0;
    *dw2 = 0;
    *dc = 0;
    size_t n = train_count;
    for (size_t i = 0; i < n; i++)
    {
        float xi = train[i][0];
        float yi = train[i][1];
        float zi = train[i][2];
        float ai = sigf(w1*xi + w2*yi+c);
        *dw1 += 2*(ai-zi)*ai*(1-ai)*xi;
        *dw2 += 2*(ai-zi)*ai*(1-ai)*yi;
        *dc += 2*(ai-zi)*ai*(1-ai )*1;
    }   
    *dw1 /=n;
    *dw2 /=n;
    *dc /=n;
}


float rand_float(void){

    return (float) rand()/(float)RAND_MAX;

}





int main(void){
    train = or_train ;

    srand(69);
    float w1 = rand_float();
    float w2 = rand_float();
    float c = rand_float();
    
    float eps = 5*1e-2;
    float rate = 8 * 1e-2;
    size_t epochs = 5*  1e+3;

    for (size_t i = 0; i < epochs; i++)
    {
        float dw1, dw2,dc;
        #if 0
        fdcost(w1,w2,c,&dw1 , &dw2 ,&dc,eps);
        #else
        gcost(w1,w2,c,&dw1,&dw2,&dc);
        #endif
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        c  -= rate*dc;

        printf("------%zu-------\n",i);
        printf("--------------------/----------------------------------------------------------------------\nLoss\t:\t%f\tw2\t:\t%f\tw2\t:\t%f\tc\t:\t%f\n",loss(w1,w2,c),w1,w2,c);
    }


    for (size_t i = 0; i < (int)train_count; i++)
    {
        
            printf("%f | %f : %f\n",train[i][0],train[i][1],sigf(w1*train[i][0]+w2*train[i][1]+c));
        
    }

    return 0;

}
