#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float train[][2]={
    {0,0},
    {1,2},
    {2,4},
    {3,6},
    {4,8},

};

#define train_count  (sizeof(train)/sizeof(train[0]))
// #define rate 0.01

float rand_float(void){
    return (float) rand()/(float)RAND_MAX;

}


float loss (float w){
    float result = 0.0f;
        // does the model satisfies the data ? 

    for (size_t i = 0; i < train_count; i++)
    {
        float x = train[i][0];
        float y = x*w;
        float d = y-train[i][1];
        result += d*d;

    }
    result /= train_count;
    return result;
}


float dcost(float w){
    float result = 0.0f ;
    size_t  n = train_count;
    for (size_t i = 0; i < n; i++)
    {
        float x = train[i][0];
        float y = train[i][1];
        result += 2 * x * (w*x - y);
    } 

    result /= n;
    return result;
}

int main(void){

    // the only information we got is that the model has some form of : 
    // y = x *w {+c};
    //compression ? 
    // srand(time(0));
    srand(69);

    float w = rand_float() * 10.0f;
    // float c = rand_float()* 5.0f;
    printf("supus %f\n",w);

    // float eps = 1e-3;
    float rate = 1e-2;
    size_t epochs = 10;
    for (size_t i = 0; i < epochs; i++)
    {
        float current_loss = loss(w);
        // we wiggle it arround
        float dw = dcost(w);
        // float dw = (loss(w+eps,c)-current_loss)/eps;
        // float dc = (loss(w,c+eps)-current_loss)/eps;
        // float dc = dcost(w);
        //notice that wiggling by derivatives only is not so good as they are big for adjustmentent
        w -=rate*dw;
        // c -=rate*dc;

        printf("------%ld-------\n",i);
        // printf("------------------------------------------------------------------------------------------\nLoss\t:\t%f\tweight\t:\t%f\tbias\t:\t%f\n",loss(w,c),w,c);
        
        printf("------------------------------------------------------------------------------------------\nLoss\t:\t%f\tweight\t:\t%f\t\n",loss(w),w);
        
    }

    return 0;
}














