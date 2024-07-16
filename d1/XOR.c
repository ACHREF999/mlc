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


sample *train = xor_train;



typedef struct {
    float or_w1;
    float or_w2;
    float or_b;

    float nand_w1;
    float nand_w2;
    float nand_b;
    
    float and_w1;
    float and_w2;
    float and_b;
}Xor;


#define train_count 4

float sigf(float x){
    return 1.f/(1.f+expf(-x));
}


float forward(Xor m , float x , float y){

    float a = sigf(m.or_w1*x + m.or_w2*y + m.or_b);
    float b = m.nand_w1*x + m.nand_w2*y + m.nand_b;
    float result = m.and_w1*a + m.and_w2*b + m.and_b;

    return result;
}   

// we pass the xor model params and calculate the average erro based on the training dataset
float loss(Xor m){
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y  = forward(m,x1,x2);
        float d  = y-train[i][2];
        result += d*d;

    }
    result /= train_count;

    return result;
}


float rand_float(void){

    return (float) rand()/(float)RAND_MAX;

}


Xor rand_xor(void){

    Xor m = {};

    m.and_b = rand_float();
    m.and_w1 = rand_float();
    m.and_w2 = rand_float();
    m.or_b = rand_float();
    m.or_w1 = rand_float();
    m.or_w2 = rand_float();
    m.nand_w1 = rand_float();
    m.nand_b = rand_float();
    m.nand_w2 = rand_float();
    
    return m;
}


void print_xor(Xor m ){
    printf("or_w1 = %f\n",m.or_w1);
    printf("or_w2 = %f\n",m.or_w2);
    printf("or_b = %f\n",m.or_b);
    printf("and_w1 = %f\n",m.and_w1);
    printf("and_w2 = %f\n",m.and_w2);
    printf("and_b = %f\n",m.and_b);
    printf("nand_w1 = %f\n",m.nand_w1);
    printf("nand_w2 = %f\n",m.nand_w2);
    printf("nand_b = %f\n",m.nand_b);
}



Xor finite_diff(Xor m,float eps){
    Xor diff;
    float l = loss(m);

    m.or_b += eps;
    diff.or_b = (loss(m)-l)/eps;
    m.or_b -= eps;

    
    m.or_w1 += eps;
    diff.or_w1 = (loss(m)-l)/eps;
    m.or_w1 -= eps;


    m.or_w2 += eps;
    diff.or_w2 = (loss(m)-l)/eps;
    m.or_w2 -= eps;
    
    m.and_b += eps;
    diff.and_b = (loss(m)-l)/eps;
    m.and_b -= eps;
    
    m.and_w1 += eps;
    diff.and_w1 = (loss(m)-l)/eps;
    m.and_w1 -= eps;
    
    m.and_w2 += eps;
    diff.and_w2 = (loss(m)-l)/eps;
    m.and_w2 -= eps;
    
    m.nand_b += eps;
    diff.nand_b = (loss(m)-l)/eps;
    m.nand_b -= eps;
    
    m.nand_w1 += eps;
    diff.nand_w1 = (loss(m)-l)/eps;
    m.nand_w1 -= eps;
    
    m.nand_w2 += eps;
    diff.nand_w2 = (loss(m)-l)/eps;
    m.nand_w2 -= eps;


    return diff;
}


Xor learn(Xor m , Xor diff,float rate)
{

    Xor result ;
    result.or_w1 = m.or_w1 - rate*diff.or_w1;
    result.or_w2 = m.or_w2 - rate*diff.or_w2;
    result.or_b = m.or_b - rate*diff.or_b;
    result.nand_w1 = m.nand_w1 - rate*diff.nand_w1;
    result.nand_w2 = m.nand_w2 - rate*diff.nand_w2;
    result.nand_b = m.nand_b - rate*diff.nand_b;
    result.and_w1 = m.and_w1 - rate*diff.and_w1;
    result.and_w2 = m.and_w2 - rate*diff.and_w2;
    result.and_b = m.and_b - rate*diff.and_b;


    return result;

}

int main(void){

    Xor m = rand_xor();

    float eps = 1e-2;
    float rate =1e-1;
    float epochs = 1*1e5;



    for (size_t i = 0; i < epochs; i++)
    {
    Xor diff = finite_diff(m,eps);
    m = learn(m,diff,rate);
    printf("Cost  : %f\n",loss(m));
    }
    // printf("Cost  : %f\n",loss(learn(m,diff,rate)));

    // print_xor(m);


    printf("-------------XOR--------------\n");
    for (size_t i = 0; i < (int)train_count; i++)
    {
        
            printf("%f ^ %f : %f\n",train[i][0],train[i][1],forward(m,train[i][0],train[i][1]));
        
    }
    printf("-------------OR--------------\n");

    for (size_t i = 0; i < (int)train_count; i++)
    {
        
            printf("%f | %f : %f\n",train[i][0],train[i][1],sigf(m.or_b + m.or_w1*train[i][0] + m.or_w2*train[i][1]));
        
    }

    printf("------------AND---------------\n");

    for (size_t i = 0; i < (int)train_count; i++)
    {
        
            printf("%f | %f : %f\n",train[i][0],train[i][1],sigf(m.and_b + m.and_w1*train[i][0] + m.and_w2*train[i][1]));
        
    }


    printf("-------------NAND--------------\n");

    for (size_t i = 0; i < (int)train_count; i++)
    {
        
            printf("%f | %f : %f\n",train[i][0],train[i][1],sigf(m.nand_b + m.nand_w1*train[i][0] + m.nand_w2*train[i][1]));
        
    }


    // NOTICE THAT : IT IS NOT CREATING A CIRCUIT BUT RATHER SATISFYING  A MATHEMATICAL (could be continous) SYSTEM SO THAT THE FINAL OUTPUT IS XOR
    return 0;
}