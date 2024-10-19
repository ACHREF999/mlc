#include <stdio.h>
#include <stdlib.h>
#define NN_IMPLEMENTATION
#include "nn.h"

#define BITS 3


int main(void) {
    size_t n = (1<<BITS);
    size_t rows = (1<<BITS) * (1<<BITS);
    Matrice ti = mat_alloc(rows,2*BITS);
    Matrice to = mat_alloc(rows , BITS+1);

    for (size_t i = 0; i < ti.rows; i++)
    {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x+y;
        MAT_AT(to,i,BITS) = z>=n;
        for (size_t j = 0; j < BITS; j++)
        {
            if(MAT_AT(to,i,BITS)){
                MAT_AT(to,i,j) = 0;
            }
            else{
                MAT_AT(to,i,j) = (z>>j)&1;
            }
            // do we set the other bits to zero when overflow ?
            MAT_AT(to,i,j) = (z>>j)&1;

            MAT_AT(ti,i,j) = (x>>j)&1;
            MAT_AT(ti,i,j+BITS) = (y>>j)&1;
        }
    }

    size_t arch[] = {2*BITS,2*BITS+4, BITS+1};
    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN g = nn_alloc(arch,ARRAY_LEN(arch));
    nn_rand(nn,0,1);
    NN_PRINT(nn);
    

    float rate = 0.1;

    for (size_t i = 0; i < 50 * 1000; i++)
    {
        
        printf("c = %f\n",nn_cost(nn,ti,to));
        nn_backprop(nn,g,ti,to);
        nn_learn(nn,g,rate);
    
    }

    for (size_t x = 0; x < n; x++)
    {
        for (size_t y = 0; y < n; y++)
        {
            for (size_t j = 0; j < BITS; j++)
            {
                MAT_AT(NN_INPUT(nn),0,j) = (x>>j)&1;
                MAT_AT(NN_INPUT(nn),0,j+BITS) = (y>>j)&1;
            }
            nn_forward(nn);
            size_t z = 0;
            for (size_t j = 0; j < BITS; j++)
            {
                size_t bit = MAT_AT(NN_OUTPUT(nn),0,j) > 0.5f;
                z |= bit<<j;
            }
            if(MAT_AT(NN_OUTPUT(nn),0,BITS)>0.5f){
                z = -1;
            }
            if(z==-1){
                printf("%zu + %zu = overflows\n",x,y);
            }else{
                printf("%zu + %zu = %zu\n",x,y,z);
            }
        }
    }
    return 0;
}