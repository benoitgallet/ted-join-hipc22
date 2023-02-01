#include <stdio.h>

#include "utils.h"


void cudaErrCheck_(cudaError_t errCode, const char* file, int line)
{
    if (errCode != cudaSuccess)
    {
        fprintf(stderr, "[Error] ~ In %s (line %d): %s.\n", file, line, cudaGetErrorString(errCode));
    }
}
