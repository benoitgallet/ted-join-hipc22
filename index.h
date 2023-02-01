#ifndef INDEX_H
#define INDEX_H

#include <vector>

#include "params.h"


__global__ void kernelIndexComputeNonEmptyCells(
	INPUT_DATA_TYPE* database,
	unsigned int* nbQueryPoints,
	ACCUM_TYPE* epsilon,
	INPUT_DATA_TYPE* minArr,
	unsigned int* nCells,
	uint64_t* pointCellArr,
	unsigned int* databaseVal,
	bool enumerate);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


void reorderDimensionsByVariance(
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector);

void generateNDGridDimensions(
    std::vector< std::vector <INPUT_DATA_TYPE> >* inputVector,
    ACCUM_TYPE epsilon,
    INPUT_DATA_TYPE* minArr,
    INPUT_DATA_TYPE* maxArr,
    unsigned int* nCells,
    uint64_t* totalCells);

void gridIndexingGPU(
    unsigned int* nbQueryPoints,
    uint64_t totalCells,
    INPUT_DATA_TYPE* database,
    INPUT_DATA_TYPE** dev_database,
    ACCUM_TYPE* epsilon,
    ACCUM_TYPE** dev_epsilon,
    INPUT_DATA_TYPE* minArr,
    INPUT_DATA_TYPE** dev_minArr,
    struct grid** index,
    struct grid** dev_index,
    unsigned int* indexLookupArr,
    unsigned int** dev_indexLookupArr,
    struct gridCellLookup** gridCellLookupArr,
    struct gridCellLookup** dev_gridCellLookupArr,
    unsigned int* nNonEmptyCells,
    unsigned int** dev_nNonEmptyCells,
    unsigned int* nCells,
    unsigned int** dev_nCells);

#endif
