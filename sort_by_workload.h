#ifndef SORT_BY_WORKLOAD_H
#define SORT_BY_WORKLOAD_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "params.h"
#include "structs.h"

__global__ void sortByWorkLoadGlobal(
	INPUT_DATA_TYPE* database,
	ACCUM_TYPE* epsilon,
	struct grid* index,
	unsigned int* indexLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	INPUT_DATA_TYPE* minArr,
	unsigned int* nCells,
	unsigned int* nNonEmptyCells,
	schedulingCell* sortedCells);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


void sortByWorkLoad(
    unsigned int searchMode,
    unsigned int* nbQueryPoints,
    boost::multiprecision::uint128_t* totalCandidates,
    struct schedulingCell** sortedDatabaseTmp,
    ACCUM_TYPE* epsilon,
    ACCUM_TYPE** dev_epsilon,
    INPUT_DATA_TYPE* database,
    INPUT_DATA_TYPE** dev_database,
    struct grid* index,
    struct grid** dev_index,
    unsigned int* indexLookupArr,
    unsigned int** dev_indexLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    struct gridCellLookup** dev_gridCellLookupArr,
    INPUT_DATA_TYPE* minArr,
    INPUT_DATA_TYPE** dev_minArr,
    unsigned int* nCells,
    unsigned int** dev_nCells,
    unsigned int* nNonEmptyCells,
    unsigned int** dev_nNonEmptyCells,
    unsigned int** originPointIndex,
    unsigned int** dev_originPointIndex;

#endif
