#ifndef GPU_JOIN_H
#define GPU_JOIN_H

#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

#include "params.h"
#include "structs.h"


boost::multiprecision::uint128_t GPUBatchEst(
    unsigned int* nbQueryPoints,
    INPUT_DATA_TYPE* dev_database,
    unsigned int* dev_originPointIndex,
    ACCUM_TYPE* dev_epsilon,
    struct grid* dev_grid,
    unsigned int* dev_gridLookupArr,
    struct gridCellLookup* dev_gridCellLookupArr,
    INPUT_DATA_TYPE* dev_minArr,
    unsigned int* dev_nCells,
    unsigned int* dev_nNonEmptyCells,
    unsigned int* retNumBatches,
    unsigned int* retGPUBufferSize,
    std::vector<struct batch>* batches);


void GPUJoinMainIndex(
    int searchMode,
    INPUT_DATA_TYPE* dataset,
    INPUT_DATA_TYPE* dev_database,
    unsigned int* nbQueryPoints,
    ACCUM_TYPE* epsilon,
    ACCUM_TYPE* dev_epsilon,
    struct grid* grid,
    struct grid* dev_grid,
    unsigned int* gridLookupArr,
    unsigned int* dev_gridLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    struct gridCellLookup* dev_gridCellLookupArr,
    INPUT_DATA_TYPE* minArr,
    INPUT_DATA_TYPE* dev_minArr,
    unsigned int* nCells,
    unsigned int* dev_nCells,
    unsigned int* nNonEmptyCells,
    unsigned int* dev_nNonEmptyCells,
    unsigned int* originPointIndex,
    unsigned int* dev_originPointIndex,
    struct neighborTableLookup* neighborTable,
    std::vector<struct neighborDataPtrs>* pointersToNeighbors,
    uint64_t* totalNeighbors,
    uint64_t* totalNeighborsCuda,
    uint64_t* totalNeighborsTensor,
    boost::multiprecision::uint128_t* totalCandidatesCuda,
    boost::multiprecision::uint128_t* totalCandidatesTensor,
    unsigned int* totalQueriesCuda,
    unsigned int* totalQueriesTensor,
    unsigned int* totalKernelsCuda,
    unsigned int* totalKernelsTensor,
    struct schedulingCell* sortedCells);


void constructNeighborTableKeyValueWithPtrs(
    int* pointIDKey,
    int* pointInDistValue,
    struct neighborTableLookup* neighborTable,
    int* pointersToNeighbors,
    unsigned int* cnt);

#endif
