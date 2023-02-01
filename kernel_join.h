#ifndef KERNEL_JOIN_H
#define KERNEL_JOIN_H

#include <cuda_fp16.h>

#include "structs.h"
#include "params.h"

__device__ uint64_t getLinearID_nDimensionsGPUKernelAlt(
	unsigned int* indexes,
	unsigned int* dimLen,
	unsigned int nDimensions);

__global__ void convertAndResizeDataset(
    INPUT_DATA_TYPE* in,
    COMPUTE_TYPE* out,
    unsigned int nbQueries);

__global__ void convertDataset(
	INPUT_DATA_TYPE* in,
	half* out,
	unsigned int nbPoints);

__global__ void convertMinArr(
	INPUT_DATA_TYPE* in,
	COMPUTE_TYPE* out);


__global__ void preComputedSquaredCoordinates(
	COMPUTE_TYPE* dataset,
	ACCUM_TYPE* preComputeCoordinates,
	unsigned int nbQueryPoints);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void batchEstimatorKernel(
	unsigned int* N,
	unsigned int* sampleOffset,
	INPUT_DATA_TYPE* database,
	unsigned int* originPointIndex,
	ACCUM_TYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	INPUT_DATA_TYPE* minArr,
	unsigned int* nCells,
	unsigned int* cnt,
	unsigned int* nNonEmptyCells,
	unsigned int* estimatedResult,
	unsigned int* candidatesCounter);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__device__ void evaluateCell(
	unsigned int* nCells,
	unsigned int* indexes,
	struct gridCellLookup* gridCellLookupArr,
	unsigned int* nNonEmptyCells,
	COMPUTE_TYPE* database,
	ACCUM_TYPE* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx,
	unsigned int* nDCellIDs);

__forceinline__ __device__ void evalPoint(
	unsigned int* gridLookupArr,
	int k,
	COMPUTE_TYPE* database,
	ACCUM_TYPE* epsilon,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx);

__forceinline__ __device__ void evalPointILP(
	unsigned int* gridLookupArr,
	int k,
	COMPUTE_TYPE* database,
	ACCUM_TYPE* epsilon,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


__global__ void distanceCalculationGridCuda(
    unsigned int* batchBegin,
    unsigned int* batchSize,
    COMPUTE_TYPE* database,
    unsigned int* originPointIndex,
    ACCUM_TYPE* epsilon,
    struct grid* grid,
    unsigned int* gridLookupArr,
    struct gridCellLookup* gridCellLookupArr,
    COMPUTE_TYPE* minArr,
    unsigned int* nCells,
    unsigned int* cnt,
    unsigned int* nNonEmptyCells,
    int* pointIDKey,
    int* pointInDistVal);


//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\


#if COMPUTE_PREC == 64
__global__ void distanceCalculationGridTensor_multiQueryPoints_double_8_8_4_tensor_mixed(
	unsigned int* batchBegin,
	unsigned int* batchEnd,
	double* database,
	unsigned int* nbQueryPoints,
	unsigned int* originPointIndex,
	unsigned int* tensorBatches,
	unsigned int* tensorBatchesSize,
	double* preComputedSquaredCoordinates,
	double* epsilon,
	struct grid* grid,
	unsigned int* gridLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	double* minArr,
	unsigned int* nCells,
	unsigned int* cnt,
	unsigned int* nNonEmptyCells,
	int* pointIDKey,
	int* pointInDistVal);
#endif

#endif // ifndef
