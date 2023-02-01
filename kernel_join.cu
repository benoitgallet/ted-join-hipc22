#include <math.h>
#include <stdio.h>

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "kernel_join.h"
#include "params.h"
#include "structs.h"

// Specific to tensor cores
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace cooperative_groups;


__device__ uint64_t getLinearID_nDimensionsGPUKernelAlt(
	unsigned int* indexes,
	unsigned int* dimLen,
	unsigned int nDimensions)
{
    uint64_t offset = 0;
	uint64_t multiplier = 1;

	for (int i = 0; i < nDimensions; ++i)
	{
		offset += (uint64_t) indexes[i] * multiplier;
		multiplier *= dimLen[i];
	}

	return offset;
}


__global__ void convertAndResizeDataset(
    INPUT_DATA_TYPE* in,
    COMPUTE_TYPE* out,
    unsigned int nbQueries)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nbQueries)
    {
	    // Copy the coordinates from the dataset
        for (int i = 0; i < INPUT_DATA_DIM; ++i)
        {
            out[tid * COMPUTE_DIM + i] = (COMPUTE_TYPE)in[tid * INPUT_DATA_DIM + i];
        }
		// Fill with 0s so the dimensionality of the dataset is compatible with the tensor cores
		for (int i = INPUT_DATA_DIM; i < COMPUTE_DIM; ++i)
		{
			out[tid * COMPUTE_DIM + i] = (COMPUTE_TYPE)0.0;
		}
    }
    // The original dataset does not have the 15 supplemental points, so need to do it in another step
    if (tid < ADDITIONAL_POINTS)
    {
	    // Create "fake points" with 0s coordinates so the last query point will still have 15 points after when loading using load_matrix_sync
		for (int i = 0; i < COMPUTE_DIM; ++i)
		{
			out[tid * COMPUTE_DIM + i] = (COMPUTE_TYPE)0.0;
		}
    }
}


__global__ void convertDataset(
	INPUT_DATA_TYPE* in,
	half* out,
	unsigned int nbPoints)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nbPoints)
	{
		for (unsigned int i = 0; i < COMPUTE_DIM; ++i)
		{
			out[tid * COMPUTE_DIM + i] = (half)(in[tid * COMPUTE_DIM + i]);
		}
	}
}


__global__ void convertMinArr(
	INPUT_DATA_TYPE* in,
	COMPUTE_TYPE* out)
{
	for (int i = 0; i < INDEXED_DIM; ++i)
	{
		out[i] = (COMPUTE_TYPE)in[i];
	}
}


__global__ void preComputedSquaredCoordinates(
	COMPUTE_TYPE* dataset,
	ACCUM_TYPE* preComputeCoordinates,
	unsigned int nbQueryPoints)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (nbQueryPoints <= tid)
	{
		return;
	}

	#if ACCUM_PREC == 64
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
		{
			double a = dataset[tid * COMPUTE_DIM + i] * dataset[tid * COMPUTE_DIM + i];
			double b = dataset[tid * COMPUTE_DIM + i + 1] * dataset[tid * COMPUTE_DIM + i + 1];
			double c = dataset[tid * COMPUTE_DIM + i + 2] * dataset[tid * COMPUTE_DIM + i + 2];
			double d = dataset[tid * COMPUTE_DIM + i + 3] * dataset[tid * COMPUTE_DIM + i + 3];
			preComputeCoordinates[tid * (COMPUTE_DIM / 4) + (i / 4)] = a + b + c + d;
		}
	#else
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 16)
		{
            ACCUM_TYPE accum = (ACCUM_TYPE)0.0;
            
            #pragma unroll
            for (unsigned int j = 0; j < 16; ++j)
            {
                accum += (ACCUM_TYPE)(dataset[tid * COMPUTE_DIM + i + j]) * (ACCUM_TYPE)(dataset[tid * COMPUTE_DIM + i + j]);
            }
            preComputeCoordinates[tid * (COMPUTE_DIM / 16) + (i / 16)] = accum;
		}
	#endif
}


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
	unsigned int* candidatesCounter)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ((*N) <= tid)
	{
		return;
	}

	unsigned int pointId = tid * (*sampleOffset);

	INPUT_DATA_TYPE point[INPUT_DATA_DIM];
	for (int i = 0; i < INPUT_DATA_DIM; ++i)
	{
		point[i] = database[ originPointIndex[pointId] * COMPUTE_DIM + i ];
	}

	unsigned int nDCellIDs[INDEXED_DIM];
	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

	for (int i = 0; i < INDEXED_DIM; ++i)
	{
		nDCellIDs[i] = (point[i] - minArr[i]) / (INPUT_DATA_TYPE)(*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1);
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1);
	}

	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	unsigned int localNeighborCounter = 0;
	unsigned int localCandidateCounter = 0;

	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < INDEXED_DIM; ++x)
			{
				indexes[x] = loopRng[x];
			}

			uint64_t calcLinearID = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, INDEXED_DIM);
			//compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
			//a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)

			struct gridCellLookup tmp;
			tmp.gridLinearID = calcLinearID;

			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;

				for (int k = grid[GridIndex].indexmin; k <= grid[GridIndex].indexmax; ++k)
				{
					ACCUM_TYPE runningTotalDist = 0;
					unsigned int dataIdx = gridLookupArr[k];

					for (int l = 0; l < INPUT_DATA_DIM; ++l)
					{
						runningTotalDist += (ACCUM_TYPE)((database[dataIdx * COMPUTE_DIM + l]  - point[l])
								* (database[dataIdx * COMPUTE_DIM + l] - point[l]));
					}

					#if ACCUM_PREC == 16
					if (hsqrt(runningTotalDist) <= (*epsilon))
					#else
					if (sqrt(runningTotalDist) <= (*epsilon))
					#endif
					{
						unsigned int idx = atomicAdd(cnt, int(1));
						localNeighborCounter++;
					}
				}

				localCandidateCounter += grid[GridIndex].indexmax - grid[GridIndex].indexmin + 1;
			}
		} //end loop body

	estimatedResult[tid] = localNeighborCounter;
	candidatesCounter[tid] = localCandidateCounter;
}


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
	unsigned int* nDCellIDs)
{
	// Compare the linear ID with the gridCellLookupArr to determine if the cell is non-empty: this can happen because one point says
	// a cell in a particular dimension is non-empty, but that's because it was related to a different point (not adjacent to the query point)
	uint64_t calcLinearID = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, INDEXED_DIM);

	struct gridCellLookup tmp;
	tmp.gridLinearID = calcLinearID;
	//find if the cell is non-empty
	if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
	{
		//compute the neighbors for the adjacent non-empty cell
		struct gridCellLookup* resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
                gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
		unsigned int GridIndex = resultBinSearch->idx;

		// unsigned int nbCandidates = grid[GridIndex].indexmax - grid[GridIndex].indexmin + 1;
		// atomicAdd(cnt, nbCandidates);

		for(int k = grid[GridIndex].indexmin; k <= grid[GridIndex].indexmax; ++k)
        {
			#if ILP == 1
				evalPoint(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
            #else
                #if KAHAN_CUDA
                    evalPointILPKahan(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
                #else
                    evalPointILP(gridLookupArr, k, database, epsilon, point, cnt, pointIDKey, pointInDistVal, pointIdx);
                #endif
			#endif
		}
	}
}


__forceinline__ __device__ void evalPoint(
	unsigned int* gridLookupArr,
	int k,
	COMPUTE_TYPE* database,
	ACCUM_TYPE* epsilon,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	ACCUM_TYPE runningTotalDist = 0;
	unsigned int dataIdx = gridLookupArr[k];

    half cHalf = (half)0.0;

//    unsigned int print = 1;
//    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = 0; i < INPUT_DATA_DIM; ++i)
	{
        runningTotalDist += (ACCUM_TYPE)((database[dataIdx * COMPUTE_DIM + i] - point[i]) * (database[dataIdx * COMPUTE_DIM + i] - point[i]));

        #if SHORT_CIRCUIT
            #if ACCUM_PREC == 16
            if (hsqrt(runningTotalDist) > (*epsilon))
            #else
            if (sqrt(runningTotalDist) > (*epsilon))
            #endif
            {
                return;
            }
        #endif
    }

	#if ACCUM_PREC == 16
	if(hsqrt(runningTotalDist) <= (*epsilon))
    #else
    if(sqrt(runningTotalDist) <= (*epsilon))
    #endif
    {
		unsigned int idx = atomicAdd(cnt, int(1));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
	}
}


__forceinline__ __device__ void evalPointILP(
	unsigned int* gridLookupArr,
	int k,
	COMPUTE_TYPE* database,
	ACCUM_TYPE* epsilon,
	COMPUTE_TYPE* point,
	unsigned int* cnt,
	int* pointIDKey,
	int* pointInDistVal,
	int pointIdx)
{
	unsigned int dataIdx = gridLookupArr[k];
	ACCUM_TYPE runningTotalDist[ILP];

	#pragma unroll
	for (int i = 0; i < ILP; ++i)
	{
		runningTotalDist[i] = 0.0;
	}

	for(int i = 0; i < INPUT_DATA_DIM; i += ILP)
    {
		#pragma unroll
		for (int j = 0; j < ILP && (i + j) < INPUT_DATA_DIM; ++j)
		{
			runningTotalDist[j] += (ACCUM_TYPE)((database[dataIdx * COMPUTE_DIM + i + j] - point[i + j])
											* (database[dataIdx * COMPUTE_DIM + i + j] - point[i + j]));
		}

		#if SHORT_CIRCUIT
			#pragma unroll
			for (int j = 1; j < ILP; ++j)
			{
				runningTotalDist[0] += runningTotalDist[j];
				runningTotalDist[j] = (ACCUM_TYPE)0.0;
			}

			#if ACCUM_PREC == 16
			if (hsqrt(runningTotalDist[0]) > (*epsilon))
			#else
			if (sqrt(runningTotalDist[0]) > (*epsilon))
			#endif
			{
				return;
			}
		#endif
	}

	#if !SHORT_CIRCUIT
		#pragma unroll
		for (int i = 1; i < ILP; ++i)
		{
			runningTotalDist[0] += runningTotalDist[i];
		}
	#endif

    // if(runningTotalDist <= ((*epsilon) * (*epsilon)))
    #if ACCUM_PREC == 16
	if(hsqrt(runningTotalDist[0]) <= (*epsilon))
    #else
    if(sqrt(runningTotalDist[0]) <= (*epsilon))
    #endif
    {
		unsigned int idx = atomicAdd(cnt, int(1));
		pointIDKey[idx] = pointIdx;
		pointInDistVal[idx] = dataIdx;
	}
}


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
    int* pointInDistVal)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ((*batchSize) <= tid)
    {
        return;
    }

    // Get the next query point in the "local" queue
    unsigned int pointId = atomicAdd(batchBegin, int(1));

    COMPUTE_TYPE point[INPUT_DATA_DIM];
    for (int i = 0; i < INPUT_DATA_DIM; ++i)
    {
        point[i] = database[ originPointIndex[pointId] * COMPUTE_DIM + i ];
    }

    // Calculate the coords of the Cell for the point and the min/max ranges in each dimension
	unsigned int nDCellIDs[INDEXED_DIM];
    unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

    for (int i = 0; i < INDEXED_DIM; ++i)
    {
        nDCellIDs[i] = (ACCUM_TYPE)(point[i] - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1); // Boundary conditions (don't go beyond cell 0)
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1); // Boundary conditions (don't go beyond the maximum number of cells)
    }

    unsigned int indexes[INDEXED_DIM];
    unsigned int loopRng[INDEXED_DIM];

    for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body

			for (int x = 0; x < INDEXED_DIM; x++)
			{
				indexes[x] = loopRng[x];
			}

			evaluateCell(nCells, indexes, gridCellLookupArr, nNonEmptyCells, database, epsilon, grid,
					gridLookupArr, point, cnt, pointIDKey, pointInDistVal, originPointIndex[pointId], nDCellIDs);

		} //end loop body
}


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
	int* pointInDistVal)
{
	__shared__ double sharedArrayQueryPoints[WARP_PER_BLOCK * 8 * COMPUTE_DIM];
	__shared__ double sharedArrayTmp8x4[WARP_PER_BLOCK * 8 * 4];
	__shared__ double sharedArraySquaredQueries[WARP_PER_BLOCK * 8 * (COMPUTE_DIM / 4)];
	__shared__ double sharedArraySquaredCandidates[WARP_PER_BLOCK * 8];
	__shared__ double sharedArrayResultTmp[WARP_PER_BLOCK * 8 * 8];
	__shared__ double sharedArrayResult[WARP_PER_BLOCK * 8 * 8];

	unsigned int print = 1;
	unsigned int batchToPrint = 0;

	unsigned int warpIdInBlock = threadIdx.x / WARP_SIZE;

	unsigned int sharedArrayQueryOffset = warpIdInBlock * 8 * COMPUTE_DIM;
	unsigned int sharedArray8x4Offset = warpIdInBlock * 8 * 4;
	unsigned int sharedArraySquaredOffset = warpIdInBlock * 8 * (COMPUTE_DIM / 4);
	unsigned int sharedArrayOffset = warpIdInBlock * 8 * 8;

	// Thread block with the full warp (32 threads)
	thread_block_tile<32> warp = tiled_partition<32>(this_thread_block());
	// Thread blocks of 8 threads
	thread_block_tile<8> tile8 = tiled_partition<8>(warp);
	// Thread blocks of 4 threads
	thread_block_tile<4> tile4 = tiled_partition<4>(warp);

	unsigned int batchIndex;
	if (0 == warp.thread_rank())
	{
		batchIndex = atomicAdd(batchBegin, int(1));
	}
	batchIndex = __shfl_sync(0xffffffff, batchIndex, 0);

	if ((*batchEnd) <= batchIndex)
	{
		return;
	}

	#if OUTPUT_DEBUG_TENSOR
		if (0 == batchIndex)
		{
			printf("Thread %d: Tile4 group %d, thread %d, tile8 group %d, thread %d\n", warp.thread_rank(),
				tile4.meta_group_rank(), tile4.thread_rank(), tile8.meta_group_rank(), tile8.thread_rank());
		}
	#endif

	unsigned int nbStepsToPage = ceil((1.0 * COMPUTE_DIM) / (1.0 * WARP_SIZE));
	unsigned int nbQueriesBatch = tensorBatches[batchIndex + 1] - tensorBatches[batchIndex];

	// Page query points
	if (tile4.meta_group_rank() < nbQueriesBatch)
	{
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
		{
			if ((tile4.thread_rank() + i) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[sharedArrayQueryOffset + tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() + i] =
					database[originPointIndex[tensorBatches[batchIndex] + tile4.meta_group_rank()] * COMPUTE_DIM + tile4.thread_rank() + i];
			}
		}
	} else {
		for (unsigned int i = 0; i < COMPUTE_DIM; i += 4)
		{
			if ((tile4.thread_rank() + i) < COMPUTE_DIM)
			{
				sharedArrayQueryPoints[sharedArrayQueryOffset + tile4.meta_group_rank() * COMPUTE_DIM + tile4.thread_rank() + i] = 0.0;
			}
		}
	}

	#if OUTPUT_DEBUG_TENSOR
		if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
		{
			printf("\nQuery points: \n");
			for (unsigned int i = 0; i < 8; ++i)
			{
				printf("Query %d: ", i);
				for (unsigned int j = 0; j < COMPUTE_DIM; ++j)
				{
					printf("%f, ", sharedArrayQueryPoints[sharedArrayQueryOffset + i * COMPUTE_DIM + j]);
				}
				printf("\n");
			}
		}
	#endif

	// Page query points' pre-computed squared coordinates
	// For simplicity, only the 8 first threads of the warp page the squared queries' coordinates into shared memory
	// Should be optimized a bit so each tile4 takes care of a query point instead
	// The 8 squared coordinates of the 8 queries are stored next to each other to improve memory accesses performance
	// E.g.: a0 b0 c0 d0 e0 f0 g0 h0 a1 b1 c1 d1 etc.
	if (warp.thread_rank() < 8)
	{
		for (unsigned int i = 0; i < (COMPUTE_DIM / 4); ++i)
		{
			if (warp.thread_rank() < nbQueriesBatch)
			{
				sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] =
					preComputedSquaredCoordinates[originPointIndex[tensorBatches[batchIndex] + warp.thread_rank()] * (COMPUTE_DIM / 4) + i];
			} else {
				sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + warp.thread_rank()] = 0.0;
			}
		}
	}

	#if OUTPUT_DEBUG_TENSOR
		if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
		{
			printf("\nSquared queries (Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7): \n");
			for (unsigned int i = 0; i < (COMPUTE_DIM / 4); ++i)
			{
				for (unsigned int j = 0; j < 8; ++j)
				{
					printf("%f, ", sharedArraySquaredQueries[sharedArraySquaredOffset + i * 8 + j]);
				}
				printf("\n");
			}
		}
	#endif

	unsigned int nDCellIDs[INDEXED_DIM];
	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

	for (unsigned int i = 0; i < INDEXED_DIM; ++i)
	{
		double queryCoordinateTmp = sharedArrayQueryPoints[sharedArrayQueryOffset + i];
		nDCellIDs[i] = (queryCoordinateTmp - minArr[i]) / (*epsilon);
		nDMinCellIDs[i] = max(0, nDCellIDs[i] - 1);
		nDMaxCellIDs[i] = min(nCells[i] - 1, nDCellIDs[i] + 1);
	}

	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	// Iterate over neighboring cells
	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{ //beginning of loop body
			for (unsigned int x = 0; x < INDEXED_DIM; ++x)
			{
				indexes[x] = loopRng[x];
			}

			uint64_t cellLinearId = getLinearID_nDimensionsGPUKernelAlt(indexes, nCells, INDEXED_DIM);
			struct gridCellLookup tmp;
			tmp.gridLinearID = cellLinearId;

			// Find if the neighboring cell is empty or not
			if(thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				struct gridCellLookup* resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr,
					gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int gridIndex = resultBinSearch->idx;

				// For all the candidates in the cell, process them 8 by 8
				for (unsigned int k = grid[gridIndex].indexmin; k <= grid[gridIndex].indexmax; k += 8)
				{
					unsigned int nbCandidatesLeft = grid[gridIndex].indexmax - k + 1;
					unsigned int nbCandidatesCurrent = min(8, nbCandidatesLeft);

					// Result matrix is 8x8 (64), so use the full warp twice
					sharedArrayResult[sharedArrayOffset + warp.thread_rank()] = 0.0;
					sharedArrayResult[sharedArrayOffset + 32 + warp.thread_rank()] = 0.0;

					// Process dimensions 4 by 4
					for (unsigned int n = 0; n < COMPUTE_DIM; n += 4)
					{
						#if OUTPUT_DEBUG_TENSOR
							if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
							{
								printf("\n\n\n===== n = %d =====\n", n);
							}
						#endif

						// Page the candidate points
						if ((k + tile4.meta_group_rank()) < (*nbQueryPoints))
						{
							unsigned int candidateId = gridLookupArr[k + tile4.meta_group_rank()];
							sharedArrayTmp8x4[sharedArray8x4Offset + tile4.meta_group_rank() * 4 + tile4.thread_rank()] = 
								database[candidateId * COMPUTE_DIM + n + tile4.thread_rank()];

							if (0 == tile4.thread_rank())
							{
								sharedArraySquaredCandidates[warpIdInBlock * 8 + tile4.meta_group_rank()] = 
									preComputedSquaredCoordinates[candidateId * (COMPUTE_DIM / 4) + (n / 4)];
							}
						} else {
							sharedArrayTmp8x4[sharedArray8x4Offset + tile4.meta_group_rank() * 4 + tile4.thread_rank()] = 0.0;
							if (0 == tile4.thread_rank())
							{
								sharedArraySquaredCandidates[warpIdInBlock * 8 + tile4.meta_group_rank()] = 0.0;
							}
						}

						#if OUTPUT_DEBUG_TENSOR
							if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
							{
								printf("\nCandidate points: \n");
								for (unsigned int i = 0; i < 8; ++i)
								{
									printf("Candidate %d: ", i);
									for (unsigned int j = 0; j < 4; ++j)
									{
										printf("%f, ", sharedArrayTmp8x4[sharedArray8x4Offset + i * 4 + j]);
									}
									printf("\n");
								}

								printf("\nSquared candidate points (C0, C1, C2, C3, C4, C5, C6, C7): \n");
								for (unsigned int i = 0; i < 8; ++i)
								{
									printf("%f, ", sharedArraySquaredCandidates[warpIdInBlock * 8 + i]);
								}
								printf("\n");
							}
						#endif

						// Matrix fragments: Res, Q, C, C^2
						wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> matrixQ;
						wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> matrixC;
						wmma::fragment<wmma::accumulator, 8, 8, 4, double> matrixC2;
						wmma::fragment<wmma::accumulator, 8, 8, 4, double> matrixQCC2;

						// Load matrix fragments
						wmma::load_matrix_sync(matrixQ, sharedArrayQueryPoints + sharedArrayQueryOffset + n, COMPUTE_DIM);
						wmma::load_matrix_sync(matrixC, sharedArrayTmp8x4 + sharedArray8x4Offset, 4);
						wmma::load_matrix_sync(matrixC2, sharedArraySquaredCandidates + warpIdInBlock * 8, 0, wmma::mem_row_major);
						wmma::fill_fragment(matrixQCC2, 0.0);
						
						// Scale the query points matrix Q by -2.0
						for (unsigned int i = 0; i < matrixQ.num_elements; ++i)
						{
							matrixQ.x[i] *= (-2.0);
						}

						// Compute -2QC + C^2
						wmma::mma_sync(matrixQCC2, matrixQ, matrixC, matrixC2);
						wmma::store_matrix_sync(sharedArrayResultTmp + sharedArrayOffset, matrixQCC2, 8, wmma::mem_row_major);

						#if OUTPUT_DEBUG_TENSOR
							if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
							{
								printf("\n-2QC + C^2: \n");
								for (unsigned int i = 0; i < 8; ++i)
								{
									printf("Query %d: ", i);
									for (unsigned int j = 0; j < 8; ++j)
									{
										printf("%f, ", sharedArrayResultTmp[sharedArrayOffset + i * 8 + j]);
									}
									printf("\n");
								}
							}
						#endif

						// Accumulate the previous result with Q^2 for the Euclidean distance of the current dimension
						// Accumulate the previous results with the Euclidean distance from previous dimensions
						// Upper half of the result matrix
						sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()] =
							sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]
							+ sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]
							+ sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 + tile8.meta_group_rank()];
						// Lower half of the result matrix
						sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32] =
								sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32]
								+ sharedArrayResultTmp[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank() + 32]
								+ sharedArraySquaredQueries[sharedArraySquaredOffset + (n / 4) * 8 + tile8.meta_group_rank() + 4];

						#if OUTPUT_DEBUG_TENSOR
							if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
							{
								printf("\nResult: \n");
								for (unsigned int i = 0; i < 8; ++i)
								{
									printf("Query %d: ", i);
									for (unsigned int j = 0; j < 8; ++j)
									{
										printf("%f, ", sharedArrayResult[sharedArrayOffset + i * 8 + j]);
									}
									printf("\n");
								}
							}
						#endif

						#if SHORT_CIRCUIT
							unsigned int shortCircuit = 0;
							unsigned int shortCircuitTotal = nbQueriesBatch * nbCandidatesCurrent;

							// Upper half of the result matrix
							// Make sure that the threads don't access results that don't exist
							if (tile8.meta_group_rank() < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
							{
								double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
								if ((*epsilon) < sqrt(tmpDistance))
								{
									shortCircuit++;
								}
							}
							// Lower half of the result matrix
							// Make sure that the threads don't access results that don't exist
							if ((tile8.meta_group_rank() + 4) < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
							{
								double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + 32 + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
								if ((*epsilon) < sqrt(tmpDistance))
								{
									shortCircuit++;
								}
							}

							// Reduction of the number of query/candidate points that should short-circuit their computation
							unsigned int shortCircuitReduce = __reduce_add_sync(0xffffffff, shortCircuit);
							if (shortCircuitReduce >= shortCircuitTotal)
							{
								#if OUTPUT_DEBUG_TENSOR
									if (1 == print && batchToPrint == batchIndex && 0 == warp.thread_rank())
									{
										printf("\n\nSHORT CIRCUIT\n\n");
									}
								#endif
								n = COMPUTE_DIM;
							}
						#endif
					} // for COMPUTE_DIM

					// Upper half of the result matrix
					// Make sure that the threads don't access results that don't exist
					if (tile8.meta_group_rank() < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
					{
						double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
						if (sqrt(tmpDistance) <= (*epsilon))
						{
							unsigned int tmpIdx = atomicAdd(cnt, int(1));
							pointIDKey[tmpIdx] = originPointIndex[tensorBatches[batchIndex] + tile8.meta_group_rank()];
							pointInDistVal[tmpIdx] = gridLookupArr[k + tile8.thread_rank()];
						}
					}

					// Lower half of the result matrix
					// Make sure that the threads don't access results that don't exist
					if ((tile8.meta_group_rank() + 4) < nbQueriesBatch && tile8.thread_rank() < nbCandidatesCurrent)
					{
						double tmpDistance = fabs(sharedArrayResult[sharedArrayOffset + 32 + tile8.meta_group_rank() * 8 + tile8.thread_rank()]);
						if (sqrt(tmpDistance) <= (*epsilon))
						{
							unsigned int tmpIdx = atomicAdd(cnt, int(1));
							pointIDKey[tmpIdx] = originPointIndex[tensorBatches[batchIndex] + tile8.meta_group_rank() + 4];
							pointInDistVal[tmpIdx] = gridLookupArr[k + tile8.thread_rank()];
						}
					}

					print = 0;
				} // for candidates
			} // if non-empty cells
		} // for neighboring cells
}
#endif