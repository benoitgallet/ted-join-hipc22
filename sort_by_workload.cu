#include <stdio.h>
#include <algorithm>
#include <iostream>

#include "omp.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>

#include "params.h"
#include "structs.h"
#include "utils.h"
#include "sort_by_workload.h"


__device__ uint64_t getLinearID_nDimensionsGPUSortByWL(
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


__global__ void sortByWorkLoadGlobal(
	INPUT_DATA_TYPE* database,
	ACCUM_TYPE* epsilon,
	struct grid* index,
	unsigned int* indexLookupArr,
	struct gridCellLookup* gridCellLookupArr,
	INPUT_DATA_TYPE* minArr,
	unsigned int* nCells,
	unsigned int* nNonEmptyCells,
	schedulingCell* sortedCells)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (*nNonEmptyCells <= tid)
	{
		return;
	}

	unsigned int cell = gridCellLookupArr[tid].idx;
	unsigned int nbNeighborPoints = 0;
	unsigned int tmpId = indexLookupArr[ index[cell].indexmin ];

	INPUT_DATA_TYPE point[INPUT_DATA_DIM];
	for (int i = 0; i < INPUT_DATA_DIM; ++i)
	{
		point[i] = database[tmpId * COMPUTE_DIM + i];
	}

	unsigned int nDCellIDs[INDEXED_DIM];

	unsigned int nDMinCellIDs[INDEXED_DIM];
	unsigned int nDMaxCellIDs[INDEXED_DIM];

	for (int n = 0; n < INDEXED_DIM; ++n)
	{
        #if ACCUM_PREC == 16
		nDCellIDs[n] = (point[n] - minArr[n]) / (INPUT_DATA_TYPE)(*epsilon);
        #else
        nDCellIDs[n] = (point[n] - minArr[n]) / (*epsilon);
        #endif
		nDMinCellIDs[n] = max(0, nDCellIDs[n] - 1);;
		nDMaxCellIDs[n] = min(nCells[n] - 1, nDCellIDs[n] + 1);
	}

	unsigned int indexes[INDEXED_DIM];
	unsigned int loopRng[INDEXED_DIM];

	// for (loopRng[0] = rangeFilteredCellIdsMin[0]; loopRng[0] <= rangeFilteredCellIdsMax[0]; loopRng[0]++)
	// 	for (loopRng[1] = rangeFilteredCellIdsMin[1]; loopRng[1] <= rangeFilteredCellIdsMax[1]; loopRng[1]++)
	for (loopRng[0] = nDMinCellIDs[0]; loopRng[0] <= nDMaxCellIDs[0]; loopRng[0]++)
		for (loopRng[1] = nDMinCellIDs[1]; loopRng[1] <= nDMaxCellIDs[1]; loopRng[1]++)
		#include "kernelloops.h"
		{
			for (int x = 0; x < INDEXED_DIM; x++) 
            {
				indexes[x] = loopRng[x];
			}

			uint64_t cellID = getLinearID_nDimensionsGPUSortByWL(indexes, nCells, INDEXED_DIM);
			struct gridCellLookup tmp;
			tmp.gridLinearID = cellID;
			if (thrust::binary_search(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp)))
			{
				struct gridCellLookup * resultBinSearch = thrust::lower_bound(thrust::seq, gridCellLookupArr, gridCellLookupArr + (*nNonEmptyCells), gridCellLookup(tmp));
				unsigned int GridIndex = resultBinSearch->idx;
				nbNeighborPoints += index[GridIndex].indexmax - index[GridIndex].indexmin + 1;
			}
		}

	sortedCells[tid].nbPoints = nbNeighborPoints;
	sortedCells[tid].cellId = cell;

}


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
    unsigned int** dev_originPointIndex)
{
    cudaError_t errCode;

    cudaEvent_t startKernel, endKernel;
    cudaEventCreate(&startKernel);
    cudaEventCreate(&endKernel);

	// struct schedulingCell * sortedDatabaseTmp = new schedulingCell[sizeof(struct schedulingCell) * (*nNonEmptyCells)];
    (*sortedDatabaseTmp) = new schedulingCell[sizeof(struct schedulingCell) * (*nNonEmptyCells)];

	struct schedulingCell* dev_sortedDatabaseTmp;
    cudaErrCheck(cudaMalloc((void**)&dev_sortedDatabaseTmp, sizeof(struct schedulingCell) * (*nNonEmptyCells)));

    // Beginning of the sorting section
    int nbBlock = ((*nNonEmptyCells) / BLOCKSIZE) + 1;

    cudaEventRecord(startKernel);
    sortByWorkLoadGlobal<<<nbBlock, BLOCKSIZE>>>((*dev_database), (*dev_epsilon), (*dev_index), (*dev_indexLookupArr),
            (*dev_gridCellLookupArr), (*dev_minArr), (*dev_nCells), (*dev_nNonEmptyCells), dev_sortedDatabaseTmp);
    cudaEventRecord(endKernel);

    cudaDeviceSynchronize();

    cudaErrCheck(cudaMemcpy((*sortedDatabaseTmp), dev_sortedDatabaseTmp, sizeof(struct schedulingCell) * (*nNonEmptyCells), cudaMemcpyDeviceToHost));

    cudaEventSynchronize(endKernel);
    float timeKernel = 0;
    cudaEventElapsedTime(&timeKernel, startKernel, endKernel);
    fprintf(stdout, "[Sort | Time] ~ Kernel time to determine the workload: %f\n", timeKernel);

    double tBeginSort = omp_get_wtime();
    std::sort((*sortedDatabaseTmp), (*sortedDatabaseTmp) + (*nNonEmptyCells),
            [](const schedulingCell& a, const schedulingCell& b){ return a.nbPoints > b.nbPoints; });
    double tEndSort = omp_get_wtime();
    fprintf(stdout, "[Sort | Time] ~ Time to sort by workload: %f\n", tEndSort - tBeginSort);

    unsigned int maxNeighbor = (*sortedDatabaseTmp)[0].nbPoints;
    unsigned int minNeighbor = (*sortedDatabaseTmp)[(*nNonEmptyCells) - 1].nbPoints;
    // cout << "max = " << maxNeighbor << '\n';
    // cout << "min = " << minNeighbor << '\n';
    boost::multiprecision::uint128_t accNeighbor = 0;

    (*originPointIndex) = new unsigned int [(*nbQueryPoints)];
    
    #if REORDER_DATASET
        (*reorderedDataset) = new INPUT_DATA_TYPE[(*nbQueryPoints) + ADDITIONAL_POINTS];
    #endif

    unsigned int prec = 0;
    uint64_t runningPartition = 0;
    for (int i = 0; i < (*nNonEmptyCells); ++i)
    {
        int cellId = (*sortedDatabaseTmp)[i].cellId;
        int nbNeighbor = index[cellId].indexmax - index[cellId].indexmin + 1;
        accNeighbor += (((unsigned long long)nbNeighbor) * (unsigned long long)(*sortedDatabaseTmp)[i].nbPoints);

        for (int j = 0; j < nbNeighbor; ++j)
        {
            int tmpId = indexLookupArr[ index[cellId].indexmin + j ];
            (*originPointIndex)[prec + j] = tmpId;
            runningPartition += (*sortedDatabaseTmp)[i].nbPoints;
        }
        prec += nbNeighbor;
    }

    cudaErrCheck(cudaMalloc((void**)dev_originPointIndex, (*nbQueryPoints) * sizeof(unsigned int)));

    cudaErrCheck(cudaMemcpy((*dev_originPointIndex), (*originPointIndex), (*nbQueryPoints) * sizeof(unsigned int), cudaMemcpyHostToDevice));

    (*totalCandidates) = accNeighbor;

    unsigned int decileMark = (*nNonEmptyCells) / 10;
	std::cout << "[Sort | Result] ~ Total number of candidate points to refine: " << accNeighbor << '\n';
    fprintf(stdout, "[Sort | Result] ~ Number of candidates: min = %d, median = %d, max = %d\n",
            minNeighbor, (*sortedDatabaseTmp)[(*nNonEmptyCells) / 2].nbPoints, maxNeighbor);
    fprintf(stdout, "[Sort] ~ Deciles number of candidates:\n");
    for (int i = 1; i < 10; ++i)
    {
        fprintf(stdout, "   [Sort] ~ %d = %d\n", i * 10, (*sortedDatabaseTmp)[decileMark * i].nbPoints);
    }

    cudaFree(dev_sortedDatabaseTmp);
}
