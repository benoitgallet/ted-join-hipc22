#include <vector>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include "omp.h"

#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>

#include "params.h"
#include "structs.h"
#include "utils.h"
#include "index.h"


__device__ uint64_t getLinearID_nDimensionsGPUIndex(
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


__global__ void kernelIndexComputeNonEmptyCells(
	INPUT_DATA_TYPE* database,
	unsigned int* nbQueryPoints,
	ACCUM_TYPE* epsilon,
	INPUT_DATA_TYPE* minArr,
	unsigned int* nCells,
	uint64_t* pointCellArr,
	unsigned int* databaseVal,
	bool enumerate)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (*nbQueryPoints <= tid)
	{
		return;
	}

	unsigned int pointID = tid * COMPUTE_DIM;

	unsigned int tmpNDCellIdx[INDEXED_DIM];
	for (int i = 0; i < INDEXED_DIM; ++i)
	{
        #if ACCUM_PREC == 16
		tmpNDCellIdx[i] = ((database[pointID + i] - minArr[i]) / (INPUT_DATA_TYPE)(*epsilon));
        #else
        tmpNDCellIdx[i] = ((database[pointID + i] - minArr[i]) / (*epsilon));
        #endif
	}

	uint64_t linearID = getLinearID_nDimensionsGPUIndex(tmpNDCellIdx, nCells, INDEXED_DIM);
	pointCellArr[tid] = linearID;

	if (enumerate)
	{
		databaseVal[tid] = tid;
	}
}



//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\



void reorderDimensionsByVariance(
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector)
{
	INPUT_DATA_TYPE sums[INPUT_DATA_DIM];
	INPUT_DATA_TYPE average[INPUT_DATA_DIM];
	struct dim_reorder_sort dim_variance[INPUT_DATA_DIM];

	for (int i = 0; i < INPUT_DATA_DIM; ++i)
    {
		sums[i] = 0;
		average[i] = 0;
	}

	INPUT_DATA_TYPE greatest_variance = 0;
	int greatest_variance_dim = 0;

	int sample = 100;
	INPUT_DATA_TYPE inv_sample = 1.0 / (sample * 1.0);

    #if VERBOSITY >= 2
    fprintf(stdout, "[Index] ~ Calculating variance based on the following fraction of points: %f\n", inv_sample);
    #endif

    double tvariancestart = omp_get_wtime();
	//calculate the variance in each dimension
	for (int i = 0; i < INPUT_DATA_DIM; ++i)
	{
		//first calculate the average in the dimension:
		//only use every 10th point
		for (int j = 0; j < (*inputVector).size(); j += sample)
		{
    		sums[i] += (*inputVector)[j][i];
		}

		average[i] = (sums[i]) / ((*inputVector).size() * inv_sample);

		//Next calculate the std. deviation
		sums[i] = 0; //reuse this for other sums
		for (int j = 0; j < (*inputVector).size(); j += sample)
		{
    		sums[i] += (((*inputVector)[j][i]) - average[i]) * (((*inputVector)[j][i]) - average[i]);
		}

		dim_variance[i].variance = sums[i] / ((*inputVector).size() * inv_sample);
		dim_variance[i].dim = i;

		if (greatest_variance<dim_variance[i].variance)
		{
			greatest_variance = dim_variance[i].variance;
			greatest_variance_dim = i;
		}
	}

	// std::sort(dim_variance, dim_variance + GPUNUMDIM, compareByDimVariance);
    std::sort(dim_variance, dim_variance + INPUT_DATA_DIM,
        [](const dim_reorder_sort &a, const dim_reorder_sort &b){ return a.variance > b.variance; });

    #if VERBOSITY >= 2
        for (int i = 0; i < INPUT_DATA_DIM; ++i)
        {
            std::cout << "[Index] ~ Reordering by: dim:" << dim_variance[i].dim << ", variance: "  << dim_variance[i].variance << '\n';
        }
    #endif

    #if VERBOSITY >= 2
        std::cout << "[Index] ~ Dimension with greatest variance: " << greatest_variance_dim << '\n';
    #endif

	std::vector< std::vector<INPUT_DATA_TYPE> > tmp_database;

	tmp_database = (*inputVector);

	#pragma omp parallel for num_threads(5) shared(inputVector, tmp_database)
	for (int j = 0; j < INPUT_DATA_DIM; ++j)
    {

		int originDim = dim_variance[j].dim;
		for (int i = 0; i < (*inputVector).size(); ++i)
		{
			(*inputVector)[i][j] = tmp_database[i][originDim];
		}
	}
}



void generateNDGridDimensions(
    std::vector< std::vector <INPUT_DATA_TYPE> >* inputVector,
    ACCUM_TYPE epsilon,
    INPUT_DATA_TYPE* minArr,
    INPUT_DATA_TYPE* maxArr,
    unsigned int* nCells,
    uint64_t* totalCells)
{
    fprintf(stdout, "[Index] ~ Number of dimensions data: %d, Number of dimensions indexed: %d\n", INPUT_DATA_DIM, INDEXED_DIM);

    //make the min/max values for each grid dimension the first data element
    for (int j = 0; j < INDEXED_DIM; ++j)
    {
        minArr[j] = (*inputVector)[0][j];
        maxArr[j] = (*inputVector)[0][j];
    }



    for (int i = 1; i < inputVector->size(); ++i)
    {
        for (int j = 0; j < INDEXED_DIM; ++j)
        {
            if ((*inputVector)[i][j] < minArr[j])
            {
                minArr[j] = (*inputVector)[i][j];
            }
            if((*inputVector)[i][j] > maxArr[j])
            {
                maxArr[j] = (*inputVector)[i][j];
            }
        }
    }

    #if VERBOSITY >= 2
    for (int j = 0; j < INDEXED_DIM; ++j)
    {
        printf("Data Dim: %d, min/max: %f, %f\n", j, minArr[j], maxArr[j]);
    }
    #endif

    //add buffer around each dim so no weirdness later with putting data into cells
    for (int j = 0; j < INDEXED_DIM; ++j)
    {
        #if ACCUM_PREC == 16
            minArr[j] -= (INPUT_DATA_TYPE)epsilon;
            maxArr[j] += (INPUT_DATA_TYPE)epsilon;
        #else
            minArr[j] -= epsilon;
            maxArr[j] += epsilon;
        #endif
    }

    #if VERBOSITY >= 2
    for (int j = 0; j < INDEXED_DIM; ++j)
    {
        printf("Appended by epsilon Dim: %d, min/max: %f, %f\n", j, minArr[j], maxArr[j]);
    }
    #endif

    //calculate the number of cells:
    for (int j = 0; j < INDEXED_DIM; ++j)
    {
        #if ACCUM_PREC == 16
        nCells[j] = ceil((maxArr[j] - minArr[j]) / (INPUT_DATA_TYPE)epsilon);
        #else
        nCells[j] = ceil((maxArr[j] - minArr[j]) / epsilon);
        #endif

        #if VERBOSITY >= 2
        printf("Number of cells dim: %d: %d\n", j, nCells[j]);
        #endif
    }

    //calc total cells: num cells in each dim multiplied
    uint64_t tmpTotalCells = nCells[0];
    for (int j = 1; j < INDEXED_DIM; ++j)
    {
        tmpTotalCells *= nCells[j];
    }

    *totalCells = tmpTotalCells;
}


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
    unsigned int** dev_nCells)
{
    cudaError_t errCode;

    cudaErrCheck(cudaMalloc((void**)dev_database, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * ((*nbQueryPoints) + ADDITIONAL_POINTS)));

    cudaErrCheck(cudaMalloc((void**)dev_epsilon, sizeof(ACCUM_TYPE)));

    cudaErrCheck(cudaMalloc((void**)dev_minArr, sizeof(INPUT_DATA_TYPE) * (INDEXED_DIM)));

    cudaErrCheck(cudaMalloc((void**)dev_indexLookupArr, sizeof(unsigned int) * (*nbQueryPoints)));

    cudaErrCheck(cudaMalloc((void**)dev_nNonEmptyCells, sizeof(unsigned int)));

    cudaErrCheck(cudaMalloc((void**)dev_nCells, sizeof(unsigned int) * (INDEXED_DIM)));

    uint64_t* dev_pointCellArr;
    cudaErrCheck(cudaMalloc((void**)&dev_pointCellArr, sizeof(uint64_t) * (*nbQueryPoints)));

    unsigned int* dev_databaseVal;
	cudaErrCheck(cudaMalloc((void**)&dev_databaseVal, sizeof(unsigned int) * (*nbQueryPoints)));

    unsigned int* N = new unsigned int;
	unsigned int* dev_N;
	cudaErrCheck(cudaMalloc((void**)&dev_N, sizeof(unsigned int) * GPUSTREAMS));



    ////////////////////////////////////////////////////////////////////////////



    cudaErrCheck(cudaMemcpy((*dev_database), database, sizeof(INPUT_DATA_TYPE) * (COMPUTE_DIM) * ((*nbQueryPoints) + ADDITIONAL_POINTS), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy((*dev_epsilon), epsilon, sizeof(ACCUM_TYPE), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy((*dev_minArr), minArr, sizeof(INPUT_DATA_TYPE) * (INDEXED_DIM), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy((*dev_nCells), nCells, sizeof(unsigned int) * (INDEXED_DIM), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy(dev_N, nbQueryPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));



    ////////////////////////////////////////////////////////////////////////////



    const int TOTALBLOCKS = ceil((1.0 * (*nbQueryPoints)) / (1.0 * BLOCKSIZE));

	kernelIndexComputeNonEmptyCells<<<TOTALBLOCKS, BLOCKSIZE>>>((*dev_database), dev_N, (*dev_epsilon), (*dev_minArr),
            (*dev_nCells), dev_pointCellArr, nullptr, false);
    cudaDeviceSynchronize();

    thrust::device_ptr<uint64_t> dev_pointCellArr_ptr(dev_pointCellArr);
	thrust::device_ptr<uint64_t> dev_new_end;

	try
    {
		//first sort
		thrust::sort(thrust::device, dev_pointCellArr_ptr, dev_pointCellArr_ptr + (*nbQueryPoints)); //, thrust::greater<uint64_t>()
		//then unique
		dev_new_end = thrust::unique(thrust::device, dev_pointCellArr_ptr, dev_pointCellArr_ptr + (*nbQueryPoints));
	} catch (std::bad_alloc &e) {
        fprintf(stderr, "[Index] ~ Ran out of memory while sorting\n");
	    exit(1);
    }

    uint64_t* new_end = thrust::raw_pointer_cast(dev_new_end);
    uint64_t numNonEmptyCells = std::distance(dev_pointCellArr_ptr, dev_new_end);
    fprintf(stdout, "[Index] ~ Number of full cells (non-empty): %lu\n", numNonEmptyCells);
    *nNonEmptyCells = numNonEmptyCells;

    (*gridCellLookupArr) = new struct gridCellLookup[numNonEmptyCells];
    uint64_t * pointCellArrTmp = new uint64_t[numNonEmptyCells];
    cudaErrCheck(cudaMemcpy(pointCellArrTmp, dev_pointCellArr, sizeof(uint64_t) * numNonEmptyCells, cudaMemcpyDeviceToHost));

	for (uint64_t i = 0; i < numNonEmptyCells; ++i)
	{
		(*gridCellLookupArr)[i].idx = i;
		(*gridCellLookupArr)[i].gridLinearID = pointCellArrTmp[i];
	}

    kernelIndexComputeNonEmptyCells<<<TOTALBLOCKS, BLOCKSIZE>>>((*dev_database), dev_N, (*dev_epsilon), (*dev_minArr),
            (*dev_nCells), dev_pointCellArr, dev_databaseVal, true);

    try
	{
    	thrust::sort_by_key(thrust::device, dev_pointCellArr, dev_pointCellArr + (*nbQueryPoints), dev_databaseVal);
	} catch (std::bad_alloc &e) {
        fprintf(stderr, "[Index] ~ Ran out of memory while sorting key/value pairs\n");
	    exit(1);
	}

    uint64_t * cellKey = new uint64_t[(*nbQueryPoints)];
    cudaErrCheck(cudaMemcpy(cellKey, dev_pointCellArr, sizeof(uint64_t) * (*nbQueryPoints), cudaMemcpyDeviceToHost));

    cudaErrCheck(cudaMemcpy(indexLookupArr, dev_databaseVal, sizeof(unsigned int) * (*nbQueryPoints), cudaMemcpyDeviceToHost));

    (*index) = new grid[numNonEmptyCells];
    (*index)[0].indexmin = 0;
	uint64_t cnt=0;
	for (uint64_t i = 1; i < (*nbQueryPoints); ++i)
    {
		if (cellKey[i - 1] != cellKey[i])
		{
			//grid index
			cnt++;
			(*index)[cnt].indexmin = i;
			(*index)[cnt - 1].indexmax = i - 1;
		}
	}
	(*index)[numNonEmptyCells - 1].indexmax = (*nbQueryPoints) - 1;

    #if VERBOSITY >= 1
    fprintf(stdout, "[Index] ~ Full cells: %d (%f, fraction full)\n", (unsigned int)numNonEmptyCells, numNonEmptyCells * 1.0 / double(totalCells));
	fprintf(stdout, "[Index] ~ Empty cells: %ld (%f, fraction empty)\n", totalCells - (unsigned int)numNonEmptyCells, (totalCells - numNonEmptyCells * 1.0) / double(totalCells));
	fprintf(stdout, "[Index] ~ Size of index that would be sent to GPU (GiB) -- (if full index sent), excluding the data lookup arr: %f\n",
        (double)sizeof(struct grid) * (totalCells) / (1024.0 * 1024.0 * 1024.0));
	fprintf(stdout, "[Index] ~ Size of compressed index to be sent to GPU (GiB), excluding the data and grid lookup arr: %f\n",
        (double)sizeof(struct grid) * (numNonEmptyCells * 1.0) / (1024.0 * 1024.0 * 1024.0));
	fprintf(stdout, "[Index] ~ When copying from entire index to compressed index: number of non-empty cells: %lu\n", numNonEmptyCells);
    #endif

    ////////////////////////////////////////////////////////////////////////////

    cudaErrCheck(cudaMalloc((void**)dev_index, sizeof(struct grid) * (*nNonEmptyCells)));

    cudaErrCheck(cudaMalloc((void**)dev_gridCellLookupArr, sizeof(struct gridCellLookup) * (*nNonEmptyCells)));

    ////////////////////////////////////////////////////////////////////////////

    cudaErrCheck(cudaMemcpy((*dev_nNonEmptyCells), nNonEmptyCells, sizeof(unsigned int), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy((*dev_index), (*index), sizeof(struct grid) * numNonEmptyCells, cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy((*dev_indexLookupArr), indexLookupArr, sizeof(unsigned int) * (*nbQueryPoints), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMemcpy((*dev_gridCellLookupArr), (*gridCellLookupArr), sizeof(struct gridCellLookup) * numNonEmptyCells, cudaMemcpyHostToDevice));

    ////////////////////////////////////////////////////////////////////////////

    delete N;
    delete[] pointCellArrTmp;
    cudaFree(dev_pointCellArr);
    cudaFree(dev_databaseVal);
    cudaFree(dev_N);
}
