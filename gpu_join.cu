#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <utility>

#include "omp.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

//thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h> // for streams for thrust (added with Thrust v1.8)

#include "params.h"
#include "structs.h"
#include "utils.h"
#include "gpu_join.h"
#include "kernel_join.h"

using std::cout;
using std::endl;


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
    std::vector<struct batch>* batches)
{
    double sampleRate = 0.10;
    int offsetRate = 1.0 / sampleRate;
    fprintf(stdout, "[GPU Est] ~ Sample rate: %f, offset: %d\n", sampleRate, offsetRate);

    unsigned int* N_batchEst = new unsigned int;
    (*N_batchEst) = (*nbQueryPoints) * sampleRate;

    unsigned int* cnt_batchEst = new unsigned int;
    (*cnt_batchEst) = 0;

    unsigned int* estimatedResult = new unsigned int[(*N_batchEst)];
    unsigned int* estimateCandidates = new unsigned int[(*N_batchEst)];

    unsigned int* dev_N_batchEst;
    cudaErrCheck(cudaMalloc((void**)&dev_N_batchEst, sizeof(unsigned int)));
    cudaErrCheck(cudaMemcpy(dev_N_batchEst, N_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int* dev_cnt_batchEst;
    cudaErrCheck(cudaMalloc((void**)&dev_cnt_batchEst, sizeof(unsigned int)));
    cudaErrCheck(cudaMemcpy(dev_cnt_batchEst, cnt_batchEst, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int* dev_sampleOffset;
    cudaErrCheck(cudaMalloc((void**)&dev_sampleOffset, sizeof(unsigned int)));
    cudaErrCheck(cudaMemcpy(dev_sampleOffset, &offsetRate, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int* dev_estimatedResult;
    cudaErrCheck(cudaMalloc((void**)&dev_estimatedResult, (*N_batchEst) * sizeof(unsigned int)));

    unsigned int* dev_nbCandidatesEst;
    cudaErrCheck(cudaMalloc((void**)&dev_nbCandidatesEst, (*N_batchEst) * sizeof(unsigned int)));

    unsigned int nbBlock = ceil((1.0 * (*N_batchEst)) / (1.0 * BLOCKSIZE));

    batchEstimatorKernel<<<nbBlock, BLOCKSIZE>>>(dev_N_batchEst, dev_sampleOffset,
        dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_gridLookupArr,
        dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_cnt_batchEst, dev_nNonEmptyCells,
        dev_estimatedResult, dev_nbCandidatesEst);

    cudaErrCheck(cudaMemcpy(cnt_batchEst, dev_cnt_batchEst, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(estimatedResult, dev_estimatedResult, (*N_batchEst) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(estimateCandidates, dev_nbCandidatesEst, (*N_batchEst) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    fprintf(stdout, "[GPU Est] ~ Result set size for estimating the number of batches (sampled): %d\n", (*cnt_batchEst));

    unsigned int GPUBufferSize = GPU_BUFFER_SIZE;

    boost::multiprecision::uint128_t fullEst = 0;
    boost::multiprecision::uint128_t fullEstCandidates = 0;
    unsigned int* estimatedResultFull = new unsigned int[(*nbQueryPoints)];
    unsigned int* estimatedCandidatesFull = new unsigned int[(*nbQueryPoints)];
    unsigned int nbUnestimatedSequences = (*nbQueryPoints) / offsetRate;

    unsigned int estAfter;

    for (unsigned int i = 0; i < nbUnestimatedSequences - 1; ++i)
    {
        unsigned int nbEstBefore = estimatedResult[i];
        unsigned int nbEstAfter = estimatedResult[i + 1];
        unsigned int maxEst = (nbEstBefore < nbEstAfter) ? nbEstAfter : nbEstBefore;

        unsigned int nbCandidatesBefore = estimateCandidates[i];
        unsigned int nbCandidatesAfter = estimateCandidates[i + 1];
        unsigned int maxCandidates = (nbCandidatesBefore < nbCandidatesAfter) ? nbCandidatesAfter : nbCandidatesBefore;

        unsigned int estBefore = i * offsetRate;
        estAfter = (i + 1) * offsetRate;
        estimatedResultFull[estBefore] = nbEstBefore;
        fullEst += nbEstBefore;
        estimatedCandidatesFull[estBefore] = nbCandidatesBefore;
        fullEstCandidates += nbCandidatesBefore;

        for (int j = estBefore + 1; j < estAfter; ++j)
        {
            estimatedResultFull[j] = maxEst;
            fullEst += maxEst;
            estimatedCandidatesFull[j] = maxCandidates;
            fullEstCandidates += maxCandidates;
        }
    }

    if (estAfter < (*nbQueryPoints))
    {
        // cout << "   COUCOU Y A UN SOUCIS   " << endl;
        for (unsigned int i = estAfter; i < (*nbQueryPoints); ++i)
        {
            estimatedResultFull[i] = estimatedResultFull[estAfter - 1];
            estimatedCandidatesFull[i] = estimatedCandidatesFull[estAfter - 1];
        }
    }

    std::cout << "[GPU Est] ~ Estimated total result set size: " << fullEst << '\n';
    std::cout << "[GPU Est] ~ Estimated number of candidate points: " << fullEstCandidates << '\n';
    // fprintf(stdout, "[GPU Est] ~ Estimated total result set size: %lu\n", fullEst);
    // fprintf(stdout, "[GPU Est] ~ Estimated number of candidate points: %lu\n", fullEstCandidates);

    if (fullEst < (GPUBufferSize * GPUSTREAMS))
    {
        GPUBufferSize = min((int)(ceil((1.0 * fullEst.convert_to<int>()) / (1.0 * GPUSTREAMS)) * 1.10), GPU_BUFFER_SIZE);
        // GPUBufferSize = ceil((1.0 * fullEst.convert_to<int>()) / (1.0 * GPUSTREAMS));
        // GPUBufferSize = (fullEst.convert_to<int>() / GPUSTREAMS) + 1;
        fprintf(stdout, "[GPU Est] ~ Result set size too small, reducing GPUBufferSize to %d\n", GPUBufferSize);
    }

    // GPUBufferSize = 100000000;

    unsigned int batchBegin = 0;
    uint64_t runningEst = 0;
    boost::multiprecision::uint128_t runningCandidates = 0;
    // Keeping a 5% margin to avoid a potential buffer overflow due to a wrong estimation of the result set size
    unsigned int reserveBuffer = GPUBufferSize * 0.05;

    for (unsigned int i = 0; i < (*nbQueryPoints); ++i)
    {
        runningEst += estimatedResultFull[i];
        runningCandidates += estimatedCandidatesFull[i];

        if ((GPUBufferSize - reserveBuffer) <= runningEst)
        {
            struct batch newBatch = {batchBegin, i, runningCandidates, runningEst};
            // fprintf(stdout, "[Batch Est] ~ New batch: begin = %u, end = %u, candidates = %llu, neighbors = %llu\n",
            //     newBatch.begin, newBatch.end, newBatch.nbCandidates, newBatch.nbNeighborsEst);
            batches->push_back(newBatch);
            batchBegin = i;
            runningEst = 0;
            runningCandidates = 0;
        } else {
            // The last batch is unlikely to pass the above condition, so we force its creation when the end of the dataset is reached
            if (i == ((*nbQueryPoints) - 1))
            {
                struct batch newBatch = {batchBegin, (*nbQueryPoints), runningCandidates, runningEst};
                batches->push_back(newBatch);
                runningEst = 0;
                runningCandidates = 0;
            }
        }
    }

    fprintf(stdout, "[GPU Est] ~ Total number of batches: %lu\n", batches->size());

    (*retNumBatches) = batches->size();
    (*retGPUBufferSize) = GPUBufferSize;

    cudaFree(dev_N_batchEst);
    cudaFree(dev_cnt_batchEst);
    cudaFree(dev_sampleOffset);
    cudaFree(dev_estimatedResult);
    cudaFree(dev_nbCandidatesEst);

    delete[] estimatedResult;
    delete[] estimateCandidates;
    delete[] estimatedResultFull;
    delete[] estimatedCandidatesFull;

    return fullEst;
}



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
    struct schedulingCell* sortedCells)
{
    double tFunctionStart = omp_get_wtime();

    cudaError_t errCode;

    #if INPUT_DATA_PREC != COMPUTE_PREC
        half* dev_datasetAlt;
        cudaErrCheck(cudaMalloc((void**)&dev_datasetAlt, sizeof(half) * ((*nbQueryPoints) + ADDITIONAL_POINTS) * COMPUTE_DIM));
        unsigned int nbBlock = ceil((1.0 * ((*nbQueryPoints) + ADDITIONAL_POINTS)) / (1.0 * BLOCKSIZE));
        convertDataset<<<nbBlock, BLOCKSIZE>>>(dev_database, dev_datasetAlt, ((*nbQueryPoints) + ADDITIONAL_POINTS));
        // cudaFree(dev_database);

        half* dev_minArrAlt;
        cudaErrCheck(cudaMalloc((void**)&dev_minArrAlt, INDEXED_DIM * sizeof(half)));
        convertMinArr<<<1, 1>>>(dev_minArr, dev_minArrAlt);
    #endif

    double* dev_preComputedSquaredCoordinates;
    if (searchMode == SM_TENSOR_MQ_HYBRID)
    {
        unsigned int nbBlocks = ceil((1.0 * ((*nbQueryPoints) + ADDITIONAL_POINTS)) / (1.0 * BLOCKSIZE));
        // Pre-compute the square of points' coordinates
        cudaErrCheck(cudaMalloc((void**)&dev_preComputedSquaredCoordinates, sizeof(double) * ((*nbQueryPoints) + ADDITIONAL_POINTS) * (COMPUTE_DIM / 4)));
        preComputedSquaredCoordinates<<<nbBlocks, BLOCKSIZE>>>(dev_database, dev_preComputedSquaredCoordinates, ((*nbQueryPoints) + ADDITIONAL_POINTS));
    }

    unsigned int* totalResultSetCnt = new unsigned int;
    *totalResultSetCnt = 0;

    unsigned int* cnt = new unsigned int[GPUSTREAMS];
    unsigned int* dev_cnt;
    cudaErrCheck(cudaMalloc((void**)&dev_cnt, sizeof(unsigned int) * GPUSTREAMS));

    unsigned int* N = new unsigned int[GPUSTREAMS];
	unsigned int* dev_N;
	cudaErrCheck(cudaMalloc((void**)&dev_N, sizeof(unsigned int) * GPUSTREAMS));

    unsigned int* batchOffset = new unsigned int[GPUSTREAMS];
	unsigned int* dev_offset;
	cudaErrCheck(cudaMalloc((void**)&dev_offset, sizeof(unsigned int) * GPUSTREAMS));

    unsigned int * batchNumber = new unsigned int[GPUSTREAMS];
	unsigned int * dev_batchNumber;
	cudaErrCheck(cudaMalloc((void**)&dev_batchNumber, sizeof(unsigned int) * GPUSTREAMS));

    boost::multiprecision::uint128_t estimatedNeighbors = 0;
	unsigned int numBatches = 0;
	unsigned int GPUBufferSize = 0;

    std::vector<struct batch> batchesVector;

	double tstartbatchest = omp_get_wtime();
    estimatedNeighbors = GPUBatchEst(nbQueryPoints, dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_gridLookupArr,
            dev_gridCellLookupArr, dev_minArr, dev_nCells, dev_nNonEmptyCells, &numBatches, &GPUBufferSize, &batchesVector);
    double tendbatchest = omp_get_wtime();

    #if INPUT_DATA_PREC != COMPUTE_PREC
        cudaFree(dev_database);
    #endif

    fprintf(stdout, "[GPU] ~ Time to estimate batches: %f\n", tendbatchest - tstartbatchest);
    cout << "[GPU] ~ In calling function: Estimated neighbors = " << estimatedNeighbors
            << ", num. batches = " << numBatches << ", GPU buffer size = " << GPUBufferSize << '\n';
    cout.flush();

    #if VERBOSITY >= 2
        fprintf(stdout, "[GPU] ~ Regular batches: \n");
        for (unsigned int i = 0; i < batchesVector.size(); ++i)
        {
            fprintf(stdout, "  Batch %u: %u, %u (%u)\n", i, batchesVector[i].begin, batchesVector[i].end,
                    batchesVector[i].end - batchesVector[i].begin);
        }
    #endif

    #if NB_QUERY_POINTS_TENSOR > 1
        std::vector<unsigned int> workQueueTensorBatches;
        std::vector<std::pair<unsigned int, unsigned int>> tensorBatchesIndices;
        unsigned int runningTensorBatchBegin = 0;
        unsigned int runningTensorBatchEnd = 0;
        unsigned int pointIndex = 0;
        unsigned int cellId = 0;
        unsigned int cellIndex = 0;
        unsigned int batchIndex = 0;
        unsigned int runningPointsCell = 0;
        unsigned int runningPointsBatch = 0;

        workQueueTensorBatches.push_back(pointIndex);

        cellId = sortedCells[cellIndex].cellId;
        runningPointsCell = grid[cellId].indexmax - grid[cellId].indexmin + 1;
        runningPointsBatch = batchesVector[batchIndex].end - batchesVector[batchIndex].begin;

        while (pointIndex < (*nbQueryPoints))
        {
            if ((pointIndex + NB_QUERY_POINTS_TENSOR) < runningPointsCell && (pointIndex + NB_QUERY_POINTS_TENSOR) < runningPointsBatch)
            {
                // The tensor batch is within the current regular batch and cell
                pointIndex += NB_QUERY_POINTS_TENSOR;
                workQueueTensorBatches.push_back(pointIndex);
                runningTensorBatchEnd++;
            } else {
                if ((pointIndex + NB_QUERY_POINTS_TENSOR) < runningPointsCell && (pointIndex + NB_QUERY_POINTS_TENSOR) >= runningPointsBatch)
                {
                    // The tensor batch is within the cell but exceeds the regular batch
                    pointIndex = runningPointsBatch;
                    workQueueTensorBatches.push_back(pointIndex);
                    runningTensorBatchEnd++;
                    tensorBatchesIndices.push_back(std::make_pair(runningTensorBatchBegin, runningTensorBatchEnd));
                    runningTensorBatchBegin = runningTensorBatchEnd;
                    batchIndex++;
                    runningPointsBatch += batchesVector[batchIndex].end - batchesVector[batchIndex].begin;
                } else {
                    if ((pointIndex + NB_QUERY_POINTS_TENSOR) >= runningPointsCell && (pointIndex + NB_QUERY_POINTS_TENSOR) < runningPointsBatch)
                    {
                        // The tensor batch is within the regular batch but exceeds the current cell
                        pointIndex = runningPointsCell;
                        workQueueTensorBatches.push_back(pointIndex);
                        runningTensorBatchEnd++;
                        cellIndex++;
                        cellId = sortedCells[cellIndex].cellId;
                        runningPointsCell += grid[cellId].indexmax - grid[cellId].indexmin + 1;
                    } else {
                        if ((pointIndex + NB_QUERY_POINTS_TENSOR) >= runningPointsCell && (pointIndex + NB_QUERY_POINTS_TENSOR) >= runningPointsBatch)
                        {
                            // The tensor batch exceeds both the current cell and regular batch
                            if (runningPointsCell < runningPointsBatch)
                            {
                                // The cell ends before the regular batch
                                pointIndex = runningPointsCell;
                                workQueueTensorBatches.push_back(pointIndex);
                                runningTensorBatchEnd++;
                                cellIndex++;
                                cellId = sortedCells[cellIndex].cellId;
                                runningPointsCell += grid[cellId].indexmax - grid[cellId].indexmin + 1;
                            } else {
                                if (runningPointsBatch < runningPointsCell)
                                {
                                    // The regular batch ends before the cell
                                    pointIndex = runningPointsBatch;
                                    workQueueTensorBatches.push_back(pointIndex);
                                    runningTensorBatchEnd++;
                                    tensorBatchesIndices.push_back(std::make_pair(runningTensorBatchBegin, runningTensorBatchEnd));
                                    runningTensorBatchBegin = runningTensorBatchEnd;
                                    batchIndex++;
                                    runningPointsBatch += batchesVector[batchIndex].end - batchesVector[batchIndex].begin;
                                } else {
                                    // Both the regular batch and the cell end at the same point
                                    // Unlikely, and should most often correspond to the end of the dataset
                                    pointIndex = runningPointsCell;
                                    workQueueTensorBatches.push_back(pointIndex);
                                    runningTensorBatchEnd++;
                                    tensorBatchesIndices.push_back(std::make_pair(runningTensorBatchBegin, runningTensorBatchEnd));

                                    if (pointIndex < (*nbQueryPoints))
                                    {
                                        // If there are query points left, then having the regular batch and the cell ending at the same point
                                        // is just chance, and so other batches will be created.
                                        // But if there are no more query points left, then we do not update the variables below.
                                        runningTensorBatchBegin = runningTensorBatchEnd;
                                        cellIndex++;
                                        cellId = sortedCells[cellIndex].cellId;

                                        runningPointsCell += grid[cellId].indexmax - grid[cellId].indexmin + 1;
                                        batchIndex++;
                                        runningPointsBatch += batchesVector[batchIndex].end - batchesVector[batchIndex].begin;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        unsigned int nbTensorBatches = workQueueTensorBatches.size();

        #if VERBOSITY >= 2
            fprintf(stdout, "[GPU] ~ Tensor batches ranges: (%u tensor batches)\n", nbTensorBatches);
            for (unsigned int i = 0; i < tensorBatchesIndices.size(); ++i)
            {
                fprintf(stdout, "  Tensor batch %u: %u, %u (%u)\n", i, tensorBatchesIndices[i].first, tensorBatchesIndices[i].second,
                        tensorBatchesIndices[i].second - tensorBatchesIndices[i].first);
            }
        #endif

        unsigned int* tensorBatches = new unsigned int[nbTensorBatches];
        for (unsigned int i = 0; i < nbTensorBatches; ++i)
        {
            tensorBatches[i] = workQueueTensorBatches[i];
        }
        workQueueTensorBatches.clear();
        workQueueTensorBatches.shrink_to_fit();

        unsigned int* dev_tensorBatches;
        cudaErrCheck(cudaMalloc((void**)&dev_tensorBatches, nbTensorBatches * sizeof(unsigned int)));
        cudaErrCheck(cudaMemcpy(dev_tensorBatches, tensorBatches, nbTensorBatches * sizeof(unsigned int), cudaMemcpyHostToDevice));

        unsigned int* dev_tensorBatchesSize;
        cudaErrCheck(cudaMalloc((void**)&dev_tensorBatchesSize, sizeof(unsigned int)));
        cudaErrCheck(cudaMemcpy(dev_tensorBatchesSize, &nbTensorBatches, sizeof(unsigned int), cudaMemcpyHostToDevice));
    #endif

    for (int i = 0; i < numBatches; i++)
    {
		int *ptr;
		struct neighborDataPtrs tmpStruct;
		tmpStruct.dataPtr = ptr;
		tmpStruct.sizeOfDataArr = 0;

		pointersToNeighbors->push_back(tmpStruct);
	}

    int* dev_pointIDKey[GPUSTREAMS]; //key
	int* dev_pointInDistValue[GPUSTREAMS]; //value
	for (int i = 0; i < GPUSTREAMS; i++)
	{
		cudaErrCheck(cudaMalloc((void **)&dev_pointIDKey[i], 2 * sizeof(int) * GPUBufferSize));
		cudaErrCheck(cudaMalloc((void **)&dev_pointInDistValue[i], 2 * sizeof(int) * GPUBufferSize));
	}
    fprintf(stdout, "[GPU] ~ Allocation pointIDKey and pointInDistValue on the GPU, size = %lu\n", 2 * sizeof(int) * GPUBufferSize);

    int* pointIDKey[GPUSTREAMS]; //key
	int* pointInDistValue[GPUSTREAMS]; //value

	double tStartPinnedResults = omp_get_wtime();
    #pragma omp parallel for num_threads(GPUSTREAMS)
	for (int i = 0; i < GPUSTREAMS; i++)
	{
		cudaMallocHost((void **) &pointIDKey[i], 2 * sizeof(int) * GPUBufferSize);
		cudaMallocHost((void **) &pointInDistValue[i], 2 * sizeof(int) * GPUBufferSize);
	}
	double tStopPinnedResults = omp_get_wtime();

    fprintf(stdout, "[GPU] ~ Time to allocated pinned memory for results: %f\n", tStopPinnedResults - tStartPinnedResults);

    fprintf(stdout, "[GPU] ~ Memory request for results on GPU: %f GiB\n", (double)(sizeof(int) * 2 * GPUBufferSize * GPUSTREAMS) / (1024 * 1024 * 1024));
    fprintf(stdout, "[GPU] ~ Memory requested for results in main memory: %f GiB\n", (double)(sizeof(int) * 2 * GPUBufferSize * GPUSTREAMS) / (1024 * 1024 * 1024));

    omp_set_num_threads(GPUSTREAMS);

    cudaStream_t stream[GPUSTREAMS];
	for (int i = 0; i < GPUSTREAMS; i++)
    {
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}

    unsigned int datasetSize = *nbQueryPoints;
    unsigned int batchSize = datasetSize / numBatches;
	// unsigned int batchesThatHaveOneMore = (*nbQueryPoints) - (batchSize * numBatches); //batch number 0- < this value have one more
    unsigned int batchesThatHaveOneMore = datasetSize - (batchSize * numBatches);
    fprintf(stdout, "[GPU] ~ Batches that have one more GPU thread: %d, batchSize: %d\n", batchesThatHaveOneMore, batchSize);

	uint64_t totalResultsLoop = 0;

    unsigned int* batchBegin = new unsigned int[GPUSTREAMS];
    for (int i = 0; i < GPUSTREAMS; i++)
    {
        batchBegin[i] = 0;
    }
    unsigned int* dev_batchBegin;
    cudaErrCheck(cudaMalloc((void**)&dev_batchBegin, GPUSTREAMS * sizeof(unsigned int)));

    unsigned int* dev_batchEnd;
    cudaErrCheck(cudaMalloc((void**)&dev_batchEnd, GPUSTREAMS * sizeof(unsigned int)));

    unsigned int* dev_nbQueryPoints;
    cudaErrCheck(cudaMalloc((void**)&dev_nbQueryPoints, sizeof(unsigned int)));
    cudaErrCheck(cudaMemcpy(dev_nbQueryPoints, nbQueryPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    uint64_t* nbNeighborsCuda = new uint64_t[GPUSTREAMS];
    uint64_t* nbNeighborsTensor = new uint64_t[GPUSTREAMS];
    boost::multiprecision::uint128_t* nbCandidatesCuda = new boost::multiprecision::uint128_t[GPUSTREAMS];
    boost::multiprecision::uint128_t* nbCandidatesTensor = new boost::multiprecision::uint128_t[GPUSTREAMS];
    unsigned int* nbQueriesCuda = new unsigned int[GPUSTREAMS];
    unsigned int* nbQueriesTensor = new unsigned int[GPUSTREAMS];
    unsigned int* nbKernelsCuda = new unsigned int[GPUSTREAMS];
    unsigned int* nbKernelsTensor = new unsigned int[GPUSTREAMS];

    for (int i = 0; i < GPUSTREAMS; ++i)
    {
        nbNeighborsCuda[i] = 0;
        nbNeighborsTensor[i] = 0;
        nbCandidatesCuda[i] = 0;
        nbCandidatesTensor[i] = 0;
        nbQueriesCuda[i] = 0;
        nbQueriesTensor[i] = 0;
        nbKernelsCuda[i] = 0;
        nbKernelsTensor[i] = 0;
    }

    const int tensorBlockSize = WARP_PER_BLOCK * WARP_SIZE;

    double tStartCompute = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 1) reduction(+: totalResultsLoop) num_threads(GPUSTREAMS)
    for (int i = 0; i < numBatches; ++i)
    {
        unsigned int tid = omp_get_thread_num();

        bool isTensor = false;

        printf("Batch %d\n", i);

        #if VERBOSITY >= 2
            cout << "[GPU] ~ Stream " << tid << ", starting batch " << i << endl;
        #endif

        unsigned int nbBlock;
        switch (searchMode)
        {
            case SM_GPU:
            // case SM_GPU_HALF2:
            {
                cudaErrCheck(cudaMemcpy(&dev_batchBegin[tid], &batchesVector[i].begin, sizeof(unsigned int), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(&dev_batchEnd[tid], &batchesVector[i].end, sizeof(unsigned int), cudaMemcpyHostToDevice));
                N[tid] = batchesVector[i].end - batchesVector[i].begin;

                nbKernelsCuda[tid]++;
                nbQueriesCuda[tid] += N[tid];

                nbBlock = ceil((1.0 * (N[tid])) / (1.0 * BLOCKSIZE));
                break;
            }
            case SM_TENSOR_MQ_HYBRID:
            {
                cudaErrCheck(cudaMemcpy(&dev_batchBegin[tid], &tensorBatchesIndices[i].first, sizeof(unsigned int), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(&dev_batchEnd[tid], &tensorBatchesIndices[i].second, sizeof(unsigned int), cudaMemcpyHostToDevice));
                N[tid] = (tensorBatchesIndices[i].second - tensorBatchesIndices[i].first) * WARP_SIZE;                        

                isTensor = true;
                nbKernelsTensor[tid]++;
                nbQueriesTensor[tid] += batchesVector[i].end - batchesVector[i].begin;

                // nbBlock = ceil(((1.0 * N[tid]) / (1.0 * tensorBlockSize)) * (WARP_SIZE / NB_QUERY_POINTS_TENSOR));
                nbBlock = ceil((1.0 * N[tid]) / (1.0 * tensorBlockSize));
                // nbBlock = N[tid];
                break;
            }
        }

        #if VERBOSITY >= 2
            cout << "[GPU] ~ Batch " << i << ": N = " << N[tid] << ", nbBlock = " << nbBlock << endl;
            // cout << "[GPU] ~ N (1 less): " << N[tid] << ", tid " << tid << '\n';
        #endif

        cudaErrCheck(cudaMemcpyAsync(&dev_N[tid], &N[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid]));

        cnt[tid] = 0;
        cudaErrCheck(cudaMemcpyAsync(&dev_cnt[tid], &cnt[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid]));

        batchOffset[tid] = numBatches; //for the strided
        cudaErrCheck(cudaMemcpyAsync(&dev_offset[tid], &batchOffset[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid]));

        batchNumber[tid] = i;
        cudaErrCheck(cudaMemcpyAsync(&dev_batchNumber[tid], &batchNumber[tid], sizeof(unsigned int), cudaMemcpyHostToDevice, stream[tid]));

        switch (searchMode)
        {
            case SM_GPU:
            {
                cout << "[GPU] ~ Using the CUDA Cores" << endl;
                #if COMPUTE_PREC != INPUT_DATA_PREC
                    distanceCalculationGridCuda<<<nbBlock, BLOCKSIZE, 0, stream[tid]>>>(&dev_batchBegin[tid], &dev_N[tid],
                        dev_datasetAlt, dev_originPointIndex, dev_epsilon, dev_grid, dev_gridLookupArr,
                        dev_gridCellLookupArr, dev_minArrAlt, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells,
                        dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #else
                    distanceCalculationGridCuda<<<nbBlock, BLOCKSIZE, 0, stream[tid]>>>(&dev_batchBegin[tid], &dev_N[tid],
                        dev_database, dev_originPointIndex, dev_epsilon, dev_grid, dev_gridLookupArr,
                        dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid], dev_nNonEmptyCells,
                        dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            }
            case SM_TENSOR_MQ_HYBRID:
            {
                cout << "[GPU] ~ Using the Tensor Cores" << endl;
                #if COMPUTE_PREC == 64
                    // Double precision kernel
                    distanceCalculationGridTensor_multiQueryPoints_double_8_8_4_tensor_mixed<<<nbBlock, tensorBlockSize, 0, stream[tid]>>>(
                        &dev_batchBegin[tid], &dev_batchEnd[tid], dev_database, dev_nbQueryPoints, dev_originPointIndex,
                        dev_tensorBatches, dev_tensorBatchesSize, dev_preComputedSquaredCoordinates, dev_epsilon,
                        dev_grid, dev_gridLookupArr, dev_gridCellLookupArr, dev_minArr, dev_nCells, &dev_cnt[tid],
                        dev_nNonEmptyCells, dev_pointIDKey[tid], dev_pointInDistValue[tid]);
                #endif
                break;
            }
        }

        errCode = cudaGetLastError();
        if (cudaSuccess != errCode)
        {
            cout << "[GPU] ~ ERROR IN KERNEL LAUNCH. ERROR: " << errCode << '\n';
            cout << "  Details: " << cudaGetErrorString(errCode) << endl;
        }

        // find the size of the number of results
        errCode = cudaMemcpyAsync(&cnt[tid], &dev_cnt[tid], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]);
        if (errCode != cudaSuccess)
        {
            cout << "[GPU] ~ Error: getting cnt from GPU Got error with code " << errCode << '\n';
            cout << "  Details: " << cudaGetErrorString(errCode) << endl;
        } else {
            cout << "[GPU] ~ Result set size within epsilon: " << cnt[tid] << endl;
            // cout << "  Details: " << cudaGetErrorString(errCode) << endl;
        }

        if (isTensor)
        {
            nbNeighborsTensor[tid] += cnt[tid];
        } else {
            nbNeighborsCuda[tid] += cnt[tid];
        }

        thrust::device_ptr<int> dev_keys_ptr(dev_pointIDKey[tid]);
        thrust::device_ptr<int> dev_data_ptr(dev_pointInDistValue[tid]);

        try {
            thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), dev_keys_ptr, dev_keys_ptr + cnt[tid], dev_data_ptr);
        } catch(std::bad_alloc &e) {
            std::cerr << "[GPU] ~ Ran out of memory while sorting, " << GPUBufferSize << endl;
            exit(1);
        }

        cudaMemcpyAsync(thrust::raw_pointer_cast(pointIDKey[tid]), thrust::raw_pointer_cast(dev_keys_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);
        cudaMemcpyAsync(thrust::raw_pointer_cast(pointInDistValue[tid]), thrust::raw_pointer_cast(dev_data_ptr), cnt[tid] * sizeof(int), cudaMemcpyDeviceToHost, stream[tid]);

        cudaStreamSynchronize(stream[tid]);

        double tableconstuctstart = omp_get_wtime();
        //set the number of neighbors in the pointer struct:
        (*pointersToNeighbors)[i].sizeOfDataArr = cnt[tid];
        (*pointersToNeighbors)[i].dataPtr = new int[cnt[tid]];

        constructNeighborTableKeyValueWithPtrs(pointIDKey[tid], pointInDistValue[tid], neighborTable, (*pointersToNeighbors)[i].dataPtr, &cnt[tid]);

        double tableconstuctend = omp_get_wtime();

        #if VERBOSITY >= 2
            cout << "[GPU] ~ Table construct time: " << tableconstuctend - tableconstuctstart << endl;
        #endif

        //add the batched result set size to the total count
        totalResultsLoop += cnt[tid];

        #if VERBOSITY >= 2
            cout << "[GPU] ~ Running total of total size of result array after batch " << i << ": " << totalResultsLoop << endl;
        #endif
    } // pragma omp / for numBatches
    double tEndCompute = omp_get_wtime();
    cout << "[GPU | RESULT] ~ Compute time for the GPU: " << tEndCompute - tStartCompute << '\n';

    *totalNeighbors = totalResultsLoop;

    for (unsigned int i = 0; i < GPUSTREAMS; ++i)
    {
        (*totalCandidatesCuda) += nbCandidatesCuda[i];
        (*totalCandidatesTensor) += nbCandidatesTensor[i];

        (*totalQueriesCuda) += nbQueriesCuda[i];
        (*totalQueriesTensor) += nbQueriesTensor[i];

        (*totalKernelsCuda) += nbKernelsCuda[i];
        (*totalKernelsTensor) += nbKernelsTensor[i];

        (*totalNeighborsCuda) += nbNeighborsCuda[i];
        (*totalNeighborsTensor) += nbNeighborsTensor[i];
    }

    double tFunctionEnd = omp_get_wtime();

    double tFreeStart = omp_get_wtime();

	for (int i = 0; i < GPUSTREAMS; i++)
    {
		errCode = cudaStreamDestroy(stream[i]);
		if (errCode != cudaSuccess) {
			cout << "[GPU] ~ Error: destroying stream" << errCode << '\n';
            cout.flush();
		}
	}

	delete totalResultSetCnt;
	delete[] cnt;
	delete[] N;
	delete[] batchOffset;
	delete[] batchNumber;

	cudaFree(dev_N);
	cudaFree(dev_cnt);
	cudaFree(dev_offset);
	cudaFree(dev_batchNumber);

	for (int i = 0; i < GPUSTREAMS; i++)
    {
		cudaFree(dev_pointIDKey[i]);
		cudaFree(dev_pointInDistValue[i]);

		cudaFreeHost(pointIDKey[i]);
		cudaFreeHost(pointInDistValue[i]);
	}

	cudaFreeHost(pointIDKey);
	cudaFreeHost(pointInDistValue);

    #if COMPUTE_PREC != INPUT_DATA_PREC
        cudaFree(dev_datasetAlt);
        cudaFree(dev_minArrAlt);
    #endif

	double tFreeEnd = omp_get_wtime();

    cout << "[GPU] ~ Time freeing memory: " << tFreeEnd - tFreeStart << '\n';
    cout.flush();
}



void constructNeighborTableKeyValueWithPtrs(
    int* pointIDKey,
    int* pointInDistValue,
    struct neighborTableLookup* neighborTable,
    int* pointersToNeighbors,
    unsigned int* cnt)
{
	//copy the value data:
	std::copy(pointInDistValue, pointInDistValue + (*cnt), pointersToNeighbors);

	//Step 1: find all of the unique keys and their positions in the key array
	unsigned int numUniqueKeys = 0;

	std::vector<keyData> uniqueKeyData;

	keyData tmp;
	tmp.key = pointIDKey[0];
	tmp.position = 0;
	uniqueKeyData.push_back(tmp);

	//we assign the ith data item when iterating over i+1th data item,
	//so we go 1 loop iteration beyond the number (*cnt)
	for (unsigned int i = 1; i < (*cnt) + 1; ++i)
    {
		if (pointIDKey[i - 1] != pointIDKey[i]){
			numUniqueKeys++;
			tmp.key = pointIDKey[i];
			tmp.position = i;
			uniqueKeyData.push_back(tmp);
		}
	}

	//insert into the neighbor table the values based on the positions of
	//the unique keys obtained above.
	for (unsigned int i = 0; i < uniqueKeyData.size() - 1; ++i)
    {
		int keyElem = uniqueKeyData[i].key;
		neighborTable[keyElem].pointID = keyElem;
		neighborTable[keyElem].indexmin = uniqueKeyData[i].position;
		neighborTable[keyElem].indexmax = uniqueKeyData[i + 1].position;

		//update the pointer to the data array for the values
		neighborTable[keyElem].dataPtr = pointersToNeighbors;
	}
}
