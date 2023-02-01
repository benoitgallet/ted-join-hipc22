#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>

#include "omp.h"
#include <boost/multiprecision/cpp_int.hpp>

#include "params.h"
#include "structs.h"
#include "main.h"
#include "dataset.h"
#include "index.h"
#include "sort_by_workload.h"
#include "gpu_join.h"


int main (int argc, char* argv[])
{
    fprintf(stdout, "\n\n========== TED-Join ==========\n\n\n");
    double tStartProgram = omp_get_wtime();

    if (NB_ARGS_MAX != argc)
    {
        fprintf(stderr, "[Main] ~ Expected %d arguments, found %d\n", NB_ARGS_MAX - 1, argc);
        fprintf(stderr, "[Main] ~ Arguments: filename epsilon searchmode\n");
        fprintf(stderr, "[Main] ~   Epsilon: Should be between 0 (excluded) and 1 (included).\n");
        fprintf(stderr, "[Main] ~   Search modes: \n");
        fprintf(stderr, "[Main] ~     --- CUDA cores only ---\n");
        fprintf(stderr, "[Main] ~      11: CUDA cores only.\n");
        fprintf(stderr, "[Main] ~      /!\\ WIP /!\\ 12: CUDA cores only, half2 precision.\n");
        fprintf(stderr, "[Main] ~     --- Tensor cores ---\n");
        fprintf(stderr, "[Main] ~     23: CUDA-Tensor cores mix, multiple query points per complete computation step.\n");
        return 1;
    }



    /***** Input parameters *****/
    char filename[256];
    strcpy(filename, argv[FILENAME_ARG]);
    ACCUM_TYPE epsilon = atof(argv[EPSILON_ARG]);
    unsigned int searchMode = atoi(argv[SEARCHMODE_ARG]);

    fprintf(stdout, "[Main] ~ Dataset: %s\n", filename);
    fprintf(stdout, "[Main] ~ Epsilon: %f\n", (double)epsilon);
    fprintf(stdout, "[Main] ~ Search mode: %d\n", searchMode);



    /***** Import dataset *****/
    std::vector< std::vector<INPUT_DATA_TYPE> > inputVector;
    fprintf(stdout, "[Main] ~ Importing the dataset\n");
    double tStartReadDataset = omp_get_wtime();
    importDataset_v2(&inputVector, filename);
    double tEndReadDataset = omp_get_wtime();
    double timeReadDataset = tEndReadDataset - tStartReadDataset;
    unsigned int nbQueryPoints = inputVector.size();
    fprintf(stdout, "[Main] ~ Number of query points read: %d\n", nbQueryPoints);
    fprintf(stdout, "[Main | Time] ~ Time to read the dataset: %f\n", timeReadDataset);



    /***** Reorder features of the dataset based on their variance *****/
    fprintf(stdout, "[Main] ~ Reordering the dataset by the variance of its features\n");
    double tStartReorderDim = omp_get_wtime();
    reorderDimensionsByVariance(&inputVector);
    double tEndReorderDim = omp_get_wtime();
    fprintf(stdout, "[Main | Time] ~ Time to reorder the features by their variance: %f\n", tEndReorderDim - tStartReorderDim);

    INPUT_DATA_TYPE* database = new INPUT_DATA_TYPE[(nbQueryPoints + ADDITIONAL_POINTS) * COMPUTE_DIM];
    for (unsigned int i = 0; i < nbQueryPoints; ++i)
    {
        for (unsigned int j = 0; j < INPUT_DATA_DIM; ++j)
        {
            database[i * COMPUTE_DIM + j] = inputVector[i][j];
        }
        for (unsigned int j = INPUT_DATA_DIM; j < COMPUTE_DIM; ++j)
        {
            database[i * COMPUTE_DIM + j] = (INPUT_DATA_TYPE)0.0;
        }
    }
    for (unsigned int i = 0; i < ADDITIONAL_POINTS; ++i)
    {
        for (unsigned int j = 0; j < COMPUTE_DIM; ++j)
        {
            database[(nbQueryPoints + i) * COMPUTE_DIM + j] = (INPUT_DATA_TYPE)0.0;
        }
    }



    /***** Construct the grid index *****/
    INPUT_DATA_TYPE* minArr = new INPUT_DATA_TYPE[INDEXED_DIM];
    INPUT_DATA_TYPE* maxArr = new INPUT_DATA_TYPE[INDEXED_DIM];
    unsigned int* nCells = new unsigned int[INDEXED_DIM];
    uint64_t totalCells = 0;
    unsigned int nNonEmptyCells = 0;

    generateNDGridDimensions(&inputVector, epsilon, minArr, maxArr, nCells, &totalCells);
    fprintf(stdout, "[Main] ~ Total cells (including empty): %lu\n", totalCells);

    inputVector.clear();
    inputVector.shrink_to_fit();

    struct grid* index;
    struct gridCellLookup* gridCellLookupArr;
    unsigned int* indexLookupArr = new unsigned int[nbQueryPoints];

    ACCUM_TYPE* dev_epsilon;
    INPUT_DATA_TYPE* dev_database;
    struct grid* dev_index;
    unsigned int* dev_indexLookupArr;
    struct gridCellLookup* dev_gridCellLookupArr;
    INPUT_DATA_TYPE* dev_minArr;
    unsigned int* dev_nCells;
    unsigned int* dev_nNonEmptyCells;

    double tStartIndex = omp_get_wtime();
    gridIndexingGPU(&nbQueryPoints, totalCells, database, &dev_database, &epsilon, &dev_epsilon, minArr, &dev_minArr,
                    &index, &dev_index, indexLookupArr, &dev_indexLookupArr, &gridCellLookupArr, &dev_gridCellLookupArr,
                    &nNonEmptyCells, &dev_nNonEmptyCells, nCells, &dev_nCells);
    double tEndIndex = omp_get_wtime();
    fprintf(stdout, "[Main | Time] ~ Time to generate the index: %f\n", tEndIndex - tStartIndex);



    /***** Sorting the points by their workload *****/
    unsigned int* originPointIndex;
    unsigned int* dev_originPointIndex;
    boost::multiprecision::uint128_t totalCandidates = 0;
    struct schedulingCell* sortedDatabaseTmp;

    double tStartSort = omp_get_wtime();
    sortByWorkLoad(searchMode, &nbQueryPoints, &totalCandidates, &sortedDatabaseTmp, &epsilon, &dev_epsilon,
                   database, &dev_database, index, &dev_index, indexLookupArr, &dev_indexLookupArr, gridCellLookupArr,
                   &dev_gridCellLookupArr, minArr, &dev_minArr, nCells, &dev_nCells, &nNonEmptyCells, &dev_nNonEmptyCells,
                   &originPointIndex, &dev_originPointIndex);
    double tEndSort = omp_get_wtime();
    fprintf(stdout, "[Main | Time] ~ Time to sort the points by workload: %f\n", tEndSort - tStartSort);



    /***** Neighbor table and result metrics *****/
    neighborTableLookup* neighborTable = new neighborTableLookup[nbQueryPoints];
    std::vector<struct neighborDataPtrs> pointersToNeighbors;

    uint64_t totalNeighbors = 0;
    uint64_t totalNeighborsCuda = 0;
    uint64_t totalNeighborsTensor = 0;
    boost::multiprecision::uint128_t totalCandidatesCuda = 0;
    boost::multiprecision::uint128_t totalCandidatesTensor = 0;
    unsigned int totalQueriesCuda = 0;
    unsigned int totalQueriesTensor = 0;
    unsigned int totalKernelCuda = 0;
    unsigned int totalKernelTensor = 0;



    /***** Compute the distance similarity join *****/
    double tStartJoin = omp_get_wtime();
    GPUJoinMainIndex(searchMode, database, dev_database, &nbQueryPoints, &epsilon, dev_epsilon, index, dev_index,
            indexLookupArr, dev_indexLookupArr, gridCellLookupArr, dev_gridCellLookupArr, minArr, dev_minArr,
            nCells, dev_nCells, &nNonEmptyCells, dev_nNonEmptyCells, originPointIndex, dev_originPointIndex,
            neighborTable, &pointersToNeighbors, &totalNeighbors, &totalNeighborsCuda, &totalNeighborsTensor,
            &totalCandidatesCuda, &totalCandidatesTensor, &totalQueriesCuda, &totalQueriesTensor,
            &totalKernelCuda, &totalKernelTensor, sortedDatabaseTmp);
    double tEndJoin = omp_get_wtime();
    fprintf(stdout, "[Main | Time] ~ Time to join: %f\n", tEndJoin - tStartJoin);



    /***** Output the neighbor table (for comparison or checking purposes, or just to keep the result somewhere) *****/
    #if OUTPUT_NEIGHBORS
        std::ofstream outputNeighborsFile("output_neighbors.txt");
        for (unsigned int i = 0; i < nbQueryPoints; ++i)
        {
            std::sort(neighborTable[i].dataPtr + neighborTable[i].indexmin, neighborTable[i].dataPtr + neighborTable[i].indexmax + 1);
            outputNeighborsFile << i << ": ";
            for (unsigned int j = neighborTable[i].indexmin; j <= neighborTable[i].indexmax; ++j)
            {
                outputNeighborsFile << neighborTable[i].dataPtr[j] << ", ";
            }
            outputNeighborsFile << '\n';
        }

        outputNeighborsFile.close();
    #endif



    /***** Free memory *****/
    double tStartFree = omp_get_wtime();
    delete[] minArr;
    delete[] maxArr;
    delete[] nCells;
    delete[] neighborTable;
    delete[] indexLookupArr;
    delete[] database;
    delete[] sortedDatabaseTmp;
    delete[] originPointIndex;

    cudaFree(dev_epsilon);
    #if INPUT_DATA_PREC == COMPUTE_PREC
        cudaFree(dev_database);
    #endif
    cudaFree(dev_index);
    cudaFree(dev_indexLookupArr);
    cudaFree(dev_gridCellLookupArr);
    cudaFree(dev_minArr);
    cudaFree(dev_nCells);
    cudaFree(dev_nNonEmptyCells);
    cudaFree(dev_originPointIndex);
    double tEndFree = omp_get_wtime();
    fprintf(stdout, "[Main | Time] ~ Time to free memory: %f\n", tEndFree - tStartFree);



    /***** Results *****/
    double tEndProgram = omp_get_wtime();
    double timeBench = tEndProgram - tStartProgram - timeReadDataset;
    unsigned int selectivity = abs((long long)(totalNeighbors - nbQueryPoints)) / nbQueryPoints;
    fprintf(stdout, "[Main | Result] ~ Total execution time: %f\n", timeBench);
    fprintf(stdout, "[Main | Result] ~ Total result set size: %lu (S = %d)\n", totalNeighbors, selectivity);
    fprintf(stdout, "   [Main | Result] ~ Total result set size CUDA cores: %lu\n", totalNeighborsCuda);
    fprintf(stdout, "   [Main | Result] ~ Total result set size Tensor cores: %lu\n", totalNeighborsTensor);
    fprintf(stdout, "[Main | Result] ~ Query points:\n");
    fprintf(stdout, "   [Main | Result] ~ Number of queries computed by CUDA cores: %d\n", totalQueriesCuda);
    fprintf(stdout, "   [Main | Result] ~ Number of queries computed by Tensor cores: %d\n", totalQueriesTensor);
    fprintf(stdout, "[Main | Result] ~ Kernel invocations:\n");
    fprintf(stdout, "   [Main | Result] ~ Number of kernel invocations using CUDA cores: %d\n", totalKernelCuda);
    fprintf(stdout, "   [Main | Result] ~ Number of kernel invocations using Tensor cores: %d\n", totalKernelTensor);
    std::cout << "[Main | Result] ~ Refined candidate points:\n";
    std::cout << "   [Main | Result] ~ Estimated number of candidate points refined by CUDA cores: " << totalCandidatesCuda << '\n';
    std::cout << "   [Main | Result] ~ Estimated number of candidate points refined by Tensor cores: " << totalCandidatesTensor << '\n';



    /***** Writing the results into file *****/
    #if OUTPUT_RESULTS
        std::ofstream outputResultFile;
        std::ifstream inputResultFile("hct_join.txt");
        outputResultFile.open("hct_join.txt", std::ios::out | std::ios::app);
        if (inputResultFile.peek() == std::ifstream::traits_type::eof())
        {
            // File is empty, write the header first
            outputResultFile << "Dataset, epsilon, searchMode, executionTime, totalNeighbors, selectivity, inputDim, indexedDim, computeDim, "
                << "blockSize, warpPerBlock, gpuStreams, computePrec, accumPrec, ILP, shortCircuit, reorderDimByVar, Kahan, reorderDataset\n";
        }
        outputResultFile << filename << ", " << (double)epsilon << ", " << searchMode << ", " << timeBench << ", " << totalNeighbors << ", "
                        << selectivity << ", " << INPUT_DATA_DIM << ", " << INDEXED_DIM << ", " << COMPUTE_DIM << ", " << BLOCKSIZE << ", "
                        << WARP_PER_BLOCK << ", " << GPUSTREAMS << ", " << COMPUTE_PREC << ", " << ACCUM_PREC << ", "
                        << ILP << ", " << SHORT_CIRCUIT << ", " << REORDER_DIM_BY_VAR << ", " << KAHAN_CUDA << ", " << REORDER_DATASET << std::endl;
        
        outputResultFile.close();
        inputResultFile.close();
    #endif
}
