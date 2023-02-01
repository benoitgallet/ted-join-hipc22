#ifndef PARAMS_H
#define PARAMS_H

// Input data dimensionality (i.e., number of features of the dataset).
#define INPUT_DATA_DIM 18

// Number of dimensions in which to index the data.
// INDEXED_DIM <= INPUT_DATA_DIM.
#define INDEXED_DIM 6

// Dimensionality used for the tensor cores, based on the dimensionality of the data.
// Should fit the size of the matrices used by the tensor cores, depending on the precision and configuration.
// Typically, the next multiple of 8 or 16 of INPUT_DATA_DIM.
#define COMPUTE_DIM 32

// Number of threads per block for the CUDA cores computation.
#define BLOCKSIZE 256
// Number of warp per block and point per warp for the Tensor cores computation.
#define WARP_PER_BLOCK 4

// Number of GPU streams to use to overlap kernel invocations with data transfers.
#define GPUSTREAMS 3

// Precision of the input data, in which to compute in, and in which to accumulate the result in.
// Used by both the CUDA cores and Tensor cores.
// Possible values are:
//   16: half precision
//   32: single precision
//   64: double precision
// COMPUTE_PREC <= ACCUM_PREC
#define INPUT_DATA_PREC 64
#define COMPUTE_PREC 64
#define ACCUM_PREC 64

// Add extra points at the end of the dataset to not get segmentation faults when filling tensor core fragments
#define ADDITIONAL_POINTS 7

// Amount of Insutrction Level Parallelism for the CUDA cores.
#define ILP 8

// 1 to periodically check the computed distance between a query and a candidate, and short-circuit its computation
// if it reaches epsilon.
#define SHORT_CIRCUIT 1

// Level of verbosity for informative or error messages.
// A higher level prompts more information about the ongoing of the program.
// Still WIP
#define VERBOSITY 1

// Reorder the features of the data based on their variance.
#define REORDER_DIM_BY_VAR 1

// Result array size (number of elements)
// A larger size implies fewer batches, and vice versa
#define GPU_BUFFER_SIZE 50000000

#define OUTPUT_DEBUG 0
#define OUTPUT_DEBUG_SIZE 50000000
#define OUTPUT_DEBUG_TENSOR 0
#define OUTPUT_NEIGHBORS 0
#define OUTPUT_RESULTS 1



/*********************************************************************/
/*                 Code below should not be modified                 */
/*********************************************************************/



#define NB_ARGS_MAX 4
#define FILENAME_ARG 1
#define EPSILON_ARG 2
#define SEARCHMODE_ARG 3

#define SM_GPU 11
// Compute multiple query points from a same cell, using a combination of tensor and CUDA cores
#define SM_TENSOR_MQ_HYBRID 21

#define WARP_SIZE 32

#if COMPUTE_PREC == 16
    #define NB_QUERY_POINTS_TENSOR 16
#else
    #define NB_QUERY_POINTS_TENSOR 8
#endif

#if INPUT_DATA_PREC == 16
    #define INPUT_DATA_TYPE half
#else
    #if INPUT_DATA_PREC == 32
        #define INPUT_DATA_TYPE float
    #else
        #define INPUT_DATA_TYPE double
    #endif
#endif

#if COMPUTE_PREC == 16
    #define COMPUTE_TYPE half
#else
    #if COMPUTE_PREC == 32
        #define COMPUTE_TYPE float
    #else
        #define COMPUTE_TYPE double
    #endif
#endif

#if ACCUM_PREC == 16
    #define ACCUM_TYPE half
#else
    #if ACCUM_PREC == 32
        #define ACCUM_TYPE float
    #else
        #define ACCUM_TYPE double
    #endif
#endif

#endif
