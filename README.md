# Tensor Euclidean Distance Joins (TED-Join)

Authors: Benoit Gallet and Michael Gowanlock

Insitution: Northern Arizona University, *School of Informatics, Computing, and Cyber Systems*

E-mails: <bg724@nau.edu> | <galletbenoit@microsoft.com>, <michael.gowanlock@nau.edu>

Corresponding publication:
- Benoit Gallet and Michael Gowanlock. Leveraging GPU Tensor Cores for Double Precision Euclidean Distance Calculations, ***International Conference on High Performance Computing, Data, and Analytics (HiPC)***, 2022. [Article](https://doi.org/10.1109/HiPC56025.2022.00029)

## Introduction
TED-Join is an algorithm using Nvidia GPU tensor cores to compute double precision Euclidean distances on multi-dimensional datasets, and which can be used to compute as a similarity self-join algorithm. To increase the general performance, we use an index to prune the total number of distance calculations we compute using a grid index.

This algorithm is capable of computing Euclidean distances in double precision (FP64) using either CUDA or Tensor cores. 

When using CUDA cores, a GPU thread computes Euclidean distances between its assigned *query point* and every points it finds in the grid index (*candidate points*). When using Tensor cores, a warp computes Euclidean distance between its 8 assigned *query points* and every points it finds in the grid index, 8 *candidate points* at a time.

## Pre-requisites
TED-Join is written using C++ and the CUDA API. To use this project, you will need to have CUDA installed, as well as an Nvidia GPU with compute capability 8.0 at least (CC>=80), i.e., generation Ampere or newer.

This repo also uses the Boost library, which is thus necessary to compile this project.

## Parameters
The file `params.h` contains the different parameters used by TED-Join:
- `INPUT_DATA_DIM`: dataset dimensionality (i.e., number of features).
- `INDEXED_DIM`: number of dimensions to index data on for faster computation. For best performance, `INDEXED_DIM` should be <= 6.
- `COMPUTE_DIM`: dimensionality for the tensor cores. When using FP64, `COMPUTE_DIM` is the next multiple of 4 after `INPUT_DATA_DIM`. E.g., if `INPUT_DATA_DIM=18`, `COMPUTE_DIM` should be 20.
- `BLOCK_SIZE`: number of threads per block, only used with the CUDA cores compute mode.
- `WARP_PER_BLOCK`: number of warps per thread block, only used with the Tensor cores compute mode.
- `GPUSTREAMS`: number of streams to use to overlap memory copies and kernel launches.
- `INPUT_DATA_PREC`: data precision of the input dataset, in bits. For best results, should be set to 64.
- `COMPUTE_PREC`: precision used to compute Euclidean distances, in bits. Because this is a double precision algorithm, should be set to 64.
- `ACCUM_PREC`: precision used to store the results of Euclidean distances, in bits. Because this is a double precision algorithm, should be set to 64.

***Note: `ACCUM_PREC` >= `COMPUTE_PREC` >= `INPUT_DATA_PREC`***.

- `ADDITIONAL_POINTS`: additional filler points necessary when using the Tensor cores if the number of points in the dataset is not a multiple of 8. For double precision, should be set to 7.
- `ILP`: Instrucion Level Parallelism, only used with CUDA cores to increase the performance and instruction level parallelism when computing the Euclidean distance between a query point and a candidate point.
- `SHORT_CIRCUIT`: short circuits the computation if the distance between a query point and a candidate point reaches epsilon before calculating all `COMPUTE_DIM` features of the points.
- `GPU_BUFFER_SIZE`: result array size to store the pairs of points that are within epsilon of each others. Directly influences the number of batches, and the performance of the algorithm. *Note: the result array is uses pinned memory. Thus, a large value will significantly increase the allocation time, without necessarily significantly improving the performance of the Euclidean distance calculations*.

***Note: generally, you will need to recompile the entire project when changing values in the `params.h` file, using `make clean; make`***.

## Datasets
TED-Join uses textfile datasets, formatted as follows:
- One point per line.
- Coordinates/features are separated by commas.
- Coordinates/features need to be normalized between [0, 1].

The folder `datasets` contains source files to generate exponentially distributed datasets. To generate datasets, use the following commands:
```sh
$ cd datasets
make
```
This produces the executable `genDataset`, that you can use as follows:
```sh
./genDataset number_of_features number_of_points
```
Doing `./genDataset 2 2000000` produces a dataset of 2,000,000 points in 2 dimensions, which will be named `dataset_fixed_len_pts_expo_NDIM_2_pts_2000000.txt`.

## Makefile
Changing the `makefile` at the root of the repository might be necessary.

More specifically, you should modify the rules/create a new rule to fit your GPU and its compute capability. You should also indicate the correct location to the Boost library.

## Utilisation
At the root of the repository, use the command `make` to compile the project, which will produce the `main` executable if the compilation is successful. ***Note: recompilation might be necessary when you modify values in the `params.h` file, using `make clean; make`***.

To start the program, use
```sh
./main dataset epsilon compute_mode
```

Where the arguments are as the following:
- `dataset`: an input dataset, as explained in the *Datasets* section above.
- `epsilon`: distance threshold to consider pairs of points to be similar.
- `algorithm`: 11 to use CUDA cores, 21 to use Tensor cores.

The algorithm might output a lot of information during the computation. To only keep information relevant to the results/performance, we recommend to pipe the output using `grep`:
```sh
./main dataset epsilon compute_mode | grep "Result"
```
