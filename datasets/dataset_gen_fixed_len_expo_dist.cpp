#include <stdio.h>
#include <random>
#include <fstream>
#include <math.h>
#include <iostream>
#include <string.h>
#include <iostream>

// static seed so we can reproduce the data on other machines
#define SEED 2137834274
#define EXPONENTIAL_LAMBDA_PARAMETER 40.0

using namespace std;

int main(int argc, char *argv[])
{


	if (3 != argc)
	{
		cerr << "Incorrect number of input parameters\n";
		cerr << "Parameters should be: number_of_dimensions number_of_points\n";
		return 1;
	}

	char inputnumdim[512];
	char inputdatapoints[512];

	strcpy(inputnumdim,argv[1]);
	strcpy(inputdatapoints,argv[2]);

	unsigned int numDim = atoi(argv[1]);
	unsigned int dataPoints = atoi(argv[2]);

	printf("Total datapoints: %d\n", dataPoints);
	double datasetsize = ((dataPoints * 8.0 * numDim) / (1024.0 * 1024.0));
	printf("Size of dataset (MiB): %f\n", datasetsize);

	//for my file formatting
	std::ofstream outputFile;
	std::string fname="dataset_fixed_len_pts_expo_NDIM_";
	fname += std::to_string(numDim);
	fname += "_pts_";
	fname += std::to_string(dataPoints);
	fname += ".txt";
	outputFile.open(fname);

	std::mt19937 gen(SEED); // Standard mersenne_twister_engine
	std::exponential_distribution<double> dis (EXPONENTIAL_LAMBDA_PARAMETER); //lambda = 40.0

	double total = 0;
	unsigned int length = 1;

	for (int i = 0; i < dataPoints; ++i)
	{
		for (int j = 0; j < numDim; ++j)
		{
			double val = 0;
			//generate value until its in the range 0-1
			do
			{
				val = dis(gen) * length;
			} while (val < 0 || val > 1);

			total += val;

			if (j < numDim - 1)
			{
				outputFile << val << ", "; // Comma-separated coordinates
			} else {
				outputFile << val; // Last coordinate of the point
			}
		}
		outputFile << std::endl;
	}

	printf("Average of values generated: %f\n",total / (dataPoints * numDim * 1.0));

	outputFile.close();

	return 0;
}
