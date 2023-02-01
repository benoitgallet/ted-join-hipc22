#include <vector>
#include <istream>
#include <fstream>
#include <sstream>
#include <iostream>

#include "params.h"
#include "dataset.h"


void importDataset (
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector,
    char* filename)
{
    std::vector<INPUT_DATA_TYPE> tmpAllData;
    std::ifstream in(filename);
    for (std::string f; getline(in, f, ','); )
    {
        INPUT_DATA_TYPE i;
        std::stringstream ss(f);
        while (ss >> i)
        {
            tmpAllData.push_back(i);
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
    }

    unsigned int cnt = 0;
    const unsigned int totalPoints = (unsigned int)tmpAllData.size() / INPUT_DATA_DIM;
    fprintf(stdout, "[Dataset] ~ Data import: Total data points: %d\n", totalPoints);
    fprintf(stdout, "[Dataset] ~ Data import: Total size of all data (1-D) vect (number of points * GPUNUMDIM): %zu\n", tmpAllData.size());

    for (int i = 0; i < totalPoints; ++i)
    {
        std::vector<INPUT_DATA_TYPE> tmpPoint;
        for (int j = 0; j < INPUT_DATA_DIM; ++j)
        {
            tmpPoint.push_back(tmpAllData[cnt]);
            cnt++;
        }
        inputVector->push_back(tmpPoint);
    }
}


void importDataset_v2 (
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector,
    char* filename)
{
    FILE* fptr;
    fptr = fopen(filename, "r");
    if (NULL == fptr)
    {
        fprintf(stderr, "[Dataset] ~ Could not open the input file %s\n", filename);
        exit(1);
    }
    double check = 0;

    std::vector<INPUT_DATA_TYPE> data;

    int dimCounter = 0;

    while (fscanf(fptr, "%lf, ", &check) == 1 || fscanf(fptr, "%lf ", &check) == 1)
    {
        data.push_back(check);
        dimCounter++;

        if (INPUT_DATA_DIM == dimCounter)
        {
            dimCounter = 0;
            inputVector->push_back(data);
            data.clear();
        }
    }
    fclose(fptr);
}
