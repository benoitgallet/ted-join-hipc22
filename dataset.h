#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <istream>
#include <fstream>
#include <sstream>
#include <iostream>

#include "params.h"


void importDataset (
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector,
    char* filename);


void importDataset_v2 (
    std::vector< std::vector<INPUT_DATA_TYPE> >* inputVector,
    char* filename);


#endif
