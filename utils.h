#ifndef UTILS_H
#define UTILS_H


#define cudaErrCheck(errCode) { cudaErrCheck_((errCode), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t errCode, const char* file, int line);


#endif
