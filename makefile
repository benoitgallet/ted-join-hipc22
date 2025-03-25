SOURCES = utils.cu dataset.cu index.cu main.cu sort_by_workload.cu gpu_join.cu kernel_join.cu
OBJECTS = utils.o dataset.o index.o sort_by_workload.o gpu_join.o kernel_join.o main.o
CC = nvcc
EXECUTABLE = main

FLAGS = -std=c++14 -O3 -Xcompiler -fopenmp -lcuda -lineinfo -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES

# Add or modify the rules below to fit your architecture/compute capability and the location of the Boost library.
ampere:
	echo "Compiling for Ampere generation (CC=86)"
	$(MAKE) all ARCH=compute_86 CODE=sm_86 BOOST=/home/benoit/research/boost_1_75_0

monsoon:
	echo "Compiling for Monsoon cluster with A100 (CC=80)"
	$(MAKE) all ARCH=compute_80 CODE=sm_80 BOOST=/home/bg724/boost_1_76_0

all: $(EXECUTABLE)

%.o: %.cu
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(BOOST) $^ -c $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(FLAGS) -arch=$(ARCH) -code=$(CODE) -I$(BOOST) $^ -o $@

clean:
	rm $(OBJECTS)
	rm $(EXECUTABLE)
