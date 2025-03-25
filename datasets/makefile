SOURCE = dataset_gen_fixed_len_expo_dist.cpp
EXECUTABLE = genDataset
CXX = g++
FLAGS = -std=c++14 -O3

all: $(EXECUTABLE)

$(EXECUTABLE):
	$(CXX) $(FLAGS) $(SOURCE) -o $(EXECUTABLE)

clean:
	rm $(EXECUTABLE)
