FLAGS=-O3 
#FLAGS=-O0 -g
OBJ=main.o gpu.o grid.o hydro.o cpu.o initial.o sub_grid.o
HEADERS=defs.hpp hydro.hpp gpu.hpp cpu.hpp grid.hpp initial.hpp sub_grid.hpp
LIBRARIES=-lsiloh5 -lpthread -lgomp
CXX=g++
CU=nvcc
LD=nvcc
COMMON_FLAGS=-std=c++11 $(FLAGS) 
CXX_FLAGS=-march=native -ffast-math -fopenmp -pthread
CU_FLAGS=--device-c -use_fast_math -gencode arch=compute_35,code=sm_35 -Xcompiler "$(CXX_FLAGS)" --default-stream per-thread
LD_FLAGS=-gencode arch=compute_35,code=sm_35 --default-stream per-thread
VPATH=./src/:./obj/

%.o: %.cpp  $(HEADERS)$
	g++ $(COMMON_FLAGS) $(CXX_FLAGS) -o ./obj/$@ -c $<

%.o: %.cu  $(HEADERS)$
	nvcc $(COMMON_FLAGS) $(CU_FLAGS) -o ./obj/$@ -c $<

all: X

X: $(OBJ)

	$(LD) $(LD_FLAGS) ./obj/*.o -o X $(LIBRARIES)

clean:

	rm -f X
	rm -f ./obj/*.o




