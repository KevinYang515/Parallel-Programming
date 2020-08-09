NVFLAGS  := -std=c++11 -Xcompiler="-fopenmp" -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
NVCCLDFLAGS  = -L/home/pp19/share/lib -Xlinker=-rpath,/home/pp19/share/lib -llab5

EXES     := hw4-2

alls: $(EXES)

clean:
	rm -f $(EXES)


hw4-2: hw4-2.cu
	nvcc $(NVFLAGS) $(NVCCLDFLAGS) -o $@ $?
