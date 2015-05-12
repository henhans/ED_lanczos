all: exactDiagonalization

LIBS = -L$LIBRARY_PATH -I$INCLUDE -lmkl_blas95_lp64 -lmkl_intel_lp64 -lmkl_lapack95_lp64 -lmkl_intel_thread -lmkl_core -lm  -liomp5 -lgsl
#LIBS = -llapack -lblas -lgsl
CC = icpc#g++
CFLAGS = -Os -g -funroll-loops -Wall -pedantic

exactDiagonalization: exactDiagonalization.o matrix.o
	$(CC) $(CFLAGS) exactDiagonalization.o matrix.o -o exactDiagonalization.out $(LIBS)

exactDiagonalization.o: exactDiagonalization.cpp
	$(CC) -c exactDiagonalization.cpp

matrix.o: matrix.cpp
	$(CC) -c matrix.cpp

clean: 
	rm -rf *.o exactDiagonalization.out

