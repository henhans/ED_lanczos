all: main

#LIBS = -L$LIBRARY_PATH -I$INCLUDE -lmkl_blas95_lp64 -lmkl_intel_lp64 -lmkl_lapack95_lp64 -lmkl_intel_thread -lmkl_core -lm  -liomp5 #-llapack -lblas
LIBS = /usr/lib/libgsl.a
CC = g++
CFLAGS = -Os -funroll-loops -Wall -pedantic

main: main.o
	$(CC) $(CFLAGS) main.o -o main $(LIBS)


clean: 
	rm -rf *.o main

