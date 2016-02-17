CXX=g++
CC=$(CXX)

#
CFLAGS= -O2 -march=native -pipe -funroll-loops
CXXFLAGS=$(CFLAGS)

OBJS= main.o
EXE=mc

all:main
	mv main $(EXE)
main:$(OBJS)

clean:
	rm -f $(OBJS) $(EXE)

redo: clean all
