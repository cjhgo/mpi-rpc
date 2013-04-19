$(shell mkdir -p build)

SRC := $(wildcard *.cc)
HEADERS := $(wildcard *.h) Makefile
INCLUDE := -I.
OBJ := $(patsubst %.cc,build/%.o,$(SRC))

TEST_SRC := $(wildcard test/*.cc)
TESTS := $(patsubst test/%.cc,build/%,$(TEST_SRC))

CFLAGS := -Wall -ggdb2 -pthread
CXXFLAGS := $(CFLAGS) -std=c++11 
CXX := mpic++

LDFLAGS := 

build/%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -c -o $@

all: build/libmpirpc.a $(TESTS)

test: $(TESTS)
	for t in $(TESTS); do echo Running $$t; mpirun -n 1 $$t; done

clean:
	rm -rf build/*

build/libmpirpc.a : $(OBJ) $(HEADERS)
	ar rcs $@ $(OBJ)
	ranlib $@
