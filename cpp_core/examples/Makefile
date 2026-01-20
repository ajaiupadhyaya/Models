# Makefile for C/C++ Quantitative Finance Library Examples

CC = gcc
CXX = g++
CFLAGS = -O3 -march=native -Wall -Wextra -I../include
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra -I../include
LDFLAGS = -lm

# Targets
all: example_c

example_c: example_c.c
	$(CC) $(CFLAGS) -o example_c example_c.c $(LDFLAGS)

clean:
	rm -f example_c

run: example_c
	./example_c

.PHONY: all clean run
