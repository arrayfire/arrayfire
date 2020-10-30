
all:	async_queue.hpp parallel.hpp test.cpp
	g++-4.9 -O3 -g -Wall -std=c++11 test.cpp -o test
