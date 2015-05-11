Using ArrayFire on Linux {#using_on_linux}
=====

Among the many possible build systems on Linux we suggest using ArrayFire with
either CMake or Makefiles with CMake being the preferred build system.

## CMake

This is the suggested method of using ArrayFire on Linux. ArrayFire ships with
support for CMake by default, including a series of `Find` scripts installed
in the `/usr/local/share/` (or similar) directories. If your system is equipt
with CUDA, OpenCL, and CPU support, the find script will automatically choose
the most powerful backend (typically CUDA).

To use ArrayFire simply use the `Find` script as follows:

    FIND_PACKAGE(ArrayFire)
    INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
    ...

    ADD_EXECUTABLE(some_executable ...)
    TARGET_LINK_LIBRARIES(some_executable ${ArrayFire_LIBRARIES} )

If you wish to target a specific backend, switch `${ArrayFire_LIBRARIES}` to
`${ArrayFire_CPU}` `${ArrayFire_OPENCL}` or `${ArrayFire_CUDA}` in the
`TARGET_LINK_LIBRARIES` command above.

## MakeFiles

Using ArrayFire with Makefiles is almost as easy as CMake, but you will
need to specify paths manually. In your makefile specify the include path to
the directory containing `arrayfire.h`. Typically this will be `-I /usr/include`
or `-I /usr/local/include` if you installed ArrayFire using our installation
instructions.
Then, in your linker line specify the path to ArrayFire using the `-L` option
(typically `-L/usr/lib` or `-L/usr/local/lib` and the specific ArrayFire backend
you wish to use with the `-l` option (i.e. `-lafcpu`, `-lafopencl` or `-lafcuda`
for the CPU, OpenCL and CUDA backends repsectively).

Here is a minimial example MakeFile which uses ArrayFire's CPU backend:

    LIBS=-lafcpu
    LIB_PATHS=/usr/lib
    INCLUDES=-I/usr/include
    CC=g++ $(COMPILER_OPTIONS)
    COMPILER_OPTIONS=-std=c++11 -g

    all: main.cpp Makefile
	    $(CC) main.cpp -o test $(INCLUDES) $(LIBS) $(LIB_PATHS)
