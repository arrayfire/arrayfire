// FIXME: Add a special flag for debug
#ifndef NDEBUG
#include <iostream>
#include <stdio.h>
#include <err_cuda.hpp>


#define POST_LAUNCH_CHECK() do {                \
        CUDA_CHECK(cudaDeviceSynchronize());    \
    } while(0)                                  \

#else

#define POST_LAUNCH_CHECK() do {                \
        CUDA_CHECK(cudaPeekAtLastError());      \
    } while(0)                                  \

#endif
