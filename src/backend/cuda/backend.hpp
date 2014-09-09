#pragma once
#ifdef __DH__
#undef __DH__
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define __DH__ __device__ __host__
#else
#define __DH__
#endif

#include "types.hpp"

namespace detail = cuda;
