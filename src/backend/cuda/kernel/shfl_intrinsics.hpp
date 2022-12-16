/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

namespace arrayfire {
namespace cuda {
namespace kernel {

//__all_sync wrapper
template<typename T>
__device__ T all_sync(unsigned mask, T var) {
#if (CUDA_VERSION >= 9000)
    return __all_sync(mask, var);
#else
    return __all(var);
#endif
}

//__all_sync wrapper
template<typename T>
__device__ T any_sync(unsigned mask, T var) {
#if (CUDA_VERSION >= 9000)
    return __any_sync(mask, var);
#else
    return __any(var);
#endif
}

//__shfl_down_sync wrapper
template<typename T>
__device__ T ballot_sync(unsigned mask, T var) {
#if (CUDA_VERSION >= 9000)
    return __ballot_sync(mask, var);
#else
    return __ballot(var);
#endif
}

//__shfl_down_sync wrapper
template<typename T>
__device__ T shfl_down_sync(unsigned mask, T var, int delta) {
#if (CUDA_VERSION >= 9000)
    return __shfl_down_sync(mask, var, delta);
#else
    return __shfl_down(var, delta);
#endif
}
// specialization for cfloat
template<>
inline __device__ cfloat shfl_down_sync(unsigned mask, cfloat var, int delta) {
#if (CUDA_VERSION >= 9000)
    cfloat res = {__shfl_down_sync(mask, var.x, delta),
                  __shfl_down_sync(mask, var.y, delta)};
#else
    cfloat res  = {__shfl_down(var.x, delta), __shfl_down(var.y, delta)};
#endif
    return res;
}
// specialization for cdouble
template<>
inline __device__ cdouble shfl_down_sync(unsigned mask, cdouble var,
                                         int delta) {
#if (CUDA_VERSION >= 9000)
    cdouble res = {__shfl_down_sync(mask, var.x, delta),
                   __shfl_down_sync(mask, var.y, delta)};
#else
    cdouble res = {__shfl_down(var.x, delta), __shfl_down(var.y, delta)};
#endif
    return res;
}

//__shfl_up_sync wrapper
template<typename T>
__device__ T shfl_up_sync(unsigned mask, T var, int delta) {
#if (CUDA_VERSION >= 9000)
    return __shfl_up_sync(mask, var, delta);
#else
    return __shfl_up(var, delta);
#endif
}
// specialization for cfloat
template<>
inline __device__ cfloat shfl_up_sync(unsigned mask, cfloat var, int delta) {
#if (CUDA_VERSION >= 9000)
    cfloat res = {__shfl_up_sync(mask, var.x, delta),
                  __shfl_up_sync(mask, var.y, delta)};
#else
    cfloat res  = {__shfl_up(var.x, delta), __shfl_up(var.y, delta)};
#endif
    return res;
}
// specialization for cdouble
template<>
inline __device__ cdouble shfl_up_sync(unsigned mask, cdouble var, int delta) {
#if (CUDA_VERSION >= 9000)
    cdouble res = {__shfl_up_sync(mask, var.x, delta),
                   __shfl_up_sync(mask, var.y, delta)};
#else
    cdouble res = {__shfl_up(var.x, delta), __shfl_up(var.y, delta)};
#endif
    return res;
}

}  // namespace kernel
}  // namespace cuda
}  // namespace arrayfire
