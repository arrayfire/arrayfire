/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#ifdef __CUDACC_RTC__

namespace cuda {
template<typename T>
struct SharedMemory {
    __DH__ T* getPointer() {
        extern __shared__ T ptr[];
        return ptr;
    }
};
}

#else

namespace cuda {
namespace kernel {

template<typename T>
struct SharedMemory {
    // return a pointer to the runtime-sized shared memory array.
    __device__ T* getPointer() {
        extern __device__ void
        Error_UnsupportedType();  // Ensure that we won't compile any
                                  // un-specialized types
        Error_UnsupportedType();
        return (T*)0;
    }
};

#define SPECIALIZE(T)                         \
    template<>                                \
    struct SharedMemory<T> {                  \
        __device__ T* getPointer() {          \
            extern __shared__ T ptr_##T##_[]; \
            return ptr_##T##_;                \
        }                                     \
    };

SPECIALIZE(float)
SPECIALIZE(cfloat)
SPECIALIZE(double)
SPECIALIZE(cdouble)
SPECIALIZE(char)
SPECIALIZE(int)
SPECIALIZE(uint)
SPECIALIZE(short)
SPECIALIZE(ushort)
SPECIALIZE(uchar)
SPECIALIZE(intl)
SPECIALIZE(uintl)

#undef SPECIALIZE

}  // namespace kernel
}  // namespace cuda

#endif
