/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <memory.hpp>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>

// Below Class definition is found at the following URL
// http://stackoverflow.com/questions/9007343/mix-custom-memory-managment-and-thrust-in-cuda

namespace arrayfire {
namespace cuda {

template<typename T>
struct ThrustAllocator : thrust::device_malloc_allocator<T> {
    // shorthand for the name of the base class
    typedef thrust::device_malloc_allocator<T> super_t;

    // get access to some of the base class's typedefs
    // note that because we inherited from device_malloc_allocator,
    // pointer is actually thrust::device_ptr<T>
    typedef typename super_t::pointer pointer;

    typedef typename super_t::size_type size_type;

    pointer allocate(size_type elements) {
        return thrust::device_ptr<T>(
            memAlloc<T>(elements)
                .release());  // delegate to ArrayFire allocator
    }

    void deallocate(pointer p, size_type n) {
        UNUSED(n);
        memFree<T>(p.get());  // delegate to ArrayFire allocator
    }
};
}  // namespace cuda
}  // namespace arrayfire
