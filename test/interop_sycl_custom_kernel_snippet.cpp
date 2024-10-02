/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// clang-format off
// ![interop_sycl_custom_kernel_snippet]

// 1. Add arrayfire.h and af/oneapi.h to your application
#include <arrayfire.h>
#include <af/oneapi.h>

#include <cassert>

int main() try {
    size_t length = 10;

    // Create ArrayFire array objects:
    af::array A = af::randu(length, f32);
    af::array B = af::constant(0, length, f32);

    // ... additional ArrayFire operations here

    // 2. Obtain the queue used by ArrayFire
    static sycl::queue af_queue = afoneapi::getQueue();

    // 3. Obtain sycl::buffer references to af::array objects
    sycl::buffer<float> d_A = *static_cast<sycl::buffer<float> *>(A.device<void>());
    sycl::buffer<float> d_B = *static_cast<sycl::buffer<float> *>(B.device<void>());

    // 4. Prepare, and use your sycl kernels.
    af_queue.submit([&](sycl::handler &h) {
        auto accA = d_A.get_access(h);
        auto accB = d_B.get_access(h);
        h.parallel_for(sycl::range<1>(10), [=](sycl::id<1> idx) {
            accB[idx] = accA[idx];
        });
    });

    // 5. Return control of af::array memory to ArrayFire
    A.unlock();
    B.unlock();

    // A and B should be the same because of the sycl kernel user code
    assert(af::allTrue<bool>(A == B));

    // ... resume ArrayFire operations
    return 0;
}
// ![interop_sycl_custom_kernel_snippet]
catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
}
// clang-format on
