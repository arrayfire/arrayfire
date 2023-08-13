/*******************************************************
 * Copyright (c) 2023, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// ![interop_sycl_external_context_snippet]
#include <arrayfire.h>
// 1. Add the af/sycl.h include to your project
#include <af/sycl.h>

#include <cassert>

// 1. Add arrayfire.h and af/sycl.h to your application
#include "af/sycl.h"
#include "arrayfire.h"

#include <cstdio>
#include <vector>

using std::vector;

int main() try {
    // 1. Create a sycl::queue
    sycl::queue q;

    // 2. Create a sycl::buffer of size 10 filled with ones
    constexpr int length = 10;
    sycl::buffer<float> sycl_A{sycl::range<1>(length)};

    q.submit([&](sycl::handler &h) {
        auto acc = sycl_A.get_access(h);
        h.parallel_for(sycl::range<1>(10),
                       [=](sycl::id<1> idx) { acc[idx] = 1.0f; });
    });

    // 3. Instruct sycl to complete its outstanding operations using
    // queue.wait() not strictly required as sycl::buffer will take care of data
    // access dependencies
    q.wait();

    // 4. Create ArrayFire arrays from sycl::buffer objects
    af::array af_A = afsycl::array(length, sycl_A);

    // 5. Perform ArrayFire operations on the Arrays
    af_A = af_A + af::randu(length);

    // 6. Instruct ArrayFire to finish operations using af::sync
    af::sync();

    // 7. Obtain sycl::buffer references for important memory
    sycl_A = *static_cast<sycl::buffer<float> *>(af_A.device<void>());

    // 8. Continue your sycl application
    // ...
    return EXIT_SUCCESS;
}
// ![interop_sycl_external_context_snippet]
catch (sycl::exception &e) {
    std::cout << e.what() << std::endl;
}
