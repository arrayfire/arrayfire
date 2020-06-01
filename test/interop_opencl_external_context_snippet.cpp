/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wcatch-value="
#endif
// ![interop_opencl_external_context_snippet]
#include <arrayfire.h>
// 1. Add the af/opencl.h include to your project
#include <af/opencl.h>

#include <cassert>

// definitions required by cl2.hpp
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// 1. Add arrayfire.h and af/opencl.h to your application
#include "af/opencl.h"
#include "arrayfire.h"

#include <cstdio>
#include <vector>

using std::vector;

int main() {
    // 1. Set up the OpenCL context, device, and queues
    cl::Context context;
    try {
        context = cl::Context(CL_DEVICE_TYPE_ALL);
    } catch (const cl::Error& err) {
        fprintf(stderr, "Exiting creating context");
        return EXIT_FAILURE;
    }
    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        fprintf(stderr, "Exiting. No devices found");
        return EXIT_SUCCESS;
    }
    cl::Device device = devices[0];
    cl::CommandQueue queue(context, device);

    // Create a buffer of size 10 filled with ones, copy it to the device
    int length = 10;
    vector<float> h_A(length, 1);
    cl::Buffer cl_A(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    length * sizeof(float), h_A.data());

    // 2. Instruct OpenCL to complete its operations using clFinish (or similar)
    queue.finish();

    // 3. Instruct ArrayFire to use the user-created context
    //    First, create a device from the current OpenCL device + context +
    //    queue
    afcl::addDevice(device(), context(), queue());
    //    Next switch ArrayFire to the device using the device and context as
    //    identifiers:
    afcl::setDevice(device(), context());

    // 4. Create ArrayFire arrays from OpenCL memory objects
    af::array af_A = afcl::array(length, cl_A(), f32, true);
    clRetainMemObject(cl_A());

    // 5. Perform ArrayFire operations on the Arrays
    af_A = af_A + af::randu(length);

    // NOTE: ArrayFire does not perform the above transaction using in-place
    // memory, thus the underlying OpenCL buffers containing the memory
    // containing memory to probably have changed

    // 6. Instruct ArrayFire to finish operations using af::sync
    af::sync();

    // 7. Obtain cl_mem references for important memory
    cl_mem* af_mem = af_A.device<cl_mem>();
    cl_A           = cl::Buffer(*af_mem, /*retain*/ true);

    /// Delete the af_mem pointer. The buffer returned by the device pointer is
    /// still valid
    delete af_mem;

    // 8. Continue your OpenCL application

    // ...
    return EXIT_SUCCESS;
}
// ![interop_opencl_external_context_snippet]

#pragma GCC diagnostic pop
