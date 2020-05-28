/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#if defined(AF_OPENCL)
#include <af/opencl.h>
#include <iostream>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if __GNUC__ >= 8
#pragma GCC diagnostic ignored "-Wcatch-value="
#endif
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS 1
#include <CL/cl2.hpp>
#pragma GCC diagnostic pop

using af::allocV2;
using af::array;
using af::constant;
using af::freeV2;
using af::getDeviceCount;
using af::info;
using af::randu;
using af::setDevice;
using std::endl;
using std::vector;

inline void checkErr(cl_int err, const char *name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << endl;
        exit(EXIT_FAILURE);
    }
}

class OCLExtContext : public ::testing::Test {
   public:
    cl_device_id deviceId  = NULL;
    cl_context context     = NULL;
    cl_command_queue queue = NULL;

    void SetUp() override {
        cl_platform_id platformId = NULL;
        cl_uint numPlatforms;
        cl_uint numDevices;
        cl_int errorCode = 0;

        checkErr(clGetPlatformIDs(1, &platformId, &numPlatforms),
                 "Get Platforms failed");

        checkErr(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1,
                                &deviceId, &numDevices),
                 "Get cl_device_id failed");

        context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &errorCode);
        checkErr(errorCode, "Context creation failed");

#ifdef CL_VERSION_2_0
        queue = clCreateCommandQueueWithProperties(context, deviceId, 0,
                                                   &errorCode);
#else
        queue = clCreateCommandQueue(context, deviceId, 0, &errorCode);
#endif

        checkErr(errorCode, "Command queue creation failed");
    }
    void TearDown() override {
        checkErr(clReleaseCommandQueue(queue), "clReleaseCommandQueue");
        checkErr(clReleaseContext(context), "clReleaseContext");
        checkErr(clReleaseDevice(deviceId), "clReleaseDevice");
    }
};

TEST_F(OCLExtContext, PushAndPop) {
    int dCount = getDeviceCount();
    info();

    afcl::addDevice(deviceId, context, queue);
    ASSERT_EQ(true, dCount + 1 == getDeviceCount());

    afcl::deleteDevice(deviceId, context);
    ASSERT_EQ(true, dCount == getDeviceCount());
    info();
}

TEST_F(OCLExtContext, set) {
    int dCount = getDeviceCount();  // Before user device addition
    setDevice(0);
    info();
    array t = randu(5, 5);
    af_print(t);

    afcl::addDevice(deviceId, context, queue);
    info();

    setDevice(
        dCount);  // In 0-based index, dCount is index of newly added device
    info();

    const int x = 5;
    const int y = 5;
    const int s = x * y;
    array a     = constant(1, x, y);
    vector<float> host(s);
    a.host((void *)host.data());
    for (int i = 0; i < s; ++i) ASSERT_EQ(host[i], 1.0f);

    setDevice(0);
    info();
    af_print(t);
}

TEST(OCLCheck, DeviceType) {
    afcl::deviceType devType = afcl::getDeviceType();
    cl_device_type type      = -100;
    clGetDeviceInfo(afcl::getDeviceId(), CL_DEVICE_TYPE, sizeof(cl_device_type),
                    &type, NULL);
    ASSERT_EQ(type, (cl_device_type)devType);
}

TEST(OCLCheck, DevicePlatform) {
    afcl::platform platform = afcl::getPlatform();
    ASSERT_NE(platform, AFCL_PLATFORM_UNKNOWN);
}
#else
TEST(OCLExtContext, NoopCPU) {}
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
TEST(Memory, AfAllocDeviceOpenCL) {
    /// Tests to see if the pointer returned can be used by opencl functions
    float gold_val = 5;

    void *alloc_ptr;
    ASSERT_SUCCESS(af_alloc_device(&alloc_ptr, sizeof(float)));
    // af_alloc_device returns a cl::Buffer object from alloc unfortunately
    cl::Buffer *bptr = static_cast<cl::Buffer *>(alloc_ptr);
    ASSERT_EQ(2, bptr->getInfo<CL_MEM_REFERENCE_COUNT>());

    cl_command_queue queue;
    afcl_get_queue(&queue, true);
    cl::CommandQueue cq(queue);

    cl::Buffer gold(cq, &gold_val, &gold_val + 1, false);
    cq.enqueueCopyBuffer(gold, *bptr, 0, 0, sizeof(float));

    float host;
    cq.enqueueReadBuffer(*bptr, CL_TRUE, 0, sizeof(float), &host);

    ASSERT_SUCCESS(af_free_device(alloc_ptr));
    ASSERT_EQ(gold_val, host);
}
#pragma GCC diagnostic pop

TEST(Memory, AfAllocDeviceV2OpenCLC) {
    /// Tests to see if the pointer returned can be used by opencl functions
    float gold_val = 5;

    void *alloc_ptr;
    ASSERT_SUCCESS(af_alloc_device_v2(&alloc_ptr, sizeof(float)));
    {
        cl::Buffer bptr(static_cast<cl_mem>(alloc_ptr), true);
        ASSERT_EQ(3, bptr.getInfo<CL_MEM_REFERENCE_COUNT>());

        cl_command_queue queue;
        afcl_get_queue(&queue, true);
        cl::CommandQueue cq(queue);

        cl::Buffer gold(cq, &gold_val, &gold_val + 1, false);
        cq.enqueueCopyBuffer(gold, bptr, 0, 0, sizeof(float));

        float host;
        cq.enqueueReadBuffer(bptr, CL_TRUE, 0, sizeof(float), &host);
        ASSERT_EQ(gold_val, host);
    }

    ASSERT_SUCCESS(af_free_device_v2(alloc_ptr));
}

TEST(Memory, AfAllocDeviceV2OpenCLCPP) {
    /// Tests to see if the pointer returned can be used by opencl functions
    float gold_val = 5;

    cl_mem alloc_ptr = static_cast<cl_mem>(allocV2(sizeof(float)));
    {
        cl::Buffer bptr(alloc_ptr, true);
        ASSERT_EQ(3, bptr.getInfo<CL_MEM_REFERENCE_COUNT>());

        cl_command_queue queue;
        afcl_get_queue(&queue, true);
        cl::CommandQueue cq(queue);

        cl::Buffer gold(cq, &gold_val, &gold_val + 1, false);
        cq.enqueueCopyBuffer(gold, bptr, 0, 0, sizeof(float));

        float host;
        cq.enqueueReadBuffer(bptr, CL_TRUE, 0, sizeof(float), &host);
        ASSERT_EQ(gold_val, host);
    }

    freeV2(alloc_ptr);
}

TEST(Memory, SNIPPET_AllocOpenCL) {
    // clang-format off
    //! [ex_alloc_v2_opencl]
    cl_command_queue queue;
    afcl_get_queue(&queue, true);
    cl_context context;
    afcl_get_context(&context, true);

    void *alloc_ptr = allocV2(sizeof(float));
    cl_mem mem = static_cast<cl_mem>(alloc_ptr);

    // Map memory from the device to the System memory
    cl_int map_err_code;
    void *mapped_ptr = clEnqueueMapBuffer(
        queue, // command queueu
        mem, // buffer
        CL_TRUE, // is blocking
        CL_MAP_READ | CL_MAP_WRITE, // map type
        0, // offset
        sizeof(float), // size
        0, // num_events_in_wait_list
        nullptr, // event_wait_list
        nullptr, // event
        &map_err_code); // error code

    float *float_ptr = static_cast<float *>(mapped_ptr);
    float_ptr[0]     = 5.0f;

    // Unmap buffer after we are done using it
    cl_int unmap_err_code =
        clEnqueueUnmapMemObject(queue,      // command queue
                                mem,        // buffer
                                mapped_ptr, // mapped pointer
                                0,          // num_events_in_wait_list
                                nullptr,    // event_wait_list
                                nullptr);   // event
    freeV2(alloc_ptr);
    //! [ex_alloc_v2_opencl]
    // clang-format on

    ASSERT_EQ(CL_SUCCESS, map_err_code);
    ASSERT_EQ(CL_SUCCESS, unmap_err_code);
}
