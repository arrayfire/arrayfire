/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>
#include <arrayfire.h>
#if defined(AF_OPENCL)
#include <af/opencl.h>
#include <iostream>

using namespace std;

inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void getExternals(cl_device_id &deviceId, cl_context &context, cl_command_queue &queue)
{
    static cl_device_id dId = NULL;
    static cl_context cId = NULL;
    static cl_command_queue qId = NULL;
    static bool call_once = true;

    if (call_once) {
        cl_platform_id platformId = NULL;
        cl_uint numPlatforms;
        cl_uint numDevices;
        cl_int errorCode = 0;

        checkErr(clGetPlatformIDs(1, &platformId, &numPlatforms),
                "Get Platforms failed");

        checkErr(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &dId, &numDevices),
                "Get cl_device_id failed");

        cId = clCreateContext(NULL, 1, &dId, NULL, NULL, &errorCode);
        checkErr(errorCode, "Context creation failed");

        qId = clCreateCommandQueue(cId, dId, 0, &errorCode);
        checkErr(errorCode, "Command queue creation failed");
        call_once = false;
    }
    deviceId = dId;
    context  = cId;
    queue    = qId;
}

TEST(OCLExtContext, push)
{
    cl_device_id deviceId = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    getExternals(deviceId, context, queue);
    int dCount = af::getDeviceCount();
    printf("%d devices before afcl::pushDevice\n", dCount);
    af::info();
    afcl::pushDevice(deviceId, context, queue);
    ASSERT_EQ(true, dCount+1==af::getDeviceCount());
    printf("%d devices after afcl::pushDevice\n", af::getDeviceCount());
    af::info();
}

TEST(OCLExtContext, set)
{
    cl_device_id deviceId = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    getExternals(deviceId, context, queue);
    afcl::setDevice(deviceId, context);
    af::info();

    const int x = 5;
    const int y = 5;
    const int s = x * y;
    af::array a = af::constant(1, x, y);
    vector<float> host(s);
    a.host((void*)host.data());
    for (int i=0; i<s; ++i)
        ASSERT_EQ(host[i], 1.0f);
}

TEST(OCLExtContext, pop)
{
    cl_device_id deviceId = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    getExternals(deviceId, context, queue);
    int dCount = af::getDeviceCount();
    printf("%d devices before afcl::popDevice\n", dCount);
    af::setDevice(0);
    af::info();
    afcl::popDevice(deviceId, context);
    ASSERT_EQ(true, dCount-1==af::getDeviceCount());
    printf("%d devices after afcl::popDevice\n", af::getDeviceCount());
    af::info();
}
#else
TEST(OCLExtContext, NoopCPU)
{
}
#endif
