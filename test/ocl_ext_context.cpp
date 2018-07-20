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

using std::endl;
using std::vector;
using af::array;
using af::constant;
using af::getDeviceCount;
using af::info;
using af::randu;
using af::setDevice;

inline void checkErr(cl_int err, const char * name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name  << " (" << err << ")" << endl;
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

        #ifdef CL_VERSION_2_0
        qId = clCreateCommandQueueWithProperties(cId, dId, 0, &errorCode);
        #else
        qId = clCreateCommandQueue(cId, dId, 0, &errorCode);
        #endif


        checkErr(errorCode, "Command queue creation failed");
        call_once = false;
    }
    deviceId = dId;
    context  = cId;
    queue    = qId;
}

TEST(OCLExtContext, PushAndPop)
{
    cl_device_id deviceId = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    getExternals(deviceId, context, queue);
    int dCount = getDeviceCount();
    printf("\n%d devices before afcl::addDevice\n\n", dCount);
    info();

    afcl::addDevice(deviceId, context, queue);
    ASSERT_EQ(true, dCount+1==getDeviceCount());
    printf("\n%d devices after afcl::addDevice\n", getDeviceCount());

    afcl::deleteDevice(deviceId, context);
    ASSERT_EQ(true, dCount==getDeviceCount());
    printf("\n%d devices after afcl::deleteDevice\n\n", getDeviceCount());
    info();
}

TEST(OCLExtContext, set)
{
    cl_device_id deviceId = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;

    int dCount = getDeviceCount(); //Before user device addition
    setDevice(0);
    info();
    array t = randu(5,5);
    af_print(t);

    getExternals(deviceId, context, queue);
    afcl::addDevice(deviceId, context, queue);
    printf("\nBefore setting device to newly added one\n\n");
    info();

    printf("\n\nBefore setting device to newly added one\n\n");
    setDevice(dCount); //In 0-based index, dCount is index of newly added device
    info();

    const int x = 5;
    const int y = 5;
    const int s = x * y;
    array a = constant(1, x, y);
    vector<float> host(s);
    a.host((void*)host.data());
    for (int i=0; i<s; ++i)
        ASSERT_EQ(host[i], 1.0f);

    printf("\n\nAfter reset to default set of devices\n\n");
    setDevice(0);
    info();
    af_print(t);
}

TEST(OCLCheck, DeviceType)
{
    afcl::deviceType devType = afcl::getDeviceType();
    cl_device_type type = -100;
    clGetDeviceInfo(afcl::getDeviceId(),
                    CL_DEVICE_TYPE,
                    sizeof(cl_device_type),
                    &type,
                    NULL);
    ASSERT_EQ(type, (cl_device_type)devType);
}

TEST(OCLCheck, DevicePlatform)
{
    afcl::platform platform = afcl::getPlatform();
    ASSERT_NE(platform, AFCL_PLATFORM_UNKNOWN);
}
#else
TEST(OCLExtContext, NoopCPU)
{
}
#endif
