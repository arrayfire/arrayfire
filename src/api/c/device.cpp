/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <af/device.h>
#include <af/version.h>
#include <af/backend.h>
#include <backend.hpp>
#include <platform.hpp>
#include <Array.hpp>
#include <handle.hpp>
#include "err_common.hpp"
#include <cstring>

using namespace detail;

af_err af_set_backend(const af_backend bknd)
{
    try {
        ARG_ASSERT(0, bknd==getBackend());
    }
    CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_backend_count(unsigned* num_backends)
{
    *num_backends = 1;
    return AF_SUCCESS;
}

af_err af_get_available_backends(int* result)
{
    try {
        *result = getBackend();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_backend_id(af_backend *result, const af_array in)
{
    try {
        ARG_ASSERT(1, in != 0);
        ArrayInfo info = getInfo(in);
        *result = info.getBackendId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_active_backend(af_backend *result)
{
    *result = (af_backend)getBackend();
    return AF_SUCCESS;
}

af_err af_init()
{
    try {
        static bool first = true;
        if(first) {
            getInfo();
            first = false;
        }
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_info()
{
    try {
        printf("%s", getInfo().c_str());
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_info_string(char **str, const bool verbose)
{
    try {
        std::string infoStr = getInfo();
        af_alloc_host((void**)str, sizeof(char) * (infoStr.size() + 1));

        // Need to do a deep copy
        // str.c_str wont cut it
        infoStr.copy(*str, infoStr.size());
        (*str)[infoStr.size()] = '\0';
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_version(int *major, int *minor, int *patch)
{
    *major = AF_VERSION_MAJOR;
    *minor = AF_VERSION_MINOR;
    *patch = AF_VERSION_PATCH;

    return AF_SUCCESS;
}

af_err af_device_info(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
{
    try {
        devprop(d_name, d_platform, d_toolkit, d_compute);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_dbl_support(bool* available, const int device)
{
    try {
        *available = isDoubleSupported(device);
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_get_device_count(int *nDevices)
{
    try {
        *nDevices = getDeviceCount();
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_get_device(int *device)
{
    try {
        *device = getActiveDeviceId();
    } CATCHALL;
    return AF_SUCCESS;
}

af_err af_set_device(const int device)
{
    try {
        ARG_ASSERT(0, device >= 0);
        ARG_ASSERT(0, setDevice(device) >= 0);
    } CATCHALL;

    return AF_SUCCESS;
}

af_err af_sync(const int device)
{
    try {
        int dev = device == -1 ? getActiveDeviceId() : device;
        detail::sync(dev);
    } CATCHALL;
    return AF_SUCCESS;
}
