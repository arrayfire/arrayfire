/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_LINEAR_ALGEBRA)

#include <cusolverDnManager.hpp>
#include <platform.hpp>

#include <stdexcept>
#include <string>
#include <iostream>
#include <boost/scoped_ptr.hpp>

namespace cuda
{

cusolverDnHandle::cusolverDnHandle() : handle(0)
{
    cusolverStatus_t cErr;
    cErr = cusolverDnCreate(&handle);
    if(cErr != CUSOLVER_STATUS_SUCCESS) {
        using std::string;
        throw std::runtime_error(string("cusolverDn Error: ") + cusolverErrorString(cErr));
    }
}

cusolverDnHandle::~cusolverDnHandle()
{
    cusolverDnDestroy(handle);
}

cusolverDnHandle::operator cusolverDnHandle_t()
{
    return handle;
}

cusolverDnHandle&
getSolverHandle()
{
    using boost::scoped_ptr;
    static scoped_ptr<cusolverDnHandle> handle[DeviceManager::MAX_DEVICES];
    if(!handle[getActiveDeviceId()]) {
        handle[getActiveDeviceId()].reset(new cusolverDnHandle());
    }

    return *handle[getActiveDeviceId()];
}

}

#endif
