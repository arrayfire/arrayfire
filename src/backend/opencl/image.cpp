/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <interopManager.hpp>
#include <Array.hpp>
#include <image.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <stdexcept>
#include <vector>

namespace opencl
{

template<typename T>
void copy_image(const Array<T> &in, const fg_image_handle image)
{
    InteropManager& intrpMngr = InteropManager::getInstance();

    cl::Buffer *clPBOResource = intrpMngr.getPBOResource(image);
    const cl::Buffer *d_X = in.get();
    size_t num_bytes = in.elements()*sizeof(T);

    std::vector<cl::Memory> shared_objects;
    shared_objects.push_back(*clPBOResource);

    glFinish();
    getQueue().enqueueAcquireGLObjects(&shared_objects);
    getQueue().enqueueCopyBuffer(*d_X, *clPBOResource, 0, 0, num_bytes, NULL, NULL);
    getQueue().finish();
    getQueue().enqueueReleaseGLObjects(&shared_objects);

    CheckGL("After opencl resource copy");
    CL_DEBUG_FINISH(getQueue());
}

#define INSTANTIATE(T)      \
    template void copy_image<T>(const Array<T> &in, const fg_image_handle image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)

}

#endif
