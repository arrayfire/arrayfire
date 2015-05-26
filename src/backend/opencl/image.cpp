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
void copy_image(const Array<T> &in, const fg::Image* image)
{
    if (isGLSharingSupported()) {
        CheckGL("Begin opencl resource copy");
        InteropManager& intrpMngr = InteropManager::getInstance();

        cl::Buffer *clPBOResource = intrpMngr.getBufferResource(image);
        const cl::Buffer *d_X = in.get();
        size_t num_bytes = image->size();

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*clPBOResource);

        glFinish();
        getQueue().enqueueAcquireGLObjects(&shared_objects);
        getQueue().enqueueCopyBuffer(*d_X, *clPBOResource, 0, 0, num_bytes, NULL, NULL);
        getQueue().finish();
        getQueue().enqueueReleaseGLObjects(&shared_objects);

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End opencl resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image->pbo());
        CheckGL("1Begin OpenCL fallback-resource copy");
        glBufferData(GL_PIXEL_UNPACK_BUFFER, image->size(), 0, GL_STREAM_DRAW);
        CheckGL("2Begin OpenCL fallback-resource copy");
        GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
        CheckGL("3Begin OpenCL fallback-resource copy");
        if (ptr) {
            getQueue().enqueueReadBuffer(*in.get(), CL_TRUE, 0, image->size(), ptr);
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        CheckGL("4Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T)      \
    template void copy_image<T>(const Array<T> &in, const fg::Image* image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)

}

#endif
