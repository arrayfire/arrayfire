/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <interopManager.hpp>
#include <Array.hpp>
#include <hist_graphics.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>

namespace opencl
{

template<typename T>
void copy_histogram(const Array<T> &data, const fg::Histogram* hist)
{
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_P = data.get();
        size_t bytes = hist->size();

        InteropManager& intrpMngr = InteropManager::getInstance();

        cl::Buffer *clPBOResource = intrpMngr.getBufferResource(hist);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*clPBOResource);

        glFinish();
        getQueue().enqueueAcquireGLObjects(&shared_objects);
        getQueue().enqueueCopyBuffer(*d_P, *clPBOResource, 0, 0, bytes, NULL, NULL);
        getQueue().finish();
        getQueue().enqueueReleaseGLObjects(&shared_objects);

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, hist->vbo());
        GLubyte* ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (ptr) {
            getQueue().enqueueReadBuffer(*data.get(), CL_TRUE, 0, hist->size(), ptr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T)  \
    template void copy_histogram<T>(const Array<T> &data, const fg::Histogram* hist);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
