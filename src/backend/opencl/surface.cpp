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
#include <surface.hpp>
#include <err_opencl.hpp>
#include <debug_opencl.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <reorder.hpp>

using af::dim4;

namespace opencl
{

template<typename T>
void copy_surface(const Array<T> &P, fg::Surface* surface)
{
    if (isGLSharingSupported()) {
        CheckGL("Begin OpenCL resource copy");
        const cl::Buffer *d_P = P.get();
        size_t bytes = surface->size();

        InteropManager& intrpMngr = InteropManager::getInstance();

        cl::Buffer *clPBOResource = intrpMngr.getBufferResource(surface);

        std::vector<cl::Memory> shared_objects;
        shared_objects.push_back(*clPBOResource);

        glFinish();

        // Use of events:
        // https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReleaseGLObjects.html
        cl::Event event;

        getQueue().enqueueAcquireGLObjects(&shared_objects, NULL, &event);
        event.wait();
        getQueue().enqueueCopyBuffer(*d_P, *clPBOResource, 0, 0, bytes, NULL, &event);
        getQueue().enqueueReleaseGLObjects(&shared_objects, NULL, &event);
        event.wait();

        CL_DEBUG_FINISH(getQueue());
        CheckGL("End OpenCL resource copy");
    } else {
        CheckGL("Begin OpenCL fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, surface->vbo());
        GLubyte* ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (ptr) {
            getQueue().enqueueReadBuffer(*P.get(), CL_TRUE, 0, surface->size(), ptr);
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End OpenCL fallback-resource copy");
    }
}

#define INSTANTIATE(T)  \
    template void copy_surface<T>(const Array<T> &P, fg::Surface* surface);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
