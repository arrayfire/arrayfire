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
#include <plot.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <reorder.hpp>

using af::dim4;

namespace cuda
{
using namespace gl;

template<typename T>
void copy_plot(const Array<T> &P, forge::Plot* plot)
{
    if(InteropManager::checkGraphicsInteropCapability()) {
        const T *d_P = P.get();

        InteropManager& intrpMngr = InteropManager::getInstance();

        cudaGraphicsResource *cudaVBOResource = intrpMngr.getBufferResource(plot);
        // Map resource. Copy data to VBO. Unmap resource.
        size_t num_bytes = plot->verticesSize();
        T* d_vbo = NULL;
        cudaGraphicsMapResources(1, &cudaVBOResource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, cudaVBOResource);
        cudaMemcpyAsync(d_vbo, d_P, num_bytes, cudaMemcpyDeviceToDevice,
                cuda::getStream(cuda::getActiveDeviceId()));
        cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        CheckGL("Begin CUDA fallback-resource copy");
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, plot->vertices());
        GLubyte* ptr = (GLubyte*)glMapBuffer((gl::GLenum)GL_ARRAY_BUFFER, (gl::GLenum)GL_WRITE_ONLY);
        if (ptr) {
            CUDA_CHECK(cudaMemcpy(ptr, P.get(), plot->verticesSize(), cudaMemcpyDeviceToHost));
            glUnmapBuffer((gl::GLenum)GL_ARRAY_BUFFER);
        }
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T)  \
    template void copy_plot<T>(const Array<T> &P, forge::Plot* plot);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
