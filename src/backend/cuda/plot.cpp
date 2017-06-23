/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <Array.hpp>
#include <plot.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <reorder.hpp>
#include <GraphicsResourceManager.hpp>

using af::dim4;

namespace cuda
{
using namespace gl;

template<typename T>
void copy_plot(const Array<T> &P, forge::Plot* plot)
{
    if(DeviceManager::checkGraphicsInteropCapability()) {
        const T *d_P = P.get();

        ShrdResVector res = interopManager().getBufferResource(plot);

        // Map resource. Copy data to VBO. Unmap resource.
        size_t num_bytes = plot->verticesSize();
        T* d_vbo = NULL;
        cudaGraphicsMapResources(1, res[0].get(), cuda::getActiveStream());
        cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, *(res[0].get()));
        cudaMemcpyAsync(d_vbo, d_P, num_bytes, cudaMemcpyDeviceToDevice, cuda::getActiveStream());
        cudaGraphicsUnmapResources(1, res[0].get(), cuda::getActiveStream());

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        CheckGL("Begin CUDA fallback-resource copy");
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, plot->vertices());
        gl::GLubyte* ptr = (gl::GLubyte*)glMapBuffer((gl::GLenum)GL_ARRAY_BUFFER, (gl::GLenum)GL_WRITE_ONLY);
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
