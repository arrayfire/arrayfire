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
#include <hist_graphics.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <GraphicsResourceManager.hpp>

namespace cuda
{
using namespace gl;

template<typename T>
void copy_histogram(const Array<T> &data, const forge::Histogram* hist)
{
    if(DeviceManager::checkGraphicsInteropCapability()) {
        const T *d_P = data.get();

        ShrdResVector res = interopManager().getBufferResource(hist);

        // Map resource. Copy data to VBO. Unmap resource.
        size_t num_bytes = hist->verticesSize();
        T* d_vbo = NULL;
        cudaGraphicsMapResources(1, res[0].get(), cuda::getActiveStream());
        cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, *(res[0].get()));
        cudaMemcpyAsync(d_vbo, d_P, num_bytes, cudaMemcpyDeviceToDevice, cuda::getActiveStream());
        cudaGraphicsUnmapResources(1, res[0].get(), cuda::getActiveStream());

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        CheckGL("Begin CUDA fallback-resource copy");
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, hist->vertices());
        gl::GLubyte* ptr = (gl::GLubyte*)glMapBuffer((gl::GLenum)GL_ARRAY_BUFFER, (gl::GLenum)GL_WRITE_ONLY);
        if (ptr) {
            auto stream = cuda::getActiveStream();
            CUDA_CHECK(cudaMemcpyAsync(ptr, data.get(), hist->verticesSize(), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer((gl::GLenum)GL_ARRAY_BUFFER);
        }
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T)  \
    template void copy_histogram<T>(const Array<T> &data, const forge::Histogram* hist);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
