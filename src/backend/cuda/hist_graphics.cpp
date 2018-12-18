/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <hist_graphics.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <GraphicsResourceManager.hpp>

namespace cuda {

template<typename T>
void copy_histogram(const Array<T> &data, fg_histogram hist)
{
    ForgeModule& _ = graphics::forgePlugin();
    auto stream = cuda::getActiveStream();
    if(DeviceManager::checkGraphicsInteropCapability()) {
        const T *d_P = data.get();

        auto res = interopManager().getHistogramResources(hist);

        size_t bytes = 0;
        T* d_vbo = NULL;
        cudaGraphicsMapResources(1, res[0].get(), stream);
        cudaGraphicsResourceGetMappedPointer((void **)&d_vbo,
                                             &bytes, *(res[0].get()));
        cudaMemcpyAsync(d_vbo, d_P, bytes, cudaMemcpyDeviceToDevice, stream);
        cudaGraphicsUnmapResources(1, res[0].get(), stream);

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        unsigned bytes = 0, buffer = 0;
        FG_CHECK(fg_get_histogram_vertex_buffer(&buffer, hist));
        FG_CHECK(fg_get_histogram_vertex_buffer_size(&bytes, hist));

        CheckGL("Begin CUDA fallback-resource copy");
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        GLubyte* ptr = (GLubyte*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, data.get(), bytes,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T)  \
template void copy_histogram<T>(const Array<T> &, fg_histogram);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}
