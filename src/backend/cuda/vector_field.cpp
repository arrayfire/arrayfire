/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <GraphicsResourceManager.hpp>
#include <debug_cuda.hpp>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <vector_field.hpp>

using af::dim4;
using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace cuda {

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       fg_vector_field vfield) {
    auto stream = getActiveStream();
    if (DeviceManager::checkGraphicsInteropCapability()) {
        auto res = interopManager().getVectorFieldResources(vfield);
        cudaGraphicsResource_t resources[2] = {*res[0].get(), *res[1].get()};

        cudaGraphicsMapResources(2, resources, stream);

        // Points
        {
            const T *ptr = points.get();
            size_t bytes = 0;
            T *d_vbo     = NULL;
            cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &bytes,
                                                 resources[0]);
            cudaMemcpyAsync(d_vbo, ptr, bytes, cudaMemcpyDeviceToDevice,
                            stream);
        }
        // Directions
        {
            const T *ptr = directions.get();
            size_t bytes = 0;
            T *d_vbo     = NULL;
            cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &bytes,
                                                 resources[1]);
            cudaMemcpyAsync(d_vbo, ptr, bytes, cudaMemcpyDeviceToDevice,
                            stream);
        }
        cudaGraphicsUnmapResources(2, resources, stream);

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        ForgeModule &_ = forgePlugin();
        CheckGL("Begin CUDA fallback-resource copy");
        unsigned size1 = 0, size2 = 0;
        unsigned buff1 = 0, buff2 = 0;
        FG_CHECK(_.fg_get_vector_field_vertex_buffer_size(&size1, vfield));
        FG_CHECK(_.fg_get_vector_field_direction_buffer_size(&size2, vfield));
        FG_CHECK(_.fg_get_vector_field_vertex_buffer(&buff1, vfield));
        FG_CHECK(_.fg_get_vector_field_direction_buffer(&buff2, vfield));

        // Points
        glBindBuffer(GL_ARRAY_BUFFER, buff1);
        auto *ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, points.get(), size1,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // Directions
        glBindBuffer(GL_ARRAY_BUFFER, buff2);
        ptr =
            static_cast<GLubyte *>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, directions.get(), size2,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T)                                                     \
    template void copy_vector_field<T>(const Array<T> &, const Array<T> &, \
                                       fg_vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace cuda
}  // namespace arrayfire
