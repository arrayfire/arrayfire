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
#include <vector_field.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <GraphicsResourceManager.hpp>

using af::dim4;

namespace cuda
{
using namespace gl;

template<typename T>
void copy_vector_field(const Array<T> &points, const Array<T> &directions,
                       forge::VectorField* vector_field)
{
    if(DeviceManager::checkGraphicsInteropCapability()) {
        ShrdResVector res = interopManager().getBufferResource(vector_field);
        CGR_t resources[2] = {*res[0].get(), *res[1].get()};

        // Map resource. Copy data to VBO. Unmap resource.
        // Map all resources at once.
        cudaGraphicsMapResources(2, resources, cuda::getActiveStream());

        // Points
        {
            const T *ptr     = points.get();
            size_t num_bytes = vector_field->verticesSize();
            T* d_vbo = NULL;
            cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, resources[0]);
            cudaMemcpyAsync(d_vbo, ptr, num_bytes, cudaMemcpyDeviceToDevice, cuda::getActiveStream());
        }
        // Directions
        {
            const T *ptr = directions.get();
            size_t num_bytes = vector_field->directionsSize();
            T* d_vbo = NULL;
            cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, resources[1]);
            cudaMemcpyAsync(d_vbo, ptr, num_bytes, cudaMemcpyDeviceToDevice, cuda::getActiveStream());
        }
        cudaGraphicsUnmapResources(2, resources, cuda::getActiveStream());

        CheckGL("After cuda resource copy");

        POST_LAUNCH_CHECK();
    } else {
        CheckGL("Begin CUDA fallback-resource copy");
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, vector_field->vertices());
        gl::GLubyte* ptr = (gl::GLubyte*)glMapBuffer((gl::GLenum)GL_ARRAY_BUFFER, (gl::GLenum)GL_WRITE_ONLY);
        if (ptr) {
            auto stream = cuda::getActiveStream();
            CUDA_CHECK(cudaMemcpyAsync(ptr, points.get(), vector_field->verticesSize(),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer((gl::GLenum)GL_ARRAY_BUFFER);
        }
        glBindBuffer((gl::GLenum)GL_ARRAY_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T)                                                                      \
    template void copy_vector_field<T>(const Array<T> &points, const Array<T> &directions,  \
                                       forge::VectorField* vector_field);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
