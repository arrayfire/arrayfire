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
#include <err_cuda.hpp>
#include <debug_cuda.hpp>

namespace cuda
{

template<typename T>
void copy_histogram(const Array<T> &data, const fg::Histogram* hist)
{
    const T *d_P = data.get();

    InteropManager& intrpMngr = InteropManager::getInstance();

    cudaGraphicsResource *cudaVBOResource = intrpMngr.getBufferResource(hist);
    // Map resource. Copy data to VBO. Unmap resource.
    size_t num_bytes = hist->size();
    T* d_vbo = NULL;
    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, cudaVBOResource);
    cudaMemcpy(d_vbo, d_P, num_bytes, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);

    CheckGL("After cuda resource copy");

    POST_LAUNCH_CHECK();
}

#define INSTANTIATE(T)  \
    template void copy_histogram<T>(const Array<T> &data, const fg::Histogram* hist);

INSTANTIATE(float)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
