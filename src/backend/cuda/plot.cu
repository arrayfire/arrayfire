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

template<typename T>
void copy_plot(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot)
{
    T xmax = reduce_all<af_max_t,T, T>(X);
    T xmin = reduce_all<af_min_t,T, T>(X);
    T ymax = reduce_all<af_max_t,T, T>(Y);
    T ymin = reduce_all<af_min_t,T, T>(Y);

    // Interleave
    // TODO Create a kernel for this
    Array<T> Z = join(1, X, Y);
    Z = reorder(Z, dim4(1, 0, 2, 3));
    dim4 zdims = X.dims();
    zdims[0] += Y.dims()[0];
    Z.modDims(zdims);
    T *d_Z = Z.get();

    // Create Data Store
    glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo[0]);
    size_t bytes = (X.elements() + Y.elements()) * sizeof(T);
    if(bytes != plot->vbosize) {
        glBufferData(GL_ARRAY_BUFFER, bytes, NULL, GL_STATIC_DRAW);
        plot->vbosize = bytes;
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    InteropManager& intrpMngr = InteropManager::getInstance();

    cudaGraphicsResource *cudaVBOResource = intrpMngr.getBufferResource(plot);

    // Map resource. Copy data to VBO. Unmap resource.
    size_t num_bytes;
    T* d_vbo = NULL;
    cudaGraphicsMapResources(1, &cudaVBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, cudaVBOResource);
    cudaMemcpy(d_vbo, d_Z, num_bytes, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cudaVBOResource, 0);

    fg_plot2d(plot, xmax, xmin, ymax, ymin);
    // Unlock array
    // Not implemented yet
    // X.unlock();

    CheckGL("After cuda resource copy");

    POST_LAUNCH_CHECK();

}

#define INSTANTIATE(T)  \
    template void copy_plot<T>(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
