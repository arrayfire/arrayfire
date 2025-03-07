/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Parts of this code sourced from SnopyDogy
// https://gist.github.com/SnopyDogy/a9a22497a893ec86aa3e

#include <Array.hpp>
#include <GraphicsResourceManager.hpp>
#include <debug_cuda.hpp>
#include <device_manager.hpp>
#include <err_cuda.hpp>
#include <image.hpp>

using af::dim4;
using arrayfire::common::ForgeManager;
using arrayfire::common::ForgeModule;
using arrayfire::common::forgePlugin;

namespace arrayfire {
namespace cuda {

template<typename T>
void copy_image(const Array<T> &in, fg_image image) {
    auto stream = getActiveStream();
    if (DeviceManager::checkGraphicsInteropCapability()) {
        auto res = interopManager().getImageResources(image);

        const T *d_X = in.get();
        size_t bytes = 0;
        T *d_pixels  = NULL;
        cudaGraphicsMapResources(1, res[0].get(), stream);
        cudaGraphicsResourceGetMappedPointer((void **)&d_pixels, &bytes,
                                             *(res[0].get()));
        cudaMemcpyAsync(d_pixels, d_X, bytes, cudaMemcpyDeviceToDevice, stream);
        cudaGraphicsUnmapResources(1, res[0].get(), stream);

        POST_LAUNCH_CHECK();
        CheckGL("After cuda resource copy");
    } else {
        ForgeModule &_ = common::forgePlugin();
        CheckGL("Begin CUDA fallback-resource copy");
        unsigned data_size = 0, buffer = 0;
        FG_CHECK(_.fg_get_image_size(&data_size, image));
        FG_CHECK(_.fg_get_pixel_buffer(&buffer, image));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, data_size, 0, GL_STREAM_DRAW);
        auto *ptr = static_cast<GLubyte *>(
            glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY));
        if (ptr) {
            CUDA_CHECK(cudaMemcpyAsync(ptr, in.get(), data_size,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        CheckGL("End CUDA fallback-resource copy");
    }
}

#define INSTANTIATE(T) template void copy_image<T>(const Array<T> &, fg_image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace cuda
}  // namespace arrayfire
