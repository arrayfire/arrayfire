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

#if defined(WITH_GRAPHICS)

#include <Array.hpp>
#include <image.hpp>
#include <err_cuda.hpp>
#include <debug_cuda.hpp>
#include <platform.hpp>
#include <graphics_common.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using af::dim4;

namespace cuda
{

typedef std::map<fg_image_handle, cudaGraphicsResource *> interop_t;
typedef interop_t::iterator iter_t;

// Manager Class for cudaPBOResource: calls garbage collection at the end of the program
class InteropManager
{
    private:
        interop_t interop_maps[DeviceManager::MAX_DEVICES];

    public:
        static InteropManager& getInstance();
        ~InteropManager();
        cudaGraphicsResource* getPBOResource(const fg_image_handle handle);

    protected:
        InteropManager() {}
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        void destroyResources();
};

void InteropManager::destroyResources()
{
    int n = getActiveDeviceId();
    for(iter_t iter = interop_maps[n].begin(); iter != interop_maps[n].end(); iter++)
        cudaGraphicsUnregisterResource(iter->second);
}

InteropManager::~InteropManager()
{
    for(int i = 0; i < getDeviceCount(); i++) {
        setDevice(i);
        destroyResources();
    }
}

InteropManager& InteropManager::getInstance()
{
    static InteropManager my_instance;
    return my_instance;
}

cudaGraphicsResource* InteropManager::getPBOResource(const fg_image_handle key)
{
    int device = getActiveDeviceId();

    if(interop_maps[device].find(key) == interop_maps[device].end()) {
        cudaGraphicsResource *cudaPBOResource;
        // Register PBO with CUDA
        cudaGraphicsGLRegisterBuffer(&cudaPBOResource, key->gl_PBO, cudaGraphicsMapFlagsWriteDiscard);
        interop_maps[device][key] = cudaPBOResource;
    }

    return interop_maps[device][key];
}

template<typename T>
void copy_image(const Array<T> &in, const fg_image_handle image)
{
    InteropManager& intrpMngr = InteropManager::getInstance();

    cudaGraphicsResource *cudaPBOResource = intrpMngr.getPBOResource(image);

    const T *d_X = in.get();
    // Map resource. Copy data to PBO. Unmap resource.
    size_t num_bytes;
    T* d_pbo = NULL;
    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    cudaGraphicsResourceGetMappedPointer((void **)&d_pbo, &num_bytes, cudaPBOResource);
    cudaMemcpy(d_pbo, d_X, num_bytes, cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);

    // Unlock array
    // Not implemented yet
    // X.unlock();

    CheckGL("After cuda resource copy");

    POST_LAUNCH_CHECK();
}

#define INSTANTIATE(T)      \
    template void copy_image<T>(const Array<T> &in, const fg_image_handle image);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(char)

}

#endif
