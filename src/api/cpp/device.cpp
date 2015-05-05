/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/device.h>
#include <af/compatible.h>
#include <af/traits.hpp>
#include "error.hpp"

namespace af
{
    void info()
    {
        AF_THROW(af_info());
    }

    void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
    {
        AF_THROW(af_deviceprop(d_name, d_platform, d_toolkit, d_compute));
    }

    int getDeviceCount()
    {
        int devices = -1;
        AF_THROW(af_get_device_count(&devices));
        return devices;
    }

    int devicecount() { return getDeviceCount(); }

    void setDevice(const int device)
    {
        AF_THROW(af_set_device(device));
    }

    void deviceset(const int device) { setDevice(device); }

    int getDevice()
    {
        int device = 0;
        AF_THROW(af_get_device(&device));
        return device;
    }

    bool isDoubleAvailable(const int device)
    {
        bool temp;
        AF_THROW(af_get_dbl_support(&temp, device));
        return temp;
    }

    int deviceget() { return getDevice(); }

    void sync(int device)
    {
        AF_THROW(af_sync(device));
    }

    ///////////////////////////////////////////////////////////////////////////
    // Alloc and free host, pinned, zero copy
    static unsigned size_of(af::dtype type)
    {
        switch(type) {
        case f32: return sizeof(float);
        case f64: return sizeof(double);
        case s32: return sizeof(int);
        case u32: return sizeof(unsigned);
        case u8 : return sizeof(unsigned char);
        case b8 : return sizeof(unsigned char);
        case c32: return sizeof(float) * 2;
        case c64: return sizeof(double) * 2;
        default: return sizeof(float);
        }
    }

    void *alloc(size_t elements, af::dtype type)
    {
        void *ptr;
        AF_THROW(af_alloc_device(&ptr, elements * size_of(type)));
        // FIXME: Add to map
        return ptr;
    }

    void *pinned(size_t elements, af::dtype type)
    {
        void *ptr;
        AF_THROW(af_alloc_pinned(&ptr, elements * size_of(type)));
        // FIXME: Add to map
        return ptr;
    }

    void free(const void *ptr)
    {
        //FIXME: look up map and call the right free
        AF_THROW(af_free_device((void *)ptr));
    }

    void freePinned(const void *ptr)
    {
        //FIXME: look up map and call the right free
        AF_THROW(af_free_pinned((void *)ptr));
    }

    void deviceGC()
    {
        AF_THROW(af_device_gc());
    }

    void deviceMemInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                       size_t *lock_bytes,  size_t *lock_buffers)
    {
        AF_THROW(af_device_mem_info(alloc_bytes, alloc_buffers,
                                    lock_bytes,  lock_buffers));
    }

#define INSTANTIATE(T)                                                      \
    template<> AFAPI                                                        \
    T* alloc(size_t elements)                                               \
    {                                                                       \
        return (T*)alloc(elements, (af::dtype)dtype_traits<T>::af_type);     \
    }                                                                       \
    template<> AFAPI                                                        \
    T* pinned(size_t elements)                                              \
    {                                                                       \
        return (T*)pinned(elements, (af::dtype)dtype_traits<T>::af_type);    \
    }

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(unsigned)
    INSTANTIATE(unsigned char)
    INSTANTIATE(char)

    void initGraphics(int device)
    {
        AF_THROW(af_init_graphics(device));
    }
}
