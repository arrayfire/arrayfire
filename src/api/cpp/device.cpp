/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/device.h>
#include <af/compatible.h>
#include <af/traits.hpp>
#include <af/backend.h>
#include "error.hpp"

namespace af
{
    void setBackend(const Backend bknd)
    {
        AF_THROW(af_set_backend(bknd));
    }

    unsigned getBackendCount()
    {
        unsigned temp = 1;
        AF_THROW(af_get_backend_count(&temp));
        return temp;
    }

    int getAvailableBackends()
    {
        int result = 0;
        AF_THROW(af_get_available_backends(&result));
        return result;
    }

    af::Backend getBackendId(const array &in)
    {
        af::Backend result = (af::Backend)0;
        AF_THROW(af_get_backend_id(&result, in.get()));
        return result;
    }

    void info()
    {
        AF_THROW(af_info());
    }

    const char* infoString(const bool verbose)
    {
        char *str = NULL;
        AF_THROW(af_info_string(&str, verbose));
        return (const char *)str;
    }

    void deviceprop(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
    {
        deviceInfo(d_name, d_platform, d_toolkit, d_compute);
    }
    void deviceInfo(char* d_name, char* d_platform, char *d_toolkit, char* d_compute)
    {
        AF_THROW(af_device_info(d_name, d_platform, d_toolkit, d_compute));
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
        case s16: return sizeof(short);
        case u16: return sizeof(unsigned short);
        default: return sizeof(float);
        }
    }

    void *alloc(const size_t elements, const af::dtype type)
    {
        void *ptr;
        AF_THROW(af_alloc_device(&ptr, elements * size_of(type)));
        // FIXME: Add to map
        return ptr;
    }

    void *pinned(const size_t elements, const af::dtype type)
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

    void *allocHost(const size_t elements, const af::dtype type)
    {
        void *ptr;
        AF_THROW(af_alloc_host(&ptr, elements * size_of(type)));
        return ptr;
    }

    void freeHost(const void *ptr)
    {
        AF_THROW(af_free_host((void *)ptr));
    }

    void printMemInfo(const char *msg, const int device_id)
    {
        AF_THROW(af_print_mem_info(msg, device_id));
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

    void setMemStepSize(const size_t step_bytes)
    {
        AF_THROW(af_set_mem_step_size(step_bytes));
    }

    size_t getMemStepSize()
    {
        size_t size_bytes = 0;
        AF_THROW(af_get_mem_step_size(&size_bytes));
        return size_bytes;
    }

#define INSTANTIATE(T)                                                      \
    template<> AFAPI                                                        \
    T* alloc(const size_t elements)                                         \
    {                                                                       \
        return (T*)alloc(elements, (af::dtype)dtype_traits<T>::af_type);    \
    }                                                                       \
    template<> AFAPI                                                        \
    T* pinned(const size_t elements)                                        \
    {                                                                       \
        return (T*)pinned(elements, (af::dtype)dtype_traits<T>::af_type);   \
    }                                                                       \
    template<> AFAPI                                                        \
    T* allocHost(const size_t elements)                                     \
    {                                                                       \
        return (T*)allocHost(elements, (af::dtype)dtype_traits<T>::af_type);\
    }

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(cfloat)
    INSTANTIATE(cdouble)
    INSTANTIATE(int)
    INSTANTIATE(unsigned)
    INSTANTIATE(unsigned char)
    INSTANTIATE(char)
    INSTANTIATE(short)
    INSTANTIATE(unsigned short)

}
