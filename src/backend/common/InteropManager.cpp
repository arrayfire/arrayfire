/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)
//FIXME CPU backend doesn't required the following class implementation
//InteropManager.hpp is not used while building CPU backend.
#ifndef AF_CPU
#include <boost/thread.hpp>
#include <common/InteropManager.hpp>
#include <common/types.hpp>
#include <cstdio>
#include <err_common.hpp>
#include <util.hpp>
#include <backend.hpp>
#include <GraphicsResourceManager.hpp>
#include <platform.hpp>

typedef boost::shared_mutex smutex_t;
typedef boost::shared_lock<smutex_t> rlock_t;
typedef boost::unique_lock<smutex_t> wlock_t;
typedef boost::upgrade_lock<smutex_t> ulock_t;
typedef boost::upgrade_to_unique_lock<smutex_t> u2ulock_t;

template<typename R>
using RVector = std::vector<std::shared_ptr<R>>;

namespace common
{
static smutex_t gInteropMutexes[detail::DeviceManager::MAX_DEVICES];

template<class T, typename R>
InteropManager<T, R>::~InteropManager()
{
    try {
        destroyResources();
    } catch (AfError &ex) {

        std::string perr = getEnvVar("AF_PRINT_ERRORS");
        if(!perr.empty()) {
            if(perr != "0") fprintf(stderr, "%s\n", ex.what());
        }
    }
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Image* image)
{
    ulock_t lock(gInteropMutexes[detail::getActiveDeviceId()]);
    void * key = (void*)image;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(image->pixels());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        u2ulock_t wlock(lock);
        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Plot* plot)
{
    ulock_t lock(gInteropMutexes[detail::getActiveDeviceId()]);
    void * key = (void*)plot;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(plot->vertices());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        u2ulock_t wlock(lock);
        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Histogram* histogram)
{
    ulock_t lock(gInteropMutexes[detail::getActiveDeviceId()]);
    void * key = (void*)histogram;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(histogram->vertices());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        u2ulock_t wlock(lock);
        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Surface* surface)
{
    ulock_t lock(gInteropMutexes[detail::getActiveDeviceId()]);
    void * key = (void*)surface;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(surface->vertices());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        u2ulock_t wlock(lock);
        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::VectorField* field)
{
    ulock_t lock(gInteropMutexes[detail::getActiveDeviceId()]);
    void * key = (void*)field;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(field->vertices());
        handles.push_back(field->directions());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        u2ulock_t wlock(lock);
        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
void InteropManager<T, R>::destroyResources()
{
    wlock_t lock(gInteropMutexes[detail::getActiveDeviceId()]);
    for(auto iter : mInteropMap) {
        iter.second.clear();
    }
}

template class InteropManager<detail::GraphicsResourceManager, detail::CGR_t>;
}
#endif
#endif
