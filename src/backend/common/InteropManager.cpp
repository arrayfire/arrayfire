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
#include <common/InteropManager.hpp>
#include <cstdio>
#include <common/err_common.hpp>
#include <common/util.hpp>
#include <backend.hpp>
#include <GraphicsResourceManager.hpp>
#include <platform.hpp>

template<typename R>
using RVector = std::vector<std::shared_ptr<R>>;

namespace common
{
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
    void * key = (void*)image;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(image->pixels());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Plot* plot)
{
    void * key = (void*)plot;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(plot->vertices());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Histogram* histogram)
{
    void * key = (void*)histogram;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(histogram->vertices());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::Surface* surface)
{
    void * key = (void*)surface;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(surface->vertices());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
RVector<R> InteropManager<T, R>::getBufferResource(const forge::VectorField* field)
{
    void * key = (void*)field;

    if (mInteropMap.find(key) == mInteropMap.end()) {
        std::vector<uint32_t> handles;
        handles.push_back(field->vertices());
        handles.push_back(field->directions());
        std::vector<resource_t> output = static_cast<T*>(this)->registerResources(handles);

        mInteropMap[key] = output;
    }

    return mInteropMap[key];
}

template<class T, typename R>
void InteropManager<T, R>::destroyResources()
{
    for(auto iter : mInteropMap) {
        iter.second.clear();
    }
}

template class InteropManager<detail::GraphicsResourceManager, detail::CGR_t>;
}
#endif
#endif
