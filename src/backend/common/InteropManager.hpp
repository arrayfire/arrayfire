/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(WITH_GRAPHICS)
#include <forge.h>
#include <err_common.hpp>
#include <util.hpp>
#include <cstdio>
#include <map>
#include <vector>

namespace common
{
template<typename T, typename R>
class InteropManager
{
    public:
        InteropManager() {}

        ~InteropManager() {
            try {
                destroyResources();
            } catch (AfError &ex) {

                std::string perr = getEnvVar("AF_PRINT_ERRORS");
                if(!perr.empty()) {
                    if(perr != "0") fprintf(stderr, "%s\n", ex.what());
                }
            }
        }

        R* getBufferResource(const forge::Image* image) {
            void * key = (void*)image;

            if (interopMap.find(key) == interopMap.end()) {
                std::vector<uint32_t> handles;
                handles.push_back(image->pixels());
                std::vector<R> output = static_cast<T*>(this)->registerResources(handles);
                interopMap[key] = output;
            }

            return &interopMap[key].front();
        }

        R* getBufferResource(const forge::Plot* plot) {
            void * key = (void*)plot;

            if (interopMap.find(key) == interopMap.end()) {
                std::vector<uint32_t> handles;
                handles.push_back(plot->vertices());
                std::vector<R> output = static_cast<T*>(this)->registerResources(handles);
                interopMap[key] = output;
            }

            return &interopMap[key].front();
        }

        R* getBufferResource(const forge::Histogram* histogram) {
            void * key = (void*)histogram;

            if (interopMap.find(key) == interopMap.end()) {
                std::vector<uint32_t> handles;
                handles.push_back(histogram->vertices());
                std::vector<R> output = static_cast<T*>(this)->registerResources(handles);
                interopMap[key] = output;
            }

            return &interopMap[key].front();
        }

        R* getBufferResource(const forge::Surface* surface) {
            void * key = (void*)surface;

            if (interopMap.find(key) == interopMap.end()) {
                std::vector<uint32_t> handles;
                handles.push_back(surface->vertices());
                std::vector<R> output = static_cast<T*>(this)->registerResources(handles);
                interopMap[key] = output;
            }

            return &interopMap[key].front();
        }

        R* getBufferResource(const forge::VectorField* field) {
            void * key = (void*)field;

            if (interopMap.find(key) == interopMap.end()) {
                std::vector<uint32_t> handles;
                handles.push_back(field->vertices());
                handles.push_back(field->directions());
                std::vector<R> output = static_cast<T*>(this)->registerResources(handles);
                interopMap[key] = output;
            }

            return &interopMap[key].front();
        }

    protected:
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);

        void destroyResources() {
            for(auto iter : interopMap) {
                for(auto ct : iter.second) {
                    static_cast<T*>(this)->unregisterResource(ct);
                }
                iter.second.clear();
            }
        }

    private:
        std::map<void *, std::vector<R> > interopMap;
};
}
#endif
