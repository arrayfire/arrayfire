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
#include <map>
#include <vector>
#include <memory>

namespace common
{
template<class T, typename R>
class InteropManager
{
    using resource_t = typename std::shared_ptr<R>;
    using res_vec_t = typename std::vector<resource_t>;
    using res_map_t = typename std::map<void*, res_vec_t>;

    public:
        InteropManager() {}
        ~InteropManager();

        res_vec_t getBufferResource(const forge::Image* image);
        res_vec_t getBufferResource(const forge::Plot* plot);
        res_vec_t getBufferResource(const forge::Histogram* histogram);
        res_vec_t getBufferResource(const forge::Surface* surface);
        res_vec_t getBufferResource(const forge::VectorField* field);

    protected:
        InteropManager(InteropManager const&);
        void operator=(InteropManager const&);
        void destroyResources();

        res_map_t mInteropMap;
};
}
#endif
