/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/err_common.hpp>
#include <common/forge_loader.hpp>
#include <common/util.hpp>

#include <cstdio>
#include <map>
#include <memory>
#include <vector>

namespace arrayfire {
namespace common {
template<class T, typename R>
class InteropManager {
    using resource_t = typename std::shared_ptr<R>;
    using res_vec_t  = typename std::vector<resource_t>;
    using res_map_t  = typename std::map<void *, res_vec_t>;

   public:
    InteropManager() {}

    ~InteropManager() {
        try {
            destroyResources();
        } catch (const AfError &ex) {
            std::string perr = getEnvVar("AF_PRINT_ERRORS");
            if (!perr.empty()) {
                if (perr != "0") fprintf(stderr, "%s\n", ex.what());
            }
        }
    }

    res_vec_t getImageResources(const fg_window image) {
        if (mInteropMap.find(image) == mInteropMap.end()) {
            uint32_t buffer;
            FG_CHECK(common::forgePlugin().fg_get_pixel_buffer(&buffer, image));
            mInteropMap[image] =
                static_cast<T *>(this)->registerResources({buffer});
        }
        return mInteropMap[image];
    }

    res_vec_t getPlotResources(const fg_plot plot) {
        if (mInteropMap.find(plot) == mInteropMap.end()) {
            uint32_t buffer;
            FG_CHECK(
                common::forgePlugin().fg_get_plot_vertex_buffer(&buffer, plot));
            mInteropMap[plot] =
                static_cast<T *>(this)->registerResources({buffer});
        }
        return mInteropMap[plot];
    }

    res_vec_t getHistogramResources(const fg_histogram histogram) {
        if (mInteropMap.find(histogram) == mInteropMap.end()) {
            uint32_t buffer;
            FG_CHECK(common::forgePlugin().fg_get_histogram_vertex_buffer(
                &buffer, histogram));
            mInteropMap[histogram] =
                static_cast<T *>(this)->registerResources({buffer});
        }
        return mInteropMap[histogram];
    }

    res_vec_t getSurfaceResources(const fg_surface surface) {
        if (mInteropMap.find(surface) == mInteropMap.end()) {
            uint32_t buffer;
            FG_CHECK(common::forgePlugin().fg_get_surface_vertex_buffer(
                &buffer, surface));
            mInteropMap[surface] =
                static_cast<T *>(this)->registerResources({buffer});
        }
        return mInteropMap[surface];
    }

    res_vec_t getVectorFieldResources(const fg_vector_field field) {
        if (mInteropMap.find(field) == mInteropMap.end()) {
            uint32_t verts, dirs;
            FG_CHECK(common::forgePlugin().fg_get_vector_field_vertex_buffer(
                &verts, field));
            FG_CHECK(common::forgePlugin().fg_get_vector_field_direction_buffer(
                &dirs, field));
            mInteropMap[field] =
                static_cast<T *>(this)->registerResources({verts, dirs});
        }
        return mInteropMap[field];
    }

   protected:
    InteropManager(InteropManager const &);
    void operator=(InteropManager const &);

    void destroyResources() {
        for (auto iter : mInteropMap) iter.second.clear();
    }

    res_map_t mInteropMap;
};
}  // namespace common
}  // namespace arrayfire
