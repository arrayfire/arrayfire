/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <backend.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <common/util.hpp>
#include <platform.hpp>
#include <mutex>
#include <utility>

using arrayfire::common::getEnvVar;
using std::make_pair;
using std::string;

namespace arrayfire {
namespace common {

/// Dynamically loads forge function pointer at runtime
#define FG_MODULE_FUNCTION_INIT(NAME) \
    NAME = DependencyModule::getSymbol<decltype(&::NAME)>(#NAME)

ForgeModule::ForgeModule() : DependencyModule("forge", nullptr) {
    if (DependencyModule::isLoaded()) {
        FG_MODULE_FUNCTION_INIT(fg_create_window);
        FG_MODULE_FUNCTION_INIT(fg_get_window_context_handle);
        FG_MODULE_FUNCTION_INIT(fg_get_window_display_handle);
        FG_MODULE_FUNCTION_INIT(fg_make_window_current);
        FG_MODULE_FUNCTION_INIT(fg_set_window_font);
        FG_MODULE_FUNCTION_INIT(fg_set_window_position);
        FG_MODULE_FUNCTION_INIT(fg_set_window_title);
        FG_MODULE_FUNCTION_INIT(fg_set_window_size);
        FG_MODULE_FUNCTION_INIT(fg_set_window_colormap);
        FG_MODULE_FUNCTION_INIT(fg_draw_chart_to_cell);
        FG_MODULE_FUNCTION_INIT(fg_draw_chart);
        FG_MODULE_FUNCTION_INIT(fg_draw_image_to_cell);
        FG_MODULE_FUNCTION_INIT(fg_draw_image);
        FG_MODULE_FUNCTION_INIT(fg_swap_window_buffers);
        FG_MODULE_FUNCTION_INIT(fg_close_window);
        FG_MODULE_FUNCTION_INIT(fg_show_window);
        FG_MODULE_FUNCTION_INIT(fg_hide_window);
        FG_MODULE_FUNCTION_INIT(fg_release_window);

        FG_MODULE_FUNCTION_INIT(fg_create_font);
        FG_MODULE_FUNCTION_INIT(fg_load_system_font);
        FG_MODULE_FUNCTION_INIT(fg_release_font);

        FG_MODULE_FUNCTION_INIT(fg_create_image);
        FG_MODULE_FUNCTION_INIT(fg_get_pixel_buffer);
        FG_MODULE_FUNCTION_INIT(fg_get_image_size);
        FG_MODULE_FUNCTION_INIT(fg_release_image);

        FG_MODULE_FUNCTION_INIT(fg_create_plot);
        FG_MODULE_FUNCTION_INIT(fg_set_plot_color);
        FG_MODULE_FUNCTION_INIT(fg_get_plot_vertex_buffer);
        FG_MODULE_FUNCTION_INIT(fg_get_plot_vertex_buffer_size);
        FG_MODULE_FUNCTION_INIT(fg_release_plot);

        FG_MODULE_FUNCTION_INIT(fg_create_histogram);
        FG_MODULE_FUNCTION_INIT(fg_set_histogram_color);
        FG_MODULE_FUNCTION_INIT(fg_get_histogram_vertex_buffer);
        FG_MODULE_FUNCTION_INIT(fg_get_histogram_vertex_buffer_size);
        FG_MODULE_FUNCTION_INIT(fg_release_histogram);

        FG_MODULE_FUNCTION_INIT(fg_create_surface);
        FG_MODULE_FUNCTION_INIT(fg_set_surface_color);
        FG_MODULE_FUNCTION_INIT(fg_get_surface_vertex_buffer);
        FG_MODULE_FUNCTION_INIT(fg_get_surface_vertex_buffer_size);
        FG_MODULE_FUNCTION_INIT(fg_release_surface);

        FG_MODULE_FUNCTION_INIT(fg_create_vector_field);
        FG_MODULE_FUNCTION_INIT(fg_set_vector_field_color);
        FG_MODULE_FUNCTION_INIT(fg_get_vector_field_vertex_buffer_size);
        FG_MODULE_FUNCTION_INIT(fg_get_vector_field_direction_buffer_size);
        FG_MODULE_FUNCTION_INIT(fg_get_vector_field_vertex_buffer);
        FG_MODULE_FUNCTION_INIT(fg_get_vector_field_direction_buffer);
        FG_MODULE_FUNCTION_INIT(fg_release_vector_field);

        FG_MODULE_FUNCTION_INIT(fg_create_chart);
        FG_MODULE_FUNCTION_INIT(fg_get_chart_type);
        FG_MODULE_FUNCTION_INIT(fg_get_chart_axes_limits);
        FG_MODULE_FUNCTION_INIT(fg_set_chart_axes_limits);
        FG_MODULE_FUNCTION_INIT(fg_set_chart_axes_titles);
        FG_MODULE_FUNCTION_INIT(fg_set_chart_label_format);
        FG_MODULE_FUNCTION_INIT(fg_append_image_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_plot_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_histogram_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_surface_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_vector_field_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_release_chart);

        FG_MODULE_FUNCTION_INIT(fg_err_to_string);

        if (!DependencyModule::symbolsLoaded()) {
            string error_message =
                "Error loading Forge: " + DependencyModule::getErrorMessage() +
                "\nForge or one of it's dependencies failed to "
                "load. Try installing Forge or check if Forge is in the "
                "search path.";
            AF_ERROR(error_message.c_str(), AF_ERR_LOAD_LIB);
        }
    }
}

template<typename T>
fg_dtype getGLType() {
    return FG_FLOAT32;
}

fg_marker_type getFGMarker(const af_marker_type af_marker) {
    fg_marker_type fg_marker;
    switch (af_marker) {
        case AF_MARKER_NONE: fg_marker = FG_MARKER_NONE; break;
        case AF_MARKER_POINT: fg_marker = FG_MARKER_POINT; break;
        case AF_MARKER_CIRCLE: fg_marker = FG_MARKER_CIRCLE; break;
        case AF_MARKER_SQUARE: fg_marker = FG_MARKER_SQUARE; break;
        case AF_MARKER_TRIANGLE: fg_marker = FG_MARKER_TRIANGLE; break;
        case AF_MARKER_CROSS: fg_marker = FG_MARKER_CROSS; break;
        case AF_MARKER_PLUS: fg_marker = FG_MARKER_PLUS; break;
        case AF_MARKER_STAR: fg_marker = FG_MARKER_STAR; break;
        default: fg_marker = FG_MARKER_NONE; break;
    }
    return fg_marker;
}

#define INSTANTIATE_GET_FG_TYPE(T, ForgeEnum) \
    template<>                                \
    fg_dtype getGLType<T>() {                 \
        return ForgeEnum;                     \
    }

INSTANTIATE_GET_FG_TYPE(float, FG_FLOAT32);
INSTANTIATE_GET_FG_TYPE(int, FG_INT32);
INSTANTIATE_GET_FG_TYPE(unsigned, FG_UINT32);
INSTANTIATE_GET_FG_TYPE(char, FG_INT8);
INSTANTIATE_GET_FG_TYPE(unsigned char, FG_UINT8);
INSTANTIATE_GET_FG_TYPE(unsigned short, FG_UINT16);
INSTANTIATE_GET_FG_TYPE(short, FG_INT16);

// NOLINTNEXTLINE(misc-unused-parameters)
GLenum glErrorCheck(const char* msg, const char* file, int line) {
// Skipped in release mode
#ifndef NDEBUG
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n",
                file, line, msg, static_cast<int>(x), glGetString(x));
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
#else
    UNUSED(msg);
    UNUSED(file);
    UNUSED(line);
    return static_cast<GLenum>(0);
#endif
}

size_t getTypeSize(GLenum type) {
    switch (type) {
        case GL_FLOAT: return sizeof(float);
        case GL_INT: return sizeof(int);
        case GL_UNSIGNED_INT: return sizeof(unsigned);
        case GL_SHORT: return sizeof(short);
        case GL_UNSIGNED_SHORT: return sizeof(unsigned short);
        case GL_BYTE: return sizeof(char);
        case GL_UNSIGNED_BYTE: return sizeof(unsigned char);
        default: return sizeof(float);
    }
}

void makeContextCurrent(fg_window window) {
    FG_CHECK(common::forgePlugin().fg_make_window_current(window));
    CheckGL("End makeContextCurrent");
}

// dir -> true = round up, false = round down
double step_round(const double in, const bool dir) {
    if (in == 0) { return 0; }

    static const double LOG2 = log10(2);
    static const double LOG4 = log10(4);
    static const double LOG6 = log10(6);
    static const double LOG8 = log10(8);

    // log_in is of the form "s abc.xyz", where
    // s is either + or -; + indicates abs(in) >= 1 and - indicates 0 < abs(in)
    // < 1 (log10(1) is +0)
    const double sign   = in < 0 ? -1 : 1;
    const double log_in = std::log10(std::fabs(in));
    const double mag    = std::pow(10, std::floor(log_in)) *
                       sign;  // Number of digits either left or right of 0
    const double dec = std::log10(in / mag);  // log of the fraction

    // This means in is of the for 10^n
    if (dec == 0) { return in; }

    // For negative numbers, -ve round down = +ve round up and vice versa
    bool op_dir = in > 0 ? dir : !dir;

    double mult = 1;

    // Round up
    if (op_dir) {
        if (dec <= LOG2) {
            mult = 2;
        } else if (dec <= LOG4) {
            mult = 4;
        } else if (dec <= LOG6) {
            mult = 6;
        } else if (dec <= LOG8) {
            mult = 8;
        } else {
            mult = 10;
        }
    } else {  // Round down
        if (dec < LOG2) {
            mult = 1;
        } else if (dec < LOG4) {
            mult = 2;
        } else if (dec < LOG6) {
            mult = 4;
        } else if (dec < LOG8) {
            mult = 6;
        } else {
            mult = 8;
        }
    }

    return mag * mult;
}

ForgeModule& forgePlugin() { return detail::forgeManager().plugin(); }

ForgeManager::ForgeManager() : mPlugin(new ForgeModule()) {}

ForgeModule& ForgeManager::plugin() { return *mPlugin; }

fg_window ForgeManager::getMainWindow() {
    static std::once_flag flag;

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    std::string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");

    af_err error      = AF_SUCCESS;
    fg_err forgeError = FG_ERR_NONE;
    if (noGraphicsENV.empty()) {  // If AF_DISABLE_GRAPHICS is not defined
        std::call_once(flag, [this, &error, &forgeError] {
            if (!this->mPlugin->isLoaded()) {
                error = AF_ERR_LOAD_LIB;
                return;
            }
            fg_window w = nullptr;
            forgeError  = this->mPlugin->fg_create_window(
                 &w, WIDTH, HEIGHT, "ArrayFire", NULL, true);
            if (forgeError != FG_ERR_NONE) { return; }
            this->setWindowChartGrid(w, 1, 1);
            this->mPlugin->fg_make_window_current(w);
            this->mMainWindow.reset(new Window({w}));
            if (!gladLoadGL()) { error = AF_ERR_LOAD_LIB; }
        });
        if (error == AF_ERR_LOAD_LIB) {
            string error_message =
                "Error loading Forge: " + this->mPlugin->getErrorMessage() +
                "\nForge or one of it's dependencies failed to "
                "load. Try installing Forge or check if Forge is in the "
                "search path.";
            AF_ERROR(error_message.c_str(), AF_ERR_LOAD_LIB);
        }
        if (forgeError != FG_ERR_NONE) {
            AF_ERROR(this->mPlugin->fg_err_to_string(forgeError),
                     AF_ERR_RUNTIME);
        }
    }

    return mMainWindow->handle;
}

fg_window ForgeManager::getWindow(const int w, const int h,
                                  const char* const title,
                                  const bool invisible) {
    fg_window retVal = 0;
    FG_CHECK(mPlugin->fg_create_window(&retVal, w, h, title, getMainWindow(),
                                       invisible));
    if (retVal == 0) { AF_ERROR("Window creation failed", AF_ERR_INTERNAL); }
    setWindowChartGrid(retVal, 1, 1);
    return retVal;
}

void ForgeManager::setWindowChartGrid(const fg_window window, const int r,
                                      const int c) {
    auto chart_iter = mChartMap.find(window);

    if (chart_iter != mChartMap.end()) {
        // ChartVec found. Clear it.
        // This has to be cleared as there is no guarantee that existing
        // chart types(2D/3D) match the future grid requirements
        for (const ChartPtr& c : chart_iter->second) {
            if (c) { mChartAxesOverrideMap.erase(c->handle); }
        }
        (chart_iter->second).clear();  // Clear ChartList
        auto gIter    = mWndGridMap.find(window);
        gIter->second = make_pair(1, 1);
    }

    if (r == 0 || c == 0) {
        mChartMap.erase(window);
        mWndGridMap.erase(window);
    } else {
        mChartMap[window]   = ChartList(r * c);
        mWndGridMap[window] = std::make_pair(r, c);
    }
}

ForgeManager::WindowGridDims ForgeManager::getWindowGrid(
    const fg_window window) {
    auto gIter = mWndGridMap.find(window);
    if (gIter == mWndGridMap.end()) { mWndGridMap[window] = make_pair(1, 1); }
    return mWndGridMap[window];
}

fg_chart ForgeManager::getChart(const fg_window window, const int r,
                                const int c, const fg_chart_type ctype) {
    auto gIter = mWndGridMap.find(window);

    int rows = std::get<0>(gIter->second);
    int cols = std::get<1>(gIter->second);

    if (c >= cols || r >= rows) {
        AF_ERROR("Window Grid points are out of bounds", AF_ERR_TYPE);
    }

    // upgrade to exclusive access to make changes
    auto chart_iter = mChartMap.find(window);
    ChartPtr& chart = (chart_iter->second)[c * rows + r];

    if (!chart) {
        fg_chart temp = NULL;
        FG_CHECK(mPlugin->fg_create_chart(&temp, ctype));
        chart.reset(new Chart({temp}));
        mChartAxesOverrideMap[chart->handle] = false;
    } else {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart->handle));
        if (chart_type != ctype) {
            // Existing chart is of incompatible type
            mChartAxesOverrideMap.erase(chart->handle);
            fg_chart temp = 0;
            FG_CHECK(mPlugin->fg_create_chart(&temp, ctype));
            chart.reset(new Chart({temp}));
            mChartAxesOverrideMap[chart->handle] = false;
        }
    }
    return chart->handle;
}

unsigned long long ForgeManager::genImageKey(unsigned w, unsigned h,
                                             fg_channel_format mode,
                                             fg_dtype type) {
    assert(w <= 2U << 16U);
    assert(h <= 2U << 16U);
    unsigned long long key = ((w & _16BIT) << 16U) | (h & _16BIT);
    key = ((((key << 16U) | (mode & _16BIT)) << 16U) | (type | _16BIT));
    return key;
}

fg_image ForgeManager::getImage(int w, int h, fg_channel_format mode,
                                fg_dtype type) {
    auto key = genImageKey(w, h, mode, type);

    ChartKey keypair = std::make_pair(key, nullptr);
    auto iter        = mImgMap.find(keypair);

    if (iter == mImgMap.end()) {
        fg_image img = nullptr;
        FG_CHECK(mPlugin->fg_create_image(&img, w, h, mode, type));
        mImgMap[keypair] = ImagePtr(new Image({img}));
    }
    return mImgMap[keypair]->handle;
}

fg_image ForgeManager::getImage(fg_chart chart, int w, int h,
                                fg_channel_format mode, fg_dtype type) {
    auto key = genImageKey(w, h, mode, type);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mImgMap.find(keypair);

    if (iter == mImgMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_2D) {
            AF_ERROR("Image can only be added to chart of type FG_CHART_2D",
                     AF_ERR_TYPE);
        }
        fg_image img = nullptr;
        FG_CHECK(mPlugin->fg_create_image(&img, w, h, mode, type));
        FG_CHECK(mPlugin->fg_append_image_to_chart(chart, img));

        mImgMap[keypair] = ImagePtr(new Image({img}));
    }
    return mImgMap[keypair]->handle;
}

fg_plot ForgeManager::getPlot(fg_chart chart, int nPoints, fg_dtype dtype,
                              fg_plot_type ptype, fg_marker_type mtype) {
    unsigned long long key =
        ((static_cast<unsigned long long>(nPoints) & _48BIT) << 16U);
    key |=
        (((dtype & _4BIT) << 12U) | ((ptype & _4BIT) << 8U) | (mtype & _8BIT));

    ChartKey keypair = std::make_pair(key, chart);
    auto iter        = mPltMap.find(keypair);

    if (iter == mPltMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));

        fg_plot plt = nullptr;
        FG_CHECK(mPlugin->fg_create_plot(&plt, nPoints, dtype, chart_type,
                                         ptype, mtype));
        FG_CHECK(mPlugin->fg_append_plot_to_chart(chart, plt));

        mPltMap[keypair] = PlotPtr(new Plot({plt}));
    }
    return mPltMap[keypair]->handle;
}

fg_histogram ForgeManager::getHistogram(fg_chart chart, int nBins,
                                        fg_dtype type) {
    unsigned long long key =
        ((static_cast<unsigned long long>(nBins) & _48BIT) << 16U) |
        (type & _16BIT);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mHstMap.find(keypair);

    if (iter == mHstMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_2D) {
            AF_ERROR("Histogram can only be added to chart of type FG_CHART_2D",
                     AF_ERR_TYPE);
        }
        fg_histogram hst = nullptr;
        FG_CHECK(mPlugin->fg_create_histogram(&hst, nBins, type));
        FG_CHECK(mPlugin->fg_append_histogram_to_chart(chart, hst));
        mHstMap[keypair] = HistogramPtr(new Histogram({hst}));
    }
    return mHstMap[keypair]->handle;
}

fg_surface ForgeManager::getSurface(fg_chart chart, int nX, int nY,
                                    fg_dtype type) {
    unsigned long long surfaceSize = nX * static_cast<unsigned long long>(nY);
    assert(surfaceSize <= 2ULL << 48ULL);
    unsigned long long key = ((surfaceSize & _48BIT) << 16U) | (type & _16BIT);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mSfcMap.find(keypair);

    if (iter == mSfcMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_3D) {
            AF_ERROR("Surface can only be added to chart of type FG_CHART_3D",
                     AF_ERR_TYPE);
        }
        fg_surface surf = nullptr;
        FG_CHECK(mPlugin->fg_create_surface(&surf, nX, nY, type,
                                            FG_PLOT_SURFACE, FG_MARKER_NONE));
        FG_CHECK(mPlugin->fg_append_surface_to_chart(chart, surf));
        mSfcMap[keypair] = SurfacePtr(new Surface({surf}));
    }
    return mSfcMap[keypair]->handle;
}

fg_vector_field ForgeManager::getVectorField(fg_chart chart, int nPoints,
                                             fg_dtype type) {
    unsigned long long key =
        ((static_cast<unsigned long long>(nPoints) & _48BIT) << 16U) |
        (type & _16BIT);

    ChartKey keypair = make_pair(key, chart);
    auto iter        = mVcfMap.find(keypair);

    if (iter == mVcfMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));

        fg_vector_field vfield = nullptr;
        FG_CHECK(mPlugin->fg_create_vector_field(&vfield, nPoints, type,
                                                 chart_type));
        FG_CHECK(mPlugin->fg_append_vector_field_to_chart(chart, vfield));
        mVcfMap[keypair] = VectorFieldPtr(new VectorField({vfield}));
    }
    return mVcfMap[keypair]->handle;
}

bool ForgeManager::getChartAxesOverride(const fg_chart chart) {
    auto iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        AF_ERROR("Chart Not Found!", AF_ERR_INTERNAL);
    }
    return mChartAxesOverrideMap[chart];
}

void ForgeManager::setChartAxesOverride(const fg_chart chart, bool flag) {
    auto iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        AF_ERROR("Chart Not Found!", AF_ERR_INTERNAL);
    }
    mChartAxesOverrideMap[chart] = flag;
}

}  // namespace common
}  // namespace arrayfire
