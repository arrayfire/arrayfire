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

using namespace std;

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
        FG_MODULE_FUNCTION_INIT(fg_append_image_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_plot_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_histogram_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_surface_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_append_vector_field_to_chart);
        FG_MODULE_FUNCTION_INIT(fg_release_chart);

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

GLenum glErrorCheck(const char* msg, const char* file, int line) {
// Skipped in release mode
#ifndef NDEBUG
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n",
                file, line, msg, (int)x, glGetString(x));
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
#else
    return (GLenum)0;
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
    FG_CHECK(graphics::forgePlugin().fg_make_window_current(window));
    CheckGL("End makeContextCurrent");
}

// dir -> true = round up, false = round down
double step_round(const double in, const bool dir) {
    if (in == 0) return 0;

    static const double __log2 = log10(2);
    static const double __log4 = log10(4);
    static const double __log6 = log10(6);
    static const double __log8 = log10(8);

    // log_in is of the form "s abc.xyz", where
    // s is either + or -; + indicates abs(in) >= 1 and - indicates 0 < abs(in)
    // < 1 (log10(1) is +0)
    const double sign   = in < 0 ? -1 : 1;
    const double log_in = std::log10(std::fabs(in));
    const double mag    = std::pow(10, std::floor(log_in)) *
                       sign;  // Number of digits either left or right of 0
    const double dec = std::log10(in / mag);  // log of the fraction

    // This means in is of the for 10^n
    if (dec == 0) return in;

    // For negative numbers, -ve round down = +ve round up and vice versa
    bool op_dir = in > 0 ? dir : !dir;

    double mult = 1;

    // Round up
    if (op_dir) {
        if (dec <= __log2) {
            mult = 2;
        } else if (dec <= __log4) {
            mult = 4;
        } else if (dec <= __log6) {
            mult = 6;
        } else if (dec <= __log8) {
            mult = 8;
        } else {
            mult = 10;
        }
    } else {  // Round down
        if (dec < __log2) {
            mult = 1;
        } else if (dec < __log4) {
            mult = 2;
        } else if (dec < __log6) {
            mult = 4;
        } else if (dec < __log8) {
            mult = 6;
        } else {
            mult = 8;
        }
    }

    return mag * mult;
}

namespace graphics {

ForgeModule& forgePlugin() { return detail::forgeManager().plugin(); }

ForgeManager::ForgeManager() : mPlugin(new ForgeModule()) {}

ForgeManager::~ForgeManager() {
    /* clear all OpenGL resource objects (images, plots, histograms etc) first
     * and then delete the windows */
    for (ImgMapIter iter = mImgMap.begin(); iter != mImgMap.end(); iter++)
        mPlugin->fg_release_image(iter->second);

    for (PltMapIter iter = mPltMap.begin(); iter != mPltMap.end(); iter++)
        mPlugin->fg_release_plot(iter->second);

    for (HstMapIter iter = mHstMap.begin(); iter != mHstMap.end(); iter++)
        mPlugin->fg_release_histogram(iter->second);

    for (SfcMapIter iter = mSfcMap.begin(); iter != mSfcMap.end(); iter++)
        mPlugin->fg_release_surface(iter->second);

    for (VcfMapIter iter = mVcfMap.begin(); iter != mVcfMap.end(); iter++)
        mPlugin->fg_release_vector_field(iter->second);

    for (ChartMapIter iter = mChartMap.begin(); iter != mChartMap.end();
         iter++) {
        for (int i = 0; i < (int)(iter->second).size(); i++) {
            fg_chart chrt = (iter->second)[i];
            if (chrt) {
                mChartAxesOverrideMap.erase((chrt));
                mPlugin->fg_release_chart(chrt);
            }
        }
    }
    mPlugin->fg_release_window(wnd->handle);
}

ForgeModule& ForgeManager::plugin() { return *mPlugin; }

fg_window ForgeManager::getMainWindow() {
    static std::once_flag flag;

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    std::string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");

    if (noGraphicsENV.empty()) {  // If AF_DISABLE_GRAPHICS is not defined
        std::call_once(flag, [this] {
            if (!this->mPlugin->isLoaded()) {
                string error_message =
                    "Error loading Forge: " + this->mPlugin->getErrorMessage() +
                    "\nForge or one of it's dependencies failed to "
                    "load. Try installing Forge or check if Forge is in the "
                    "search path.";
                AF_ERROR(error_message.c_str(), AF_ERR_LOAD_LIB);
            }
            fg_window w = nullptr;
            fg_err e    = this->mPlugin->fg_create_window(&w, WIDTH, HEIGHT,
                                                       "ArrayFire", NULL, true);
            if (e != FG_ERR_NONE) {
                AF_ERROR("Graphics Window creation failed", AF_ERR_INTERNAL);
            }
            this->mPlugin->fg_make_window_current(w);
            this->setWindowChartGrid(w, 1, 1);
            this->wnd.reset(new Window({w}));
            if (!gladLoadGL()) { AF_ERROR("GL Load Failed", AF_ERR_LOAD_LIB); }
        });
    }

    return wnd->handle;
}

void ForgeManager::setWindowChartGrid(const fg_window window, const int r,
                                      const int c) {
    ChartMapIter iter = mChartMap.find(window);
    GridMapIter gIter = mWndGridMap.find(window);

    if (iter != mChartMap.end()) {
        // ChartVec found. Clear it.
        // This has to be cleared as there is no guarantee that existing
        // chart types(2D/3D) match the future grid requirements
        for (int i = 0; i < (int)(iter->second).size(); i++) {
            fg_chart chrt = (iter->second)[i];
            if (chrt) {
                mChartAxesOverrideMap.erase(chrt);
                FG_CHECK(mPlugin->fg_release_chart(chrt));
            }
        }
        (iter->second).clear();
        gIter->second = std::make_pair<int, int>(1, 1);
    }

    if (r == 0 || c == 0) {
        mChartMap.erase(window);
        mWndGridMap.erase(window);
    } else {
        mChartMap[window]   = std::vector<fg_chart>(r * c);
        mWndGridMap[window] = std::make_pair(r, c);
    }
}

WindGridDims_t ForgeManager::getWindowGrid(const fg_window window) {
    GridMapIter gIter = mWndGridMap.find(window);

    if (gIter == mWndGridMap.end()) {
        mWndGridMap[window] = std::make_pair(1, 1);
    }

    return mWndGridMap[window];
}

fg_chart ForgeManager::getChart(const fg_window window, const int r,
                                const int c, const fg_chart_type ctype) {
    fg_chart chart    = NULL;
    ChartMapIter iter = mChartMap.find(window);
    GridMapIter gIter = mWndGridMap.find(window);

    if (iter != mChartMap.end()) {
        int gRows = std::get<0>(gIter->second);
        int gCols = std::get<1>(gIter->second);

        if (c >= gCols || r >= gRows)
            AF_ERROR("Grid points are out of bounds", AF_ERR_TYPE);

        // upgrade to exclusive access to make changes
        chart = (iter->second)[c * gRows + r];

        if (chart == NULL) {
            // Chart has not been created
            FG_CHECK(mPlugin->fg_create_chart(&chart, ctype));
            (iter->second)[c * gRows + r] = chart;
            // Set Axes override to false
            mChartAxesOverrideMap[chart] = false;
        } else {
            fg_chart_type chart_type;
            FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
            if (chart_type != ctype) {
                // Existing chart is of incompatible type
                mChartAxesOverrideMap.erase(chart);
                FG_CHECK(mPlugin->fg_release_chart(chart));
                FG_CHECK(mPlugin->fg_create_chart(&chart, ctype));
                (iter->second)[c * gRows + r] = chart;
                // Set Axes override to false
                mChartAxesOverrideMap[chart] = false;
            }
        }
    } else {
        // The chart map for this was never created
        // Which should never happen
    }

    return chart;
}

fg_image ForgeManager::getImage(int w, int h, fg_channel_format mode,
                                fg_dtype type) {
    /* w, h needs to fall in the range of [0, 2^16]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Image object. So, this implementation
     * is a limitation on how big of an image can be rendered
     * using arrayfire graphics funtionality */
    assert(w <= 2ll << 16);
    assert(h <= 2ll << 16);
    long long key = ((w & _16BIT) << 16) | (h & _16BIT);
    key           = (((key << 16) | mode) << 16) | type;

    ChartKey_t keypair = std::make_pair(key, nullptr);

    ImgMapIter iter = mImgMap.find(keypair);

    if (iter == mImgMap.end()) {
        fg_image img = nullptr;
        FG_CHECK(mPlugin->fg_create_image(&img, w, h, mode, type));
        mImgMap[keypair] = img;
    }

    return mImgMap[keypair];
}

fg_image ForgeManager::getImage(fg_chart chart, int w, int h,
                                fg_channel_format mode, fg_dtype type) {
    /* w, h needs to fall in the range of [0, 2^16]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Image object. So, this implementation
     * is a limitation on how big of an image can be rendered
     * using arrayfire graphics funtionality */
    assert(w <= 2ll << 16);
    assert(h <= 2ll << 16);
    long long key = ((w & _16BIT) << 16) | (h & _16BIT);
    key           = (((key << 16) | mode) << 16) | type;

    ChartKey_t keypair = std::make_pair(key, chart);

    ImgMapIter iter = mImgMap.find(keypair);

    if (iter == mImgMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_2D)
            AF_ERROR("Image can only be added to chart of type FG_CHART_2D",
                     AF_ERR_TYPE);

        fg_image img = nullptr;
        FG_CHECK(mPlugin->fg_create_image(&img, w, h, mode, type));
        mImgMap[keypair] = img;

        FG_CHECK(mPlugin->fg_append_image_to_chart(chart, img));
    }

    return mImgMap[keypair];
}

fg_plot ForgeManager::getPlot(fg_chart chart, int nPoints, fg_dtype dtype,
                              fg_plot_type ptype, fg_marker_type mtype) {
    long long key = ((nPoints & _48BIT) << 48);
    key |= (((((dtype & 0x000F) << 12) | (ptype & 0x000F)) << 8) |
            (mtype & 0x000F));

    ChartKey_t keypair = std::make_pair(key, chart);

    PltMapIter iter = mPltMap.find(keypair);

    if (iter == mPltMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));

        fg_plot plt = nullptr;
        FG_CHECK(mPlugin->fg_create_plot(&plt, nPoints, dtype, chart_type,
                                         ptype, mtype));
        mPltMap[keypair] = plt;

        FG_CHECK(mPlugin->fg_append_plot_to_chart(chart, plt));
    }

    return mPltMap[keypair];
}

fg_histogram ForgeManager::getHistogram(fg_chart chart, int nBins,
                                        fg_dtype type) {
    long long key = ((nBins & _48BIT) << 48) | (type & _16BIT);

    ChartKey_t keypair = std::make_pair(key, chart);

    HstMapIter iter = mHstMap.find(keypair);

    if (iter == mHstMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_2D)
            AF_ERROR("Histogram can only be added to chart of type FG_CHART_2D",
                     AF_ERR_TYPE);

        fg_histogram hst = nullptr;
        FG_CHECK(mPlugin->fg_create_histogram(&hst, nBins, type));
        mHstMap[keypair] = hst;

        FG_CHECK(mPlugin->fg_append_histogram_to_chart(chart, hst));
    }

    return mHstMap[keypair];
}

fg_surface ForgeManager::getSurface(fg_chart chart, int nX, int nY,
                                    fg_dtype type) {
    /* nX * nY needs to fall in the range of [0, 2^48]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Plot object. So, this implementation
     * is a limitation on how big of an plot graph can be rendered
     * using arrayfire graphics funtionality */
    assert((long long)nX * nY <= 2ll << 48);
    long long key = (((nX * nY) & _48BIT) << 48) | (type & _16BIT);

    ChartKey_t keypair = std::make_pair(key, chart);

    SfcMapIter iter = mSfcMap.find(keypair);

    if (iter == mSfcMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));
        if (chart_type != FG_CHART_3D)
            AF_ERROR("Surface can only be added to chart of type FG_CHART_3D",
                     AF_ERR_TYPE);

        fg_surface surf = nullptr;
        FG_CHECK(mPlugin->fg_create_surface(&surf, nX, nY, type,
                                            FG_PLOT_SURFACE, FG_MARKER_NONE));
        mSfcMap[keypair] = surf;

        FG_CHECK(mPlugin->fg_append_surface_to_chart(chart, surf));
    }

    return mSfcMap[keypair];
}

fg_vector_field ForgeManager::getVectorField(fg_chart chart, int nPoints,
                                             fg_dtype type) {
    long long key = (((nPoints)&_48BIT) << 48) | (type & _16BIT);

    ChartKey_t keypair = std::make_pair(key, chart);

    VcfMapIter iter = mVcfMap.find(keypair);

    if (iter == mVcfMap.end()) {
        fg_chart_type chart_type;
        FG_CHECK(mPlugin->fg_get_chart_type(&chart_type, chart));

        fg_vector_field vfield = nullptr;
        FG_CHECK(mPlugin->fg_create_vector_field(&vfield, nPoints, type,
                                                 chart_type));
        mVcfMap[keypair] = vfield;

        FG_CHECK(mPlugin->fg_append_vector_field_to_chart(chart, vfield));
    }

    return mVcfMap[keypair];
}

bool ForgeManager::getChartAxesOverride(fg_chart chart) {
    ChartAxesOverrideIter iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        AF_ERROR("Chart Not Found!", AF_ERR_INTERNAL);
    }
    return mChartAxesOverrideMap[chart];
}

void ForgeManager::setChartAxesOverride(fg_chart chart, bool flag) {
    ChartAxesOverrideIter iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        AF_ERROR("Chart Not Found!", AF_ERR_INTERNAL);
    }
    mChartAxesOverrideMap[chart] = flag;
}
}  // namespace graphics
