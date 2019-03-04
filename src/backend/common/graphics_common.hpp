/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/forge_loader.hpp>
#include <af/graphics.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

// default to f32(float) type
template<typename T>
fg_dtype getGLType();

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.
GLenum glErrorCheck(const char* msg, const char* file, int line);

#define CheckGL(msg) glErrorCheck(msg, __AF_FILENAME__, __LINE__)

fg_marker_type getFGMarker(const af_marker_type af_marker);

void makeContextCurrent(fg_window window);

double step_round(const double in, const bool dir);

namespace graphics {

/**
 * Only device manager class can create objects of this class.
 * You have to call forgeManager() defined in platform.hpp to
 * access the object. It manages the windows, and other
 * renderables (given below) that are drawed onto chosen window.
 * Renderables:
 *      fg_image
 *      fg_plot
 *      fg_histogram
 *      fg_surface
 *      fg_vector_field
 * */
class ForgeManager {
  public:
    using WindowGridDims = std::pair<int, int>;

    ForgeManager();
    ForgeManager(ForgeManager const&) = delete;
    ForgeManager& operator=(ForgeManager const&) = delete;
    ForgeManager(ForgeManager&&)                 = delete;
    ForgeManager& operator=(ForgeManager&&) = delete;

    ForgeModule& plugin();

    fg_window getMainWindow();

    fg_window getWindow(const int width, const int height, const char* const title,
                        const bool invisible=false);

    void setWindowChartGrid(const fg_window window, const int r, const int c);

    WindowGridDims getWindowGrid(const fg_window window);

    fg_chart getChart(const fg_window window, const int r, const int c,
                      const fg_chart_type ctype);

    fg_image getImage(int w, int h, fg_channel_format mode, fg_dtype type);

    fg_image getImage(fg_chart chart, int w, int h,
                      fg_channel_format mode, fg_dtype type);

    fg_plot getPlot(fg_chart chart, int nPoints, fg_dtype dtype,
                    fg_plot_type ptype, fg_marker_type mtype);

    fg_histogram getHistogram(fg_chart chart, int nBins, fg_dtype type);

    fg_surface getSurface(fg_chart chart, int nX, int nY, fg_dtype type);

    fg_vector_field getVectorField(fg_chart chart, int nPoints, fg_dtype type);

    bool getChartAxesOverride(const fg_chart chart);
    void setChartAxesOverride(const fg_chart chart, bool flag = true);

  private:
    constexpr static unsigned int WIDTH = 1280;
    constexpr static unsigned int HEIGHT = 720;
    constexpr static long long _16BIT = 0x000000000000FFFF;
    constexpr static long long _32BIT = 0x00000000FFFFFFFF;
    constexpr static long long _48BIT = 0x0000FFFFFFFFFFFF;

#define DEFINE_WRAPPER_OBJECT(Object, ReleaseFunc)          \
struct Object {                                             \
    void* handle;                                           \
    struct Deleter {                                        \
        void operator()(Object* pHandle) const {            \
            if (pHandle) {                                  \
                forgePlugin().ReleaseFunc(pHandle->handle); \
            }                                               \
        }                                                   \
    };                                                      \
}

DEFINE_WRAPPER_OBJECT(Window     , fg_release_window      );
DEFINE_WRAPPER_OBJECT(Image      , fg_release_image       );
DEFINE_WRAPPER_OBJECT(Chart      , fg_release_chart       );
DEFINE_WRAPPER_OBJECT(Plot       , fg_release_plot        );
DEFINE_WRAPPER_OBJECT(Histogram  , fg_release_histogram   );
DEFINE_WRAPPER_OBJECT(Surface    , fg_release_surface     );
DEFINE_WRAPPER_OBJECT(VectorField, fg_release_vector_field);

#undef DEFINE_WRAPPER_OBJECT

    using ImagePtr        = std::unique_ptr<Image      , Image::Deleter      >;
    using ChartPtr        = std::unique_ptr<Chart      , Chart::Deleter      >;
    using PlotPtr         = std::unique_ptr<Plot       , Plot::Deleter       >;
    using SurfacePtr      = std::unique_ptr<Surface    , Surface::Deleter    >;
    using HistogramPtr    = std::unique_ptr<Histogram  , Histogram::Deleter  >;
    using VectorFieldPtr  = std::unique_ptr<VectorField, VectorField::Deleter>;
    using ChartList       = std::vector<ChartPtr>;
    using ChartKey        = std::pair<long long, fg_chart>;

    using ChartMapIterator     = std::map<fg_window, ChartList>::iterator;
    using WindGridMapIterator  = std::map<fg_window, WindowGridDims>::iterator;
    using AxesOverrideIterator = std::map<fg_chart, bool>::iterator;
    using ImageMapIterator     = std::map<ChartKey, ImagePtr>::iterator;
    using PlotMapIterator      = std::map<ChartKey, PlotPtr>::iterator;
    using HistogramMapIterator = std::map<ChartKey, HistogramPtr>::iterator;
    using SurfaceMapIterator   = std::map<ChartKey, SurfacePtr>::iterator;
    using VecFieldMapIterator  = std::map<ChartKey, VectorFieldPtr>::iterator;

    std::unique_ptr<ForgeModule> mPlugin;
    std::unique_ptr<Window, Window::Deleter> mMainWindow;

    std::map<fg_window, ChartList     > mChartMap;
    std::map< ChartKey, ImagePtr      > mImgMap;
    std::map< ChartKey, PlotPtr       > mPltMap;
    std::map< ChartKey, HistogramPtr  > mHstMap;
    std::map< ChartKey, SurfacePtr    > mSfcMap;
    std::map< ChartKey, VectorFieldPtr> mVcfMap;
    std::map<fg_window, WindowGridDims> mWndGridMap;
    std::map< fg_chart, bool          > mChartAxesOverrideMap;
};

}  // namespace graphics
