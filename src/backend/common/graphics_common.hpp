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
enum Defaults { WIDTH = 1280, HEIGHT = 720 };

static const long long _16BIT = 0x000000000000FFFF;
static const long long _32BIT = 0x00000000FFFFFFFF;
static const long long _48BIT = 0x0000FFFFFFFFFFFF;

typedef std::pair<long long, fg_chart> ChartKey_t;

typedef std::map<ChartKey_t, fg_image> ImageMap_t;
typedef std::map<ChartKey_t, fg_plot> PlotMap_t;
typedef std::map<ChartKey_t, fg_histogram> HistogramMap_t;
typedef std::map<ChartKey_t, fg_surface> SurfaceMap_t;
typedef std::map<ChartKey_t, fg_vector_field> VectorFieldMap_t;

typedef ImageMap_t::iterator ImgMapIter;
typedef PlotMap_t::iterator PltMapIter;
typedef HistogramMap_t::iterator HstMapIter;
typedef SurfaceMap_t::iterator SfcMapIter;
typedef VectorFieldMap_t::iterator VcfMapIter;

typedef std::vector<fg_chart> ChartVec_t;
typedef std::map<const fg_window, ChartVec_t> ChartMap_t;
typedef std::pair<int, int> WindGridDims_t;
typedef std::map<const fg_window, WindGridDims_t> WindGridMap_t;
typedef ChartVec_t::iterator ChartVecIter;
typedef ChartMap_t::iterator ChartMapIter;
typedef WindGridMap_t::iterator GridMapIter;

// Keeps track of which charts have manually assigned axes limits
typedef std::map<fg_chart, bool> ChartAxesOverride_t;
typedef ChartAxesOverride_t::iterator ChartAxesOverrideIter;

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
    struct Window {
        fg_window handle;
    };

   private:
    ForgeModule* mPlugin;
    std::unique_ptr<Window> wnd;

    ImageMap_t mImgMap;
    PlotMap_t mPltMap;
    HistogramMap_t mHstMap;
    SurfaceMap_t mSfcMap;
    VectorFieldMap_t mVcfMap;

    ChartMap_t mChartMap;
    WindGridMap_t mWndGridMap;
    ChartAxesOverride_t mChartAxesOverrideMap;

   public:
    ForgeManager();
    ForgeManager(ForgeManager const&) = delete;
    ForgeManager& operator=(ForgeManager const&) = delete;
    ForgeManager(ForgeManager&&)                 = delete;
    ForgeManager& operator=(ForgeManager&&) = delete;
    ~ForgeManager();
    ForgeModule& plugin();
    fg_window getMainWindow();

    void setWindowChartGrid(const fg_window window, const int r, const int c);

    WindGridDims_t getWindowGrid(const fg_window window);

    fg_chart getChart(const fg_window window, const int r, const int c,
                      const fg_chart_type ctype);

    fg_image getImage(int w, int h, fg_channel_format mode, fg_dtype type);

    fg_image getImage(fg_chart chart, int w, int h, fg_channel_format mode,
                      fg_dtype type);

    fg_plot getPlot(fg_chart chart, int nPoints, fg_dtype dtype,
                    fg_plot_type ptype, fg_marker_type mtype);

    fg_histogram getHistogram(fg_chart chart, int nBins, fg_dtype type);

    fg_surface getSurface(fg_chart chart, int nX, int nY, fg_dtype type);

    fg_vector_field getVectorField(fg_chart chart, int nPoints, fg_dtype type);

    bool getChartAxesOverride(fg_chart chart);
    void setChartAxesOverride(fg_chart chart, bool flag = true);
};
}  // namespace graphics
