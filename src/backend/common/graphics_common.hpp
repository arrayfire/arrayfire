/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#if defined(WITH_GRAPHICS)

#include <af/graphics.h>

#include <forge.h>
#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#include <vector>
#include <map>
#include <utility>

// default to f32(float) type
template<typename T>
forge::dtype getGLType();

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.
gl::GLenum glErrorSkip(const char *msg, const char* file, int line);
gl::GLenum glErrorCheck(const char *msg, const char* file, int line);
gl::GLenum glForceErrorCheck(const char *msg, const char* file, int line);

#define CheckGL(msg)      glErrorCheck     (msg, __AF_FILENAME__, __LINE__)
#define ForceCheckGL(msg) glForceErrorCheck(msg, __AF_FILENAME__, __LINE__)
#define CheckGLSkip(msg)  glErrorSkip      (msg, __AF_FILENAME__, __LINE__)

forge::MarkerType getFGMarker(const af_marker_type af_marker);

void makeContextCurrent(forge::Window *window);

double step_round(const double in, const bool dir);

namespace graphics
{
enum Defaults {
    WIDTH = 1280,
    HEIGHT= 720
};

static const long long _16BIT = 0x000000000000FFFF;
static const long long _32BIT = 0x00000000FFFFFFFF;
static const long long _48BIT = 0x0000FFFFFFFFFFFF;

typedef std::pair<long long, forge::Chart*> ChartKey_t;

typedef std::map<ChartKey_t, forge::Image*      > ImageMap_t;
typedef std::map<ChartKey_t, forge::Plot*       > PlotMap_t;
typedef std::map<ChartKey_t, forge::Histogram*  > HistogramMap_t;
typedef std::map<ChartKey_t, forge::Surface*    > SurfaceMap_t;
typedef std::map<ChartKey_t, forge::VectorField*> VectorFieldMap_t;

typedef ImageMap_t::iterator ImgMapIter;
typedef PlotMap_t::iterator PltMapIter;
typedef HistogramMap_t::iterator HstMapIter;
typedef SurfaceMap_t::iterator SfcMapIter;
typedef VectorFieldMap_t::iterator VcfMapIter;

typedef std::vector<forge::Chart*> ChartVec_t;
typedef std::map<const forge::Window*, ChartVec_t> ChartMap_t;
typedef std::pair<int, int> WindGridDims_t;
typedef std::map<const forge::Window*, WindGridDims_t> WindGridMap_t;
typedef ChartVec_t::iterator ChartVecIter;
typedef ChartMap_t::iterator ChartMapIter;
typedef WindGridMap_t::iterator GridMapIter;

// Keeps track of which charts have manually assigned axes limits
typedef std::map<forge::Chart*, bool> ChartAxesOverride_t;
typedef ChartAxesOverride_t::iterator ChartAxesOverrideIter;

/**
 * ForgeManager class follows a single pattern. Any user of this class, has
 * to call ForgeManager::getInstance inorder to use Forge resources for rendering.
 * It manages the windows, and other renderables (given below) that are drawed
 * onto chosen window.
 * Renderables:
 *             forge::Image
 *             forge::Plot
 *             forge::Histogram
 *             forge::Surface
 *             forge::VectorField
 * */
class ForgeManager
{
    private:
        ImageMap_t          mImgMap;
        PlotMap_t           mPltMap;
        HistogramMap_t      mHstMap;
        SurfaceMap_t        mSfcMap;
        VectorFieldMap_t    mVcfMap;

        ChartMap_t          mChartMap;
        WindGridMap_t       mWndGridMap;
        ChartAxesOverride_t mChartAxesOverrideMap;

    public:
        static ForgeManager& getInstance();
        ~ForgeManager();

        forge::Font*    getFont();
        forge::Window*  getMainWindow();

        void            setWindowChartGrid(const forge::Window* window,
                                           const int r, const int c);

        WindGridDims_t getWindowGrid(const forge::Window* window);
        forge::Chart*   getChart(const forge::Window* window, const int r, const int c,
                                 const forge::ChartType ctype);

        forge::Image*       getImage        (int w, int h, forge::ChannelFormat mode,
                                             forge::dtype type);
        forge::Image*       getImage        (forge::Chart* chart, int w, int h,
                                             forge::ChannelFormat mode, forge::dtype type);
        forge::Plot *       getPlot         (forge::Chart* chart, int nPoints, forge::dtype dtype,
                                             forge::PlotType ptype, forge::MarkerType mtype);
        forge::Histogram*   getHistogram    (forge::Chart* chart, int nBins, forge::dtype type);
        forge::Surface*     getSurface      (forge::Chart* chart, int nX, int nY, forge::dtype type);
        forge::VectorField* getVectorField  (forge::Chart* chart, int nPoints, forge::dtype type);

        bool getChartAxesOverride(forge::Chart* chart);
        void setChartAxesOverride(forge::Chart* chart, bool flag = true);

    protected:
        ForgeManager() {}
        ForgeManager(ForgeManager const&);
        void operator=(ForgeManager const&);
};
}

#define MAIN_WINDOW graphics::ForgeManager::getInstance().getMainWindow()

#endif
