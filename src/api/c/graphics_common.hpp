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

namespace graphics
{

enum Defaults {
    WIDTH = 1280,
    HEIGHT= 720
};

static const long long _16BIT = 0x000000000000FFFF;
static const long long _32BIT = 0x00000000FFFFFFFF;
static const long long _48BIT = 0x0000FFFFFFFFFFFF;

typedef std::map<long long, forge::Image*> ImageMap_t;
typedef std::map<long long, forge::Plot*> PlotMap_t;
typedef std::map<long long, forge::Histogram*> HistogramMap_t;
typedef std::map<long long, forge::Surface*> SurfaceMap_t;

typedef ImageMap_t::iterator ImgMapIter;
typedef PlotMap_t::iterator PltMapIter;
typedef HistogramMap_t::iterator HstMapIter;
typedef SurfaceMap_t::iterator SfcMapIter;

typedef std::vector<forge::Chart*> ChartVec_t;
typedef std::map<const forge::Window*, ChartVec_t> ChartMap_t;
typedef ChartVec_t::iterator ChartVecIter;
typedef ChartMap_t::iterator ChartMapIter;


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
 * */
class ForgeManager
{
    private:
        ImageMap_t      mImgMap;
        PlotMap_t       mPltMap;
        HistogramMap_t  mHstMap;
        SurfaceMap_t    mSfcMap;

        ChartMap_t      mChartMap;

    public:
        static ForgeManager& getInstance();
        ~ForgeManager();

        forge::Font*    getFont(const bool dontCreate=false);
        forge::Window*  getMainWindow(const bool dontCreate=false);

        void            setWindowChartGrid(const forge::Window* window,
                                           const int r, const int c);

        forge::Chart*   getChart(const forge::Window* window, const int r, const int c,
                                 const forge::ChartType ctype);

        forge::Image* getImage          (int w, int h, forge::ChannelFormat mode,
                                         forge::dtype type);
        forge::Image* getImage          (forge::Chart* chart, int w, int h,
                                         forge::ChannelFormat mode, forge::dtype type);
        forge::Plot * getPlot           (forge::Chart* chart, int nPoints, forge::dtype dtype,
                                         forge::PlotType ptype, forge::MarkerType mtype);
        forge::Histogram* getHistogram  (forge::Chart* chart, int nBins, forge::dtype type);
        forge::Surface* getSurface      (forge::Chart* chart, int nX, int nY, forge::dtype type);

    protected:
        ForgeManager() {}
        ForgeManager(ForgeManager const&);
        void operator=(ForgeManager const&);
        void destroyResources();
};

}

#define MAIN_WINDOW graphics::ForgeManager::getInstance().getMainWindow(true)

#endif
