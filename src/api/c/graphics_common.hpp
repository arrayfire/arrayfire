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

#include <map>

// default to f32(float) type
template<typename T>
fg::dtype getGLType();

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.
GLenum glErrorSkip(const char *msg, const char* file, int line);
GLenum glErrorCheck(const char *msg, const char* file, int line);
GLenum glForceErrorCheck(const char *msg, const char* file, int line);

#define CheckGL(msg)      glErrorCheck     (msg, __AF_FILENAME__, __LINE__)
#define ForceCheckGL(msg) glForceErrorCheck(msg, __AF_FILENAME__, __LINE__)
#define CheckGLSkip(msg)  glErrorSkip      (msg, __AF_FILENAME__, __LINE__)

fg::MarkerType getFGMarker(const af_marker_type af_marker);
namespace graphics
{

enum Defaults {
    WIDTH = 1280,
    HEIGHT= 720
};

static const long long _16BIT = 0x000000000000FFFF;
static const long long _32BIT = 0x00000000FFFFFFFF;
static const long long _48BIT = 0x0000FFFFFFFFFFFF;

typedef std::map<long long, fg::Image*> ImageMap_t;
typedef std::map<long long, fg::Plot*> PlotMap_t;
typedef std::map<long long, fg::Histogram*> HistogramMap_t;
typedef std::map<long long, fg::Plot3*> Plot3Map_t;
typedef std::map<long long, fg::Surface*> SurfaceMap_t;

typedef ImageMap_t::iterator ImgMapIter;
typedef PlotMap_t::iterator PltMapIter;
typedef Plot3Map_t::iterator Plt3MapIter;
typedef HistogramMap_t::iterator HstMapIter;
typedef SurfaceMap_t::iterator SfcMapIter;

/**
 * ForgeManager class follows a single pattern. Any user of this class, has
 * to call ForgeManager::getInstance inorder to use Forge resources for rendering.
 * It manages the windows, and other renderables (given below) that are drawed
 * onto chosen window.
 * Renderables:
 *             fg::Image
 *             fg::Plot
 *             fg::Plot3
 *             fg::Histogram
 *             fg::Surface
 * */
class ForgeManager
{
    private:
        ImageMap_t      mImgMap;
        PlotMap_t       mPltMap;
        Plot3Map_t      mPlt3Map;
        HistogramMap_t  mHstMap;
        SurfaceMap_t    mSfcMap;

    public:
        static ForgeManager& getInstance();
        ~ForgeManager();

        fg::Font* getFont(const bool dontCreate=false);
        fg::Window* getMainWindow(const bool dontCreate=false);
        fg::Image* getImage(int w, int h, fg::ChannelFormat mode, fg::dtype type);
        fg::Plot* getPlot(int nPoints, fg::dtype dtype, fg::PlotType ptype, fg::MarkerType mtype);
        fg::Plot3* getPlot3(int nPoints, fg::dtype dtype,fg::PlotType ptype, fg::MarkerType mtype);
        fg::Histogram* getHistogram(int nBins, fg::dtype type);
        fg::Surface* getSurface(int nX, int nY, fg::dtype type);

    protected:
        ForgeManager() {}
        ForgeManager(ForgeManager const&);
        void operator=(ForgeManager const&);
        void destroyResources();
};

}

#define MAIN_WINDOW graphics::ForgeManager::getInstance().getMainWindow(true)

#endif
