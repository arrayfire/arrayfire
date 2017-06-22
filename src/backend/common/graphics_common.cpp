/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <common/graphics_common.hpp>
#include <glbinding/Meta.h>
#include <common/err_common.hpp>
#include <backend.hpp>
#include <platform.hpp>
#include <common/util.hpp>
#include <mutex>
#include <utility>

using namespace std;
using namespace gl;

template<typename T>
gl::GLenum getGLType() { return GL_FLOAT; }

forge::MarkerType getFGMarker(const af_marker_type af_marker) {
    forge::MarkerType fg_marker;
    switch (af_marker) {
        case AF_MARKER_NONE     : fg_marker = FG_MARKER_NONE;        break;
        case AF_MARKER_POINT    : fg_marker = FG_MARKER_POINT;       break;
        case AF_MARKER_CIRCLE   : fg_marker = FG_MARKER_CIRCLE;      break;
        case AF_MARKER_SQUARE   : fg_marker = FG_MARKER_SQUARE;      break;
        case AF_MARKER_TRIANGLE : fg_marker = FG_MARKER_TRIANGLE;    break;
        case AF_MARKER_CROSS    : fg_marker = FG_MARKER_CROSS;       break;
        case AF_MARKER_PLUS     : fg_marker = FG_MARKER_PLUS;        break;
        case AF_MARKER_STAR     : fg_marker = FG_MARKER_STAR;        break;
        default                 : fg_marker = FG_MARKER_NONE;        break;
    }
    return fg_marker;
}

#define INSTANTIATE_GET_FG_TYPE(T, ForgeEnum)\
    template<> forge::dtype getGLType<T>() { return ForgeEnum; }

INSTANTIATE_GET_FG_TYPE(float           , forge::f32);
INSTANTIATE_GET_FG_TYPE(int             , forge::s32);
INSTANTIATE_GET_FG_TYPE(unsigned        , forge::u32);
INSTANTIATE_GET_FG_TYPE(char            , forge::s8);
INSTANTIATE_GET_FG_TYPE(unsigned char   , forge::u8);
INSTANTIATE_GET_FG_TYPE(unsigned short  , forge::u16);
INSTANTIATE_GET_FG_TYPE(short           , forge::s16);

gl::GLenum glErrorSkip(const char *msg, const char* file, int line)
{
#ifndef NDEBUG
    gl::GLenum x = gl::glGetError();
    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error Skipped at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, (int)x, glbinding::Meta::getString(x).c_str());
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
#else
    return (gl::GLenum)0;
#endif
}

gl::GLenum glErrorCheck(const char *msg, const char* file, int line)
{
// Skipped in release mode
#ifndef NDEBUG
    gl::GLenum x = gl::glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, (int)x, glbinding::Meta::getString(x).c_str());
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
#else
    return (gl::GLenum)0;
#endif
}

gl::GLenum glForceErrorCheck(const char *msg, const char* file, int line)
{
    gl::GLenum x = gl::glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, (int)x, glbinding::Meta::getString(x).c_str());
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
}

size_t getTypeSize(gl::GLenum type)
{
    switch(type) {
        case GL_FLOAT:          return sizeof(float);
        case GL_INT:            return sizeof(int  );
        case GL_UNSIGNED_INT:   return sizeof(unsigned);
        case GL_SHORT:          return sizeof(short);
        case GL_UNSIGNED_SHORT: return sizeof(unsigned short);
        case GL_BYTE:           return sizeof(char );
        case GL_UNSIGNED_BYTE:  return sizeof(unsigned char);
        default: return sizeof(float);
    }
}

void makeContextCurrent(forge::Window *window)
{
    window->makeCurrent();
    glbinding::Binding::useCurrentContext();
    CheckGL("End makeContextCurrent");
}

// dir -> true = round up, false = round down
double step_round(const double in, const bool dir)
{
    if(in == 0) return 0;

    static const double __log2 = log10(2);
    static const double __log4 = log10(4);
    static const double __log6 = log10(6);
    static const double __log8 = log10(8);

    // log_in is of the form "s abc.xyz", where
    // s is either + or -; + indicates abs(in) >= 1 and - indicates 0 < abs(in) < 1 (log10(1) is +0)
    const double sign   = in < 0 ? -1 : 1;
    const double log_in = std::log10(std::fabs(in));
    const double mag    = std::pow(10, std::floor(log_in)) * sign;  // Number of digits either left or right of 0
    const double dec    = std::log10(in / mag); // log of the fraction

    // This means in is of the for 10^n
    if(dec == 0) return in;

    // For negative numbers, -ve round down = +ve round up and vice versa
    bool op_dir = in > 0 ? dir : !dir;

    double mult = 1;

    // Round up
    if(op_dir) {
        if(dec <= __log2) {
            mult = 2;
        } else if(dec <= __log4) {
            mult = 4;
        } else if(dec <= __log6) {
            mult = 6;
        } else if(dec <= __log8) {
            mult = 8;
        } else {
            mult = 10;
        }
    } else {    // Round down
        if(dec < __log2) {
            mult = 1;
        } else if(dec < __log4) {
            mult = 2;
        } else if(dec < __log6) {
            mult = 4;
        } else if(dec < __log8) {
            mult = 6;
        } else {
            mult = 8;
        }
    }

    return mag * mult;
}

namespace graphics
{
ForgeManager& ForgeManager::getInstance()
{
    static ForgeManager my_instance;
    return my_instance;
}

ForgeManager::~ForgeManager()
{
    /* clear all OpenGL resource objects (images, plots, histograms etc) first
     * and then delete the windows */
    for(ImgMapIter iter = mImgMap.begin(); iter != mImgMap.end(); iter++)
        delete (iter->second);

    for(PltMapIter iter = mPltMap.begin(); iter != mPltMap.end(); iter++)
        delete (iter->second);

    for(HstMapIter iter = mHstMap.begin(); iter != mHstMap.end(); iter++)
        delete (iter->second);

    for(ChartMapIter iter = mChartMap.begin(); iter != mChartMap.end(); iter++) {
        for(int i = 0; i < (int)(iter->second).size(); i++) {
            if((iter->second)[i] != NULL) {
                delete (iter->second)[i];
                mChartAxesOverrideMap.erase((iter->second)[i]);
            }
        }
    }
}

forge::Font* ForgeManager::getFont()
{
    static std::once_flag flag;
    static std::unique_ptr<forge::Font> fnt;

    CheckGL("Begin ForgeManager::getFont");
    std::call_once(flag,
            [] {
                fnt.reset(new forge::Font());
#if defined(_WIN32) || defined(_MSC_VER)
                fnt->loadSystemFont("Arial");
#else
                fnt->loadSystemFont("Vera");
#endif
            });
    CheckGL("End ForgeManager::getFont");

    return fnt.get();
}

forge::Window* ForgeManager::getMainWindow()
{
    static std::once_flag flag;
    static std::unique_ptr<forge::Window> wnd;

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    std::string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");

    if (noGraphicsENV.empty()) { // If AF_DISABLE_GRAPHICS is not defined
        std::call_once(flag,
                [] {
                    wnd.reset(new forge::Window(WIDTH, HEIGHT, "ArrayFire", NULL, true));
                    makeContextCurrent(wnd.get());

                    ForgeManager::getInstance().setWindowChartGrid(wnd.get(), 1, 1);
                });
    }

    return wnd.get();
}

void ForgeManager::setWindowChartGrid(const forge::Window* window,
                                      const int r, const int c)
{
    ChartMapIter iter = mChartMap.find(window);
    GridMapIter gIter = mWndGridMap.find(window);

    if(iter != mChartMap.end()) {
        // ChartVec found. Clear it.
        // This has to be cleared as there is no guarantee that existing
        // chart types(2D/3D) match the future grid requirements
        for(int i = 0; i < (int)(iter->second).size(); i++) {
            if ((iter->second)[i] != NULL) {
                delete (iter->second)[i];
                mChartAxesOverrideMap.erase((iter->second)[i]);
            }
        }
        (iter->second).clear();
        gIter->second = std::make_pair<int, int>(1, 1);
    }

    if(r == 0 || c == 0) {
        mChartMap.erase(window);
        mWndGridMap.erase(window);
    } else {
        mChartMap[window] = std::vector<forge::Chart*>(r * c);
        mWndGridMap[window] = std::make_pair(r, c);
    }
}

WindGridDims_t ForgeManager::getWindowGrid(const forge::Window* window)
{
    GridMapIter gIter = mWndGridMap.find(window);

    if (gIter == mWndGridMap.end()) {
        mWndGridMap[window] = std::make_pair(1, 1);
    }

    return mWndGridMap[window];
}

forge::Chart* ForgeManager::getChart(const forge::Window* window, const int r, const int c,
                                     const forge::ChartType ctype)
{
    forge::Chart* chart = NULL;
    ChartMapIter iter = mChartMap.find(window);
    GridMapIter gIter = mWndGridMap.find(window);

    if (iter != mChartMap.end()) {

        int gRows = std::get<0>(gIter->second);
        int gCols = std::get<1>(gIter->second);

        if(c >= gCols || r >= gRows)
            AF_ERROR("Grid points are out of bounds", AF_ERR_TYPE);

        // upgrade to exclusive access to make changes
        chart = (iter->second)[c * gRows + r];

        if (chart == NULL) {
            // Chart has not been created
            chart = new forge::Chart(ctype);
            (iter->second)[c * gRows + r] = chart;
            // Set Axes override to false
            mChartAxesOverrideMap[chart] = false;
        } else if (chart->getChartType()!=ctype) {
            // Existing chart is of incompatible type
            delete (iter->second)[c * gRows + r];
            chart = new forge::Chart(ctype);
            (iter->second)[c * gRows + r] = chart;
            // Set Axes override to false
            mChartAxesOverrideMap[chart] = false;
        }
    } else {
        // The chart map for this was never created
        // Which should never happen
    }

    return chart;
}

forge::Image* ForgeManager::getImage(int w, int h, forge::ChannelFormat mode, forge::dtype type)
{
    /* w, h needs to fall in the range of [0, 2^16]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Image object. So, this implementation
     * is a limitation on how big of an image can be rendered
     * using arrayfire graphics funtionality */
    assert(w <= 2ll<<16);
    assert(h <= 2ll<<16);
    long long key = ((w & _16BIT) << 16) | (h & _16BIT);
    key = (((key << 16) | mode) << 16) | type;

    ChartKey_t keypair = std::make_pair(key, nullptr);

    ImgMapIter iter = mImgMap.find(keypair);

    if (iter==mImgMap.end()) {
        forge::Image* temp = new forge::Image(w, h, mode, type);

        mImgMap[keypair] = temp;
    }

    return mImgMap[keypair];
}

forge::Image* ForgeManager::getImage(forge::Chart* chart, int w, int h,
                                     forge::ChannelFormat mode, forge::dtype type)
{
    /* w, h needs to fall in the range of [0, 2^16]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Image object. So, this implementation
     * is a limitation on how big of an image can be rendered
     * using arrayfire graphics funtionality */
    assert(w <= 2ll<<16);
    assert(h <= 2ll<<16);
    long long key = ((w & _16BIT) << 16) | (h & _16BIT);
    key = (((key << 16) | mode) << 16) | type;

    ChartKey_t keypair = std::make_pair(key, chart);

    ImgMapIter iter = mImgMap.find(keypair);

    if (iter==mImgMap.end()) {
        if(chart->getChartType() != FG_CHART_2D)
            AF_ERROR("Image can only be added to chart of type FG_CHART_2D", AF_ERR_TYPE);

        forge::Image* temp = new forge::Image(w, h, mode, type);

        mImgMap[keypair] = temp;

        chart->add(*mImgMap[keypair]);
    }

    return mImgMap[keypair];
}

forge::Plot* ForgeManager::getPlot(forge::Chart* chart, int nPoints, forge::dtype dtype,
                                   forge::PlotType ptype, forge::MarkerType mtype)
{
    long long key = ((nPoints & _48BIT) << 48);
    key |= (((((dtype & 0x000F) << 12) | (ptype & 0x000F)) << 8) | (mtype & 0x000F));

    ChartKey_t keypair = std::make_pair(key, chart);

    PltMapIter iter = mPltMap.find(keypair);

    if (iter==mPltMap.end()) {
        forge::Plot* temp = new forge::Plot(nPoints, dtype, chart->getChartType(), ptype, mtype);

        mPltMap[keypair] = temp;

        chart->add(*mPltMap[keypair]);
    }

    return mPltMap[keypair];
}

forge::Histogram* ForgeManager::getHistogram(forge::Chart* chart, int nBins, forge::dtype type)
{
    long long key = ((nBins & _48BIT) << 48) | (type & _16BIT);

    ChartKey_t keypair = std::make_pair(key, chart);

    HstMapIter iter = mHstMap.find(keypair);

    if (iter==mHstMap.end()) {
        if(chart->getChartType() != FG_CHART_2D)
            AF_ERROR("Histogram can only be added to chart of type FG_CHART_2D", AF_ERR_TYPE);

        forge::Histogram* temp = new forge::Histogram(nBins, type);

        mHstMap[keypair] = temp;

        chart->add(*mHstMap[keypair]);
    }

    return mHstMap[keypair];
}

forge::Surface* ForgeManager::getSurface(forge::Chart* chart, int nX, int nY, forge::dtype type)
{
    /* nX * nY needs to fall in the range of [0, 2^48]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Plot object. So, this implementation
     * is a limitation on how big of an plot graph can be rendered
     * using arrayfire graphics funtionality */
    assert((long long)nX * nY <= 2ll<<48);
    long long key = (((nX * nY) & _48BIT) << 48) | (type & _16BIT);

    ChartKey_t keypair = std::make_pair(key, chart);

    SfcMapIter iter = mSfcMap.find(keypair);

    if (iter==mSfcMap.end()) {
        if(chart->getChartType() != FG_CHART_3D)
            AF_ERROR("Surface can only be added to chart of type FG_CHART_3D", AF_ERR_TYPE);

        forge::Surface* temp = new forge::Surface(nX, nY, type);

        mSfcMap[keypair] = temp;

        chart->add(*mSfcMap[keypair]);
    }

    return mSfcMap[keypair];
}

forge::VectorField* ForgeManager::getVectorField(forge::Chart* chart, int nPoints, forge::dtype type)
{
    long long key = (((nPoints) & _48BIT) << 48) | (type & _16BIT);

    ChartKey_t keypair = std::make_pair(key, chart);

    VcfMapIter iter = mVcfMap.find(keypair);

    if (iter==mVcfMap.end()) {
        forge::VectorField* temp = new forge::VectorField(nPoints, type, chart->getChartType());

        mVcfMap[keypair] = temp;

        chart->add(*mVcfMap[keypair]);
    }

    return mVcfMap[keypair];
}

bool ForgeManager::getChartAxesOverride(forge::Chart* chart)
{
    ChartAxesOverrideIter iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        AF_ERROR("Chart Not Found!", AF_ERR_INTERNAL);
    }
    return mChartAxesOverrideMap[chart];
}

void ForgeManager::setChartAxesOverride(forge::Chart* chart, bool flag)
{
    ChartAxesOverrideIter iter = mChartAxesOverrideMap.find(chart);
    if (iter == mChartAxesOverrideMap.end()) {
        AF_ERROR("Chart Not Found!", AF_ERR_INTERNAL);
    }
    mChartAxesOverrideMap[chart] = flag;
}
}

#endif
