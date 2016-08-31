/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined(WITH_GRAPHICS)

#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>
#include <platform.hpp>
#include <util.hpp>

#include <glbinding/gl/gl.h>
#include <glbinding/Meta.h>

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

INSTANTIATE_GET_FG_TYPE(float, forge::f32);
INSTANTIATE_GET_FG_TYPE(int  , forge::s32);
INSTANTIATE_GET_FG_TYPE(unsigned, forge::u32);
INSTANTIATE_GET_FG_TYPE(char, forge::s8);
INSTANTIATE_GET_FG_TYPE(unsigned char, forge::u8);
INSTANTIATE_GET_FG_TYPE(unsigned short, forge::u16);
INSTANTIATE_GET_FG_TYPE(short, forge::s16);

gl::GLenum glErrorSkip(const char *msg, const char* file, int line)
{
#ifndef NDEBUG
    gl::GLenum x = glGetError();
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
    gl::GLenum x = glGetError();

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
    gl::GLenum x = glGetError();

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

namespace graphics
{

ForgeManager& ForgeManager::getInstance()
{
    static ForgeManager my_instance;
    return my_instance;
}

ForgeManager::~ForgeManager()
{
    destroyResources();
}

forge::Font* ForgeManager::getFont(const bool dontCreate)
{
    static bool flag = true;
    static forge::Font* fnt = NULL;

    CheckGL("Begin ForgeManager::getFont");

    if (flag && !dontCreate) {
        fnt = new forge::Font();
#if defined(_WIN32) || defined(_MSC_VER)
        fnt->loadSystemFont("Arial");
#else
        fnt->loadSystemFont("Vera");
#endif
        CheckGL("End ForgeManager::getFont");
        flag = false;
    };

    return fnt;
}

forge::Window* ForgeManager::getMainWindow(const bool dontCreate)
{
    static bool flag = true;
    static forge::Window* wnd = NULL;

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    std::string noGraphicsENV = getEnvVar("AF_DISABLE_GRAPHICS");
    if(noGraphicsENV.empty()) { // If AF_DISABLE_GRAPHICS is not defined
        if (flag && !dontCreate) {
            wnd = new forge::Window(WIDTH, HEIGHT, "ArrayFire", NULL, true);
            makeContextCurrent(wnd);
            CheckGL("End ForgeManager::getMainWindow");
            flag = false;
        };
    }
    return wnd;
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

    ImgMapIter iter = mImgMap.find(key);
    if (iter==mImgMap.end()) {
        forge::Image* temp = new forge::Image(w, h, mode, type);
        mImgMap[key] = temp;
    }

    return mImgMap[key];
}

forge::Plot* ForgeManager::getPlot(int nPoints, forge::dtype dtype, forge::ChartType ctype, forge::PlotType ptype, forge::MarkerType mtype)
{
    /* nPoints needs to fall in the range of [0, 2^48]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Plot object. So, this implementation
     * is a limitation on how big of an plot graph can be rendered
     * using arrayfire graphics funtionality */
    assert(nPoints <= 2ll<<48);
    long long key = ((nPoints & _48BIT) << 48);
    key |= (((((dtype & 0x000F) << 12) | (ptype & 0x000F)) << 8) | (mtype & 0x000F));

    PltMapIter iter = mPltMap.find(key);
    if (iter==mPltMap.end()) {
        forge::Plot* temp = new forge::Plot(nPoints, dtype, ctype, ptype, mtype);
        mPltMap[key] = temp;
    }

    return mPltMap[key];
}

forge::Histogram* ForgeManager::getHistogram(int nBins, forge::dtype type)
{
    /* nBins needs to fall in the range of [0, 2^48]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Histogram object. So, this implementation
     * is a limitation on how big of an histogram data can be rendered
     * using arrayfire graphics funtionality */
    assert(nBins <= 2ll<<48);
    long long key = ((nBins & _48BIT) << 48) | (type & _16BIT);

    HstMapIter iter = mHstMap.find(key);
    if (iter==mHstMap.end()) {
        forge::Histogram* temp = new forge::Histogram(nBins, type);
        mHstMap[key] = temp;
    }

    return mHstMap[key];
}

forge::Surface* ForgeManager::getSurface(int nX, int nY, forge::dtype type)
{
    /* nX * nY needs to fall in the range of [0, 2^48]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Plot object. So, this implementation
     * is a limitation on how big of an plot graph can be rendered
     * using arrayfire graphics funtionality */
    assert(nX * nY <= 2ll<<48);
    long long key = (((nX * nY) & _48BIT) << 48) | (type & _16BIT);

    SfcMapIter iter = mSfcMap.find(key);
    if (iter==mSfcMap.end()) {
        forge::Surface* temp = new forge::Surface(nX, nY, type);
        mSfcMap[key] = temp;
    }

    return mSfcMap[key];
}

void ForgeManager::destroyResources()
{
    /* clear all OpenGL resource objects (images, plots, histograms etc) first
     * and then delete the windows */
    for(ImgMapIter iter = mImgMap.begin(); iter != mImgMap.end(); iter++)
        delete (iter->second);

    for(PltMapIter iter = mPltMap.begin(); iter != mPltMap.end(); iter++)
        delete (iter->second);

    for(HstMapIter iter = mHstMap.begin(); iter != mHstMap.end(); iter++)
        delete (iter->second);

    delete getFont(true);
    delete getMainWindow(true);
}

}

#endif
