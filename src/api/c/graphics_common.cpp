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

using namespace std;

template<typename T>
GLenum getGLType() { return GL_FLOAT; }

#define INSTANTIATE_GET_GL_TYPE(T, OpenGLEnum)\
    template<> GLenum getGLType<T>() { return OpenGLEnum; }

INSTANTIATE_GET_GL_TYPE(float, GL_FLOAT);
INSTANTIATE_GET_GL_TYPE(int  , GL_INT);
INSTANTIATE_GET_GL_TYPE(unsigned, GL_UNSIGNED_INT);
INSTANTIATE_GET_GL_TYPE(char, GL_BYTE);
INSTANTIATE_GET_GL_TYPE(unsigned char, GL_UNSIGNED_BYTE);

GLenum glErrorSkip(const char *msg, const char* file, int line)
{
#ifndef NDEBUG
    GLenum x = glGetError();
    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error Skipped at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
#else
    return 0;
#endif
}

GLenum glErrorCheck(const char *msg, const char* file, int line)
{
// Skipped in release mode
#ifndef NDEBUG
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
#else
    return 0;
#endif
}

GLenum glForceErrorCheck(const char *msg, const char* file, int line)
{
    GLenum x = glGetError();

    if (x != GL_NO_ERROR) {
        char buf[1024];
        sprintf(buf, "GL Error at: %s:%d Message: %s Error Code: %d \"%s\"\n", file, line, msg, x, gluErrorString(x));
        AF_ERROR(buf, AF_ERR_INTERNAL);
    }
    return x;
}

size_t getTypeSize(GLenum type)
{
    switch(type) {
        case GL_FLOAT:          return sizeof(float);
        case GL_INT:            return sizeof(int  );
        case GL_UNSIGNED_INT:   return sizeof(unsigned);
        case GL_BYTE:           return sizeof(char );
        case GL_UNSIGNED_BYTE:  return sizeof(unsigned char);
        default: return sizeof(float);
    }
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

fg::Font* ForgeManager::getFont(const bool dontCreate)
{
    static bool flag = true;
    static fg::Font* fnt = NULL;

    CheckGL("Begin ForgeManager::getFont");

    if (flag && !dontCreate) {
        fnt = new fg::Font();
#if defined(_WIN32) || defined(_MSC_VER)
        fnt->loadSystemFont("Arial", 32);
#else
        fnt->loadSystemFont("Vera", 32);
#endif
        CheckGL("End ForgeManager::getFont");
        flag = false;
    };

    return fnt;
}

fg::Window* ForgeManager::getMainWindow(const bool dontCreate)
{
    static bool flag = true;
    static fg::Window* wnd = NULL;

    // Define AF_DISABLE_GRAPHICS with any value to disable initialization
    const char* noGraphicsENV = getenv("AF_DISABLE_GRAPHICS");
    if(!noGraphicsENV) { // If AF_DISABLE_GRAPHICS is not defined
        if (flag && !dontCreate) {
            wnd = new fg::Window(WIDTH, HEIGHT, "ArrayFire", NULL, true);
            CheckGL("End ForgeManager::getMainWindow");
            flag = false;
        };
    }
    return wnd;
}

fg::Image* ForgeManager::getImage(int w, int h, fg::ColorMode mode, GLenum type)
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
        fg::Image* temp = new fg::Image(w, h, mode, type);
        mImgMap[key] = temp;
    }

    return mImgMap[key];
}

fg::Plot* ForgeManager::getPlot(int nPoints, GLenum type)
{
    /* nPoints needs to fall in the range of [0, 2^48]
     * for the ForgeManager to correctly retrieve
     * the necessary Forge Plot object. So, this implementation
     * is a limitation on how big of an plot graph can be rendered
     * using arrayfire graphics funtionality */
    assert(nPoints <= 2ll<<48);
    long long key = ((nPoints & _48BIT) << 48) | (type & _16BIT);

    PltMapIter iter = mPltMap.find(key);
    if (iter==mPltMap.end()) {
        fg::Plot* temp = new fg::Plot(nPoints, type);
        mPltMap[key] = temp;
    }

    return mPltMap[key];
}

fg::Histogram* ForgeManager::getHistogram(int nBins, GLenum type)
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
        fg::Histogram* temp = new fg::Histogram(nBins, type);
        mHstMap[key] = temp;
    }

    return mHstMap[key];
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
