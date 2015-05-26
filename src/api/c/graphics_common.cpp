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

fg::Font* ForgeManager::getFont()
{
    static bool flag = true;
    static fg::Font* fnt = NULL;

    CheckGL("Begin ForgeManager::getFont");

    if (flag) {
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

fg::Window* ForgeManager::getMainWindow()
{
    static bool flag = true;
    static fg::Window* wnd = NULL;

    if (flag) {
	wnd = new fg::Window(WIDTH, HEIGHT, "ArrayFire", NULL, true);
	CheckGL("End ForgeManager::getMainWindow");
	flag = false;
    };
    return wnd;
}

fg::Image* ForgeManager::getImage(int w, int h, fg::ColorMode mode, GLenum type)
{
    size_t size = w * h * mode * getTypeSize(type);

    ImgMapIter iter = mImgMap.find(size);
    if (iter==mImgMap.end()) {
        fg::Image* temp = new fg::Image(w, h, mode, type);
        mImgMap[size] = temp;
    }

    return mImgMap[size];
}

fg::Plot* ForgeManager::getPlot(int nPoints, GLenum type)
{
    size_t size = nPoints * getTypeSize(type);

    PltMapIter iter = mPltMap.find(size);
    if (iter==mPltMap.end()) {
        fg::Plot* temp = new fg::Plot(nPoints, type);
        mPltMap[size] = temp;
    }

    return mPltMap[size];
}

fg::Histogram* ForgeManager::getHistogram(int nBins, GLenum type)
{
    size_t size = nBins * getTypeSize(type);

    HstMapIter iter = mHstMap.find(size);
    if (iter==mHstMap.end()) {
        fg::Histogram* temp = new fg::Histogram(nBins, type);
        mHstMap[size] = temp;
    }

    return mHstMap[size];
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

    delete getFont();
    delete getMainWindow();
}

}

#endif
