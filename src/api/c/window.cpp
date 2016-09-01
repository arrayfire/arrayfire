/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/


#include <af/graphics.h>

#include <graphics_common.hpp>
#include <err_common.hpp>
#include <backend.hpp>

using af::dim4;
using namespace detail;

#if defined(WITH_GRAPHICS)
using namespace graphics;
#endif


af_err af_create_window(af_window *out, const int width, const int height, const char* const title)
{
#if defined(WITH_GRAPHICS)
    forge::Window* wnd;
    try {
        graphics::ForgeManager& fgMngr = graphics::ForgeManager::getInstance();
        forge::Window* mainWnd = NULL;

        try {
            mainWnd = fgMngr.getMainWindow();
        } catch(...) {
            std::cerr<<"OpenGL context creation failed"<<std::endl;
        }

        if(mainWnd==0) {
            std::cerr<<"Not a valid window"<<std::endl;
            return AF_SUCCESS;
        }

        wnd = new forge::Window(width, height, title, mainWnd);
        wnd->setFont(fgMngr.getFont());
        *out = reinterpret_cast<af_window>(wnd);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_set_position(const af_window wind, const unsigned x, const unsigned y)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        wnd->setPos(x, y);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_set_title(const af_window wind, const char* const title)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        wnd->setTitle(title);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_set_size(const af_window wind, const unsigned w, const unsigned h)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        wnd->setSize(w, h);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_grid(const af_window wind, const int rows, const int cols)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        wnd->grid(rows, cols);
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_show(const af_window wind)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        wnd->swapBuffers();
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_is_window_closed(bool *out, const af_window wind)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        *out = wnd->close();
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_set_visibility(const af_window wind, const bool is_visible)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        if (is_visible)
            wnd->show();
        else
            wnd->hide();
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

af_err af_destroy_window(const af_window wind)
{
#if defined(WITH_GRAPHICS)
    if(wind==0) {
        std::cerr<<"Not a valid window"<<std::endl;
        return AF_SUCCESS;
    }

    try {
        forge::Window* wnd = reinterpret_cast<forge::Window*>(wind);
        delete wnd;
    }
    CATCHALL;
    return AF_SUCCESS;
#else
    AF_RETURN_ERROR("ArrayFire compiled without graphics support", AF_ERR_NO_GFX);
#endif
}

