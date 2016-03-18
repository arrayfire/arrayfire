/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/graphics.h>
#include "error.hpp"

namespace af
{

void Window::initWindow(const int width, const int height, const char* const title)
{
    AF_THROW(af_create_window(&wnd, width, height, title));
}

Window::Window()
    : wnd(0), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT)
{
    initWindow(1280, 720, "ArrayFire");
}

Window::Window(const char* const title)
    : wnd(0), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT)
{
    initWindow(1280, 720, title);
}

Window::Window(const int width, const int height, const char* const title)
    : wnd(0), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT)
{
    initWindow(width, height, title);
}

Window::Window(const af_window window)
    : wnd(window), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT)
{
}

Window::~Window()
{
    AF_THROW(af_destroy_window(wnd));
}

void Window::setPos(const unsigned x, const unsigned y)
{
    AF_THROW(af_set_position(get(), x, y));
}

void Window::setTitle(const char* const title)
{
    AF_THROW(af_set_title(get(), title));
}

void Window::setSize(const unsigned w, const unsigned h)
{
    AF_THROW(af_set_size(get(), w, h));
}

void Window::setColorMap(const ColorMap cmap)
{
    _cmap = cmap;
}

void Window::image(const array& in, const char* const title)
{
    af_cell temp{_r, _c, title, _cmap};
    AF_THROW(af_draw_image(get(), in.get(), &temp));
}

void Window::plot(const array& X, const array& Y, const char* const title)
{
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_plot(get(), X.get(), Y.get(), &temp));
}

void Window::scatter(const array& X, const array& Y, af::markerType marker, const char* const title)
{
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_scatter(get(), X.get(), Y.get(), marker, &temp));
}

void Window::scatter3(const array& P, af::markerType marker, const char* const title)
{
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_scatter3(get(), P.get(), marker, &temp));
}

void Window::plot3(const array& P, const char* const title)
{
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    P.eval();
    AF_THROW(af_draw_plot3(get(), P.get(), &temp));
}

void Window::hist(const array& X, const double minval, const double maxval, const char* const title)
{
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_hist(get(), X.get(), minval, maxval, &temp));
}

void Window::surface(const array& S, const char* const title){
    af::array xVals = seq(0, S.dims(0)-1);
    af::array yVals = seq(0, S.dims(1)-1);
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_surface(get(), xVals.get(), yVals.get(), S.get(), &temp));
}

void Window::surface(const array& xVals, const array& yVals, const array& S, const char* const title)
{
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_surface(get(), xVals.get(), yVals.get(), S.get(), &temp));
}

void Window::grid(const int rows, const int cols)
{
    AF_THROW(af_grid(get(), rows, cols));
}

void Window::show()
{
    AF_THROW(af_show(get()));
    _r = -1;
    _c = -1;
}

bool Window::close()
{
    bool temp = true;
    AF_THROW(af_is_window_closed(&temp, get()));
    return temp;
}

void Window::setVisibility(const bool isVisible)
{
    AF_THROW(af_set_visibility(get(), isVisible));
}

}
