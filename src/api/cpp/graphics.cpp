/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/array.h>
#include <af/data.h>
#include <af/graphics.h>
#include "error.hpp"

namespace af {

void Window::initWindow(const int width, const int height,
                        const char* const title) {
    AF_THROW(af_create_window(&wnd, width, height, title));
}

Window::Window() : wnd(0), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT) {
    initWindow(1280, 720, "ArrayFire");
}

Window::Window(const char* const title)
    : wnd(0), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT) {
    initWindow(1280, 720, title);
}

Window::Window(const int width, const int height, const char* const title)
    : wnd(0), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT) {
    initWindow(width, height, title);
}

Window::Window(const af_window window)
    : wnd(window), _r(-1), _c(-1), _cmap(AF_COLORMAP_DEFAULT) {}

Window::~Window() {
    // THOU SHALL NOT THROW IN DESTRUCTORS
    if (wnd) { af_destroy_window(wnd); }
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setPos(const unsigned x, const unsigned y) {
    AF_THROW(af_set_position(get(), x, y));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setTitle(const char* const title) {
    AF_THROW(af_set_title(get(), title));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setSize(const unsigned w, const unsigned h) {
    AF_THROW(af_set_size(get(), w, h));
}

void Window::setColorMap(const ColorMap cmap) { _cmap = cmap; }

void Window::image(const array& in, const char* const title) {
    af_cell temp{_r, _c, title, _cmap};
    AF_THROW(af_draw_image(get(), in.get(), &temp));
}

void Window::plot(const array& in, const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_plot_nd(get(), in.get(), &temp));
}

void Window::plot(const array& X, const array& Y, const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_plot_2d(get(), X.get(), Y.get(), &temp));
}

void Window::plot(const array& X, const array& Y, const array& Z,
                  const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_plot_3d(get(), X.get(), Y.get(), Z.get(), &temp));
}

void Window::plot3(const array& P, const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    P.eval();
    AF_THROW(af_draw_plot_nd(get(), P.get(), &temp));
}

void Window::scatter(const array& in, af::markerType marker,
                     const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_scatter_nd(get(), in.get(), marker, &temp));
}

void Window::scatter(const array& X, const array& Y, af::markerType marker,
                     const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_scatter_2d(get(), X.get(), Y.get(), marker, &temp));
}

void Window::scatter(const array& X, const array& Y, const array& Z,
                     af::markerType marker, const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(
        af_draw_scatter_3d(get(), X.get(), Y.get(), Z.get(), marker, &temp));
}

void Window::scatter3(const array& P, af::markerType marker,
                      const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_scatter_nd(get(), P.get(), marker, &temp));
}

void Window::hist(const array& X, const double minval, const double maxval,
                  const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_hist(get(), X.get(), minval, maxval, &temp));
}

void Window::surface(const array& S, const char* const title) {
    af::array xVals = range(S.dims(0));
    af::array yVals = range(S.dims(1));
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_surface(get(), xVals.get(), yVals.get(), S.get(), &temp));
}

void Window::surface(const array& xVals, const array& yVals, const array& S,
                     const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_surface(get(), xVals.get(), yVals.get(), S.get(), &temp));
}

void Window::vectorField(const array& points, const array& directions,
                         const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(
        af_draw_vector_field_nd(get(), points.get(), directions.get(), &temp));
}

void Window::vectorField(const array& xPoints, const array& yPoints,
                         const array& zPoints, const array& xDirs,
                         const array& yDirs, const array& zDirs,
                         const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_vector_field_3d(get(), xPoints.get(), yPoints.get(),
                                     zPoints.get(), xDirs.get(), yDirs.get(),
                                     zDirs.get(), &temp));
}

void Window::vectorField(const array& xPoints, const array& yPoints,
                         const array& xDirs, const array& yDirs,
                         const char* const title) {
    af_cell temp{_r, _c, title, AF_COLORMAP_DEFAULT};
    AF_THROW(af_draw_vector_field_2d(get(), xPoints.get(), yPoints.get(),
                                     xDirs.get(), yDirs.get(), &temp));
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::grid(const int rows, const int cols) {
    AF_THROW(af_grid(get(), rows, cols));
}

void Window::setAxesLimits(const array& x, const array& y, const bool exact) {
    af_cell temp{_r, _c, NULL, AF_COLORMAP_DEFAULT};
    AF_THROW(af_set_axes_limits_compute(get(), x.get(), y.get(), NULL, exact,
                                        &temp));
}

void Window::setAxesLimits(const array& x, const array& y, const array& z,
                           const bool exact) {
    af_cell temp{_r, _c, NULL, AF_COLORMAP_DEFAULT};
    AF_THROW(af_set_axes_limits_compute(get(), x.get(), y.get(), z.get(), exact,
                                        &temp));
}

void Window::setAxesLimits(const float xmin, const float xmax, const float ymin,
                           const float ymax, const bool exact) {
    af_cell temp{_r, _c, NULL, AF_COLORMAP_DEFAULT};
    AF_THROW(
        af_set_axes_limits_2d(get(), xmin, xmax, ymin, ymax, exact, &temp));
}

void Window::setAxesLimits(const float xmin, const float xmax, const float ymin,
                           const float ymax, const float zmin, const float zmax,
                           const bool exact) {
    af_cell temp{_r, _c, NULL, AF_COLORMAP_DEFAULT};
    AF_THROW(af_set_axes_limits_3d(get(), xmin, xmax, ymin, ymax, zmin, zmax,
                                   exact, &temp));
}

void Window::setAxesTitles(const char* const xtitle, const char* const ytitle,
                           const char* const ztitle) {
    af_cell temp{_r, _c, NULL, AF_COLORMAP_DEFAULT};
    AF_THROW(af_set_axes_titles(get(), xtitle, ytitle, ztitle, &temp));
}

void Window::setAxesLabelFormat(const char* const xformat,
                                const char* const yformat,
                                const char* const zformat) {
    af_cell temp{_r, _c, NULL, AF_COLORMAP_DEFAULT};
    AF_THROW(af_set_axes_label_format(get(), xformat, yformat, zformat, &temp));
}

void Window::show() {
    AF_THROW(af_show(get()));
    _r = -1;
    _c = -1;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
bool Window::close() {
    bool temp = true;
    AF_THROW(af_is_window_closed(&temp, get()));
    return temp;
}

// NOLINTNEXTLINE(readability-make-member-function-const)
void Window::setVisibility(const bool isVisible) {
    AF_THROW(af_set_visibility(get(), isVisible));
}

}  // namespace af
