/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/forge_loader.hpp>
#include <af/graphics.h>

#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace arrayfire {
namespace common {

// default to f32(float) type
template<typename T>
fg_dtype getGLType();

// Print for OpenGL errors
// Returns 1 if an OpenGL error occurred, 0 otherwise.
GLenum glErrorCheck(const char* msg, const char* file, int line);

#define CheckGL(msg) \
    arrayfire::common::glErrorCheck(msg, __AF_FILENAME__, __LINE__)

fg_marker_type getFGMarker(const af_marker_type af_marker);

void makeContextCurrent(fg_window window);

double step_round(const double in, const bool dir);

/// \brief The singleton manager class for Forge resources
///
/// Only device manager class can create objects of this class.
/// You have to call forgeManager() defined in platform.hpp to
/// access the object. It manages the windows, and other
/// renderables (given below) that are drawed onto chosen window.
/// Renderables:
///      fg_image
///      fg_plot
///      fg_histogram
///      fg_surface
///      fg_vector_field
///
class ForgeManager {
   public:
    using WindowGridDims = std::pair<int, int>;

    ForgeManager();
    ForgeManager(ForgeManager const&)            = delete;
    ForgeManager& operator=(ForgeManager const&) = delete;
    ForgeManager(ForgeManager&&)                 = delete;
    ForgeManager& operator=(ForgeManager&&)      = delete;

    /// \brief Module used to invoke forge API calls
    common::ForgeModule& plugin();

    /// \brief The main window with which all other windows share GL context
    fg_window getMainWindow();

    /// \brief Create a window
    ///
    /// \param[in] width of the window
    /// \param[in] height of the window
    /// \param[in] title is the window title
    /// \param[in] invisible indicates that if an invisible window
    ///            has to be ceated
    ///
    /// \note Any window created will always shared OpenGL context with
    ///       with the primary(getMainWindow()) window.
    fg_window getWindow(const int width, const int height,
                        const char* const title, const bool invisible = false);

    /// \brief Set grid layout for a given Window
    ///
    /// Grid layout dictates how many renderables can be shown in a
    /// single window. For example, if r = 2, c = 2, the entire rendering
    /// area of the window will be split into four sections into which
    /// different renderables can be drawn.
    ///
    /// \param[in] window is the target rendering context
    /// \param[in] r is the number of rows in the grid
    /// \param[in] c is the number of cols in the grid
    void setWindowChartGrid(const fg_window window, const int r, const int c);

    /// \brief Get grid layout of a window
    ///
    /// This function fetches the grid layout set for given window, probably
    /// which was set by the function \ref ForgeManager::setWindowChartGrid
    ///
    /// \param[in] window is the target rendering context
    WindowGridDims getWindowGrid(const fg_window window);

    /// \brief Find/Create a Chart
    ///
    /// This function tries to find a chart fitting the given attributes
    /// from forge resource cache. If a match is found, the matching chart
    /// resource handle is returned. If no match is found, a new chart
    /// with given parameters is created, cached and returned.
    ///
    /// \param[in] window is the target rendering context
    /// \param[in] r is indicates the row index in the grid layout
    ///            of the given \p window. This is usually 0 for grids having
    ///            single cell a.k.a capable of drawing one renderable.
    /// \param[in] c is indicates the col index in the grid layout
    ///            of the given \p window. This is usually 0 for grids having
    ///            single cell a.k.a capable of drawing one renderable.
    /// \param[in] ctype is type renderables to be rendered on chart, 2D or 3D
    fg_chart getChart(const fg_window window, const int r, const int c,
                      const fg_chart_type ctype);

    /// \brief Find/Create an Image
    ///
    /// This function tries to find an image fitting the given attributes
    /// from forge resource cache. If a match is found, the matching image
    /// resource handle is returned. If no match is found, a new image
    /// with given parameters is created, cached and returned.
    ///
    /// Also do keep in mind this function has to be used only when you
    /// are rendering just an image to the window. If you want to render
    /// an image embedded into set of plots or anything else, use the getImage
    /// member function that takes in \ref fg_chart as first parameter.
    ///
    /// \param[in] w is width of the image
    /// \param[in] h is height of the image
    /// \param[in] mode is the pixel packing format in the image
    /// \param[in] type is type of data to be stored in image buffer
    ///
    /// \note The width and height of image needs to fall in the range of
    /// [0, 2^16] for the ForgeManager to correctly retrieve the necessary
    /// Forge Image object. This is an implementation limitation on how big
    /// of an image can be rendered using arrayfire graphics funtionality
    fg_image getImage(int w, int h, fg_channel_format mode, fg_dtype type);

    /// \brief Find/Create an Image to render in a Chart
    ///
    /// This function tries to find an image fitting the given attributes
    /// from forge resource cache. If a match is found, the matching image
    /// resource handle is returned. If no match is found, a new image
    /// with given parameters is created, cached and returned.
    ///
    /// \param[in] chart is the chart to which image will be rendered
    /// \param[in] w is width of the image
    /// \param[in] h is height of the image
    /// \param[in] mode is the pixel packing format in the image
    /// \param[in] type is type of data to be stored in image buffer
    ///
    /// \note The width and height of image needs to fall in the range of
    /// [0, 2^16] for the ForgeManager to correctly retrieve the necessary
    /// Forge Image object. This is an implementation limitation on how big
    /// of an image can be rendered using arrayfire graphics funtionality
    fg_image getImage(fg_chart chart, int w, int h, fg_channel_format mode,
                      fg_dtype type);

    /// \brief Find/Create a Plot to render in a Chart
    ///
    /// This function tries to find a plot fitting the given attributes
    /// from forge resource cache. If a match is found, the matching plot
    /// resource handle is returned. If no match is found, a new plot
    /// with given parameters is created, cached and returned.
    ///
    /// \param[in] chart is the chart to which plot will be rendered
    /// \param[in] nPoints is number of points in the plot
    /// \param[in] dtype is type of data to be stored in plot buffer
    /// \param[in] ptype indicates the type of plot \ref fg_plot_type
    /// \param[in] mtype indicates the type of marker/sprite to render original
    ///            points passed in the data buffer, \ref fg_marker_type
    ///
    /// \note \p nPoints needs to fall in the range of [0, 2^48]
    /// for the ForgeManager to correctly retrieve the necessary Forge
    /// plot object. This is an implementation limitation on how big of a
    /// plot can be rendered using arrayfire graphics funtionality
    fg_plot getPlot(fg_chart chart, int nPoints, fg_dtype dtype,
                    fg_plot_type ptype, fg_marker_type mtype);

    /// \brief Find/Create a Histogram to render in a Chart
    ///
    /// This function tries to find a histogram fitting the given attributes
    /// from forge resource cache. If a match is found, the matching histogram
    /// resource handle is returned. If no match is found, a new histogram
    /// with given parameters is created, cached and returned.
    ///
    /// \param[in] chart is the chart to which histogram will be rendered
    /// \param[in] nBins is the total number of bins in the histogram
    /// \param[in] type is type of data to be stored in histogram buffer
    ///
    /// \note \p nBins needs to fall in the range of [0, 2^48]
    /// for the ForgeManager to correctly retrieve the necessary Forge
    /// histogram object. This is an implementation limitation on how big
    /// of a histogram can be rendered using arrayfire graphics funtionality
    fg_histogram getHistogram(fg_chart chart, int nBins, fg_dtype type);

    /// \brief Find/Create a Surface to render in a Chart
    ///
    /// This function tries to find a surface fitting the given attributes
    /// from forge resource cache. If a match is found, the matching surface
    /// resource handle is returned. If no match is found, a new surface
    /// with given parameters is created, cached and returned.
    ///
    /// \param[in] chart is the chart to which surface will be rendered
    /// \param[in] nX is length of the surface grid
    /// \param[in] nY is width of the surface grid
    /// \param[in] type is type of data to be stored in image buffer
    ///
    /// \note \p nX * \p nY needs to fall in the range of [0, 2^48]
    /// for the ForgeManager to correctly retrieve the necessary Forge Surface
    /// object. This is an implementation limitation on how big of a surface
    /// can be rendered using arrayfire graphics funtionality
    fg_surface getSurface(fg_chart chart, int nX, int nY, fg_dtype type);

    /// \brief Find/Create a Vector Field to render in a Chart
    ///
    /// This function tries to find a vector field fitting the given attributes
    /// from forge resource cache. If a match is found, the matching vector
    /// field resource handle is returned. If no match is found, a new vector
    /// field with given parameters is created, cached and returned.
    ///
    /// \param[in] chart is the chart to which plot will be rendered
    /// \param[in] nPoints is number of points in the 2D vector field
    /// \param[in] type is type of data to be stored in plot buffer
    ///
    /// \note \p nPoints needs to fall in the range of [0, 2^48]
    /// for the ForgeManager to correctly retrieve the necessary Forge vector
    /// field object. This is an implementation limitation on how big of a
    /// vector field can be rendered using arrayfire graphics funtionality
    fg_vector_field getVectorField(fg_chart chart, int nPoints, fg_dtype type);

    /// \brief Get chart axes limits override flag
    ///
    /// \param[in] chart is the target chart for which axes limits will be
    /// overriden
    bool getChartAxesOverride(const fg_chart chart);

    /// \brief Set chart axes limits override flag
    ///
    /// \param[in] chart is the target chart for which axes limits will be
    /// overriden \param[in] flag indicates if axes limits are overriden or not
    void setChartAxesOverride(const fg_chart chart, bool flag = true);

   private:
    constexpr static unsigned int WIDTH        = 1280;
    constexpr static unsigned int HEIGHT       = 720;
    constexpr static unsigned long long _4BIT  = 0x000000000000000F;
    constexpr static unsigned long long _8BIT  = 0x00000000000000FF;
    constexpr static unsigned long long _16BIT = 0x000000000000FFFF;
    constexpr static unsigned long long _32BIT = 0x00000000FFFFFFFF;
    constexpr static unsigned long long _48BIT = 0x0000FFFFFFFFFFFF;

    static unsigned long long genImageKey(unsigned w, unsigned h,
                                          fg_channel_format mode,
                                          fg_dtype type);

#define DEFINE_WRAPPER_OBJECT(OBJECT, RELEASE)                           \
    struct OBJECT {                                                      \
        void* handle;                                                    \
        struct Deleter {                                                 \
            void operator()(OBJECT* pHandle) const {                     \
                if (pHandle) { forgePlugin().RELEASE(pHandle->handle); } \
            }                                                            \
        };                                                               \
    }

    DEFINE_WRAPPER_OBJECT(Window, fg_release_window);
    DEFINE_WRAPPER_OBJECT(Image, fg_release_image);
    DEFINE_WRAPPER_OBJECT(Chart, fg_release_chart);
    DEFINE_WRAPPER_OBJECT(Plot, fg_release_plot);
    DEFINE_WRAPPER_OBJECT(Histogram, fg_release_histogram);
    DEFINE_WRAPPER_OBJECT(Surface, fg_release_surface);
    DEFINE_WRAPPER_OBJECT(VectorField, fg_release_vector_field);

#undef DEFINE_WRAPPER_OBJECT

    using ImagePtr       = std::unique_ptr<Image, Image::Deleter>;
    using ChartPtr       = std::unique_ptr<Chart, Chart::Deleter>;
    using PlotPtr        = std::unique_ptr<Plot, Plot::Deleter>;
    using SurfacePtr     = std::unique_ptr<Surface, Surface::Deleter>;
    using HistogramPtr   = std::unique_ptr<Histogram, Histogram::Deleter>;
    using VectorFieldPtr = std::unique_ptr<VectorField, VectorField::Deleter>;
    using ChartList      = std::vector<ChartPtr>;
    using ChartKey       = std::pair<unsigned long long, fg_chart>;

    using ChartMapIterator     = std::map<fg_window, ChartList>::iterator;
    using WindGridMapIterator  = std::map<fg_window, WindowGridDims>::iterator;
    using AxesOverrideIterator = std::map<fg_chart, bool>::iterator;
    using ImageMapIterator     = std::map<ChartKey, ImagePtr>::iterator;
    using PlotMapIterator      = std::map<ChartKey, PlotPtr>::iterator;
    using HistogramMapIterator = std::map<ChartKey, HistogramPtr>::iterator;
    using SurfaceMapIterator   = std::map<ChartKey, SurfacePtr>::iterator;
    using VecFieldMapIterator  = std::map<ChartKey, VectorFieldPtr>::iterator;

    std::unique_ptr<common::ForgeModule> mPlugin;
    std::unique_ptr<Window, Window::Deleter> mMainWindow;

    std::map<fg_window, ChartList> mChartMap;
    std::map<ChartKey, ImagePtr> mImgMap;
    std::map<ChartKey, PlotPtr> mPltMap;
    std::map<ChartKey, HistogramPtr> mHstMap;
    std::map<ChartKey, SurfacePtr> mSfcMap;
    std::map<ChartKey, VectorFieldPtr> mVcfMap;
    std::map<fg_window, WindowGridDims> mWndGridMap;
    std::map<fg_chart, bool> mChartAxesOverrideMap;
};

}  // namespace common
}  // namespace arrayfire
