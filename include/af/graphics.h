/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>
#include <af/array.h>

typedef void* af_window;

typedef struct {
    int row;
    int col;
    const char* title;
    af_colormap cmap;
} af_cell;

#ifdef __cplusplus
namespace af
{

/**
   \class Window

   \brief Window object to render af::arrays

   Windows are not CopyConstructible or CopyAssignable.

   \ingroup graphics_func
 */
class AFAPI Window {
    private:
        af_window wnd;
        /* below attributes are used to track which
         * cell in the grid is being rendered currently */
        int _r;
        int _c;
        ColorMap _cmap;

        void initWindow(const int width, const int height, const char* const title);

        Window(const Window&);                 // Prevent copy-construction
        Window& operator=(const Window&);      // Prevent assignment

    public:
        /**
           Creates a window object with default width
           and height with title set to "ArrayFire"

           \ingroup gfx_func_window
         */
        Window();

        /**
           Creates a window object with default width
           and height using the title provided by the user

           \param[in] title is the window title

           \ingroup gfx_func_window
         */
        Window(const char* const title);

        /**
           Creates a window object using the parameters
           provided by the user

           \param[in] width is the window width
           \param[in] height is the window height
           \param[in] title is the window title with default value as "ArrayFire"

           \ingroup gfx_func_window
         */
        Window(const int width, const int height, const char* const title="ArrayFire");

        /**
           Creates a window object with default width
           and height with title set to "ArrayFire"

           \param[in] window is an \ref af_window handle which can be retrieved
                             by
           doing a get call on any \ref Window object

           \ingroup gfx_func_window
         */
        Window(const af_window window);

        /**
           Destroys the window handle

           \ingroup gfx_func_window
         */
        ~Window();

        // FIXME handle copying properly

        /**
           \return Returns the \ref af_window window handle.

           \ingroup gfx_func_window
         */
        af_window get() const { return wnd; }

        /**
           Set the start position where the window will appear

           \param[in] x is horizontal coordinate
           \param[in] y is vertical coordinate

           \ingroup gfx_func_window
         */
        void setPos(const unsigned x, const unsigned y);

        /**
           Set the window title

           \param[in] title is the window title

           \ingroup gfx_func_window
         */
        void setTitle(const char* const title);

#if AF_API_VERSION >= 31
        /**
           Set the window size

           \param[in]   w is target width of the window
           \param[in]   h is target height of the window

           \ingroup gfx_func_window
         */
        void setSize(const unsigned w, const unsigned h);
#endif

        /**
           Set the colormap to be used for subsequent rendering calls

           \param[in] cmap should be one of the enum values from \ref ColorMap

           \ingroup gfx_func_window
         */
        void setColorMap(const ColorMap cmap);

        /**
           Renders the input array as an image to the window

           \param[in] in is an \ref array
           \param[in] title parameter is used when this function is called in grid mode

           \note \p in should be 2d array or 3d array with 3 channels.

           \ingroup gfx_func_draw
         */
        void image(const array& in, const char* title=NULL);

#if AF_API_VERSION >= 32
        /**
           Renders the input array as an 3d line plot to the window

           \param[in] in is an \ref array
           \param[in] title parameter is used when this function is called in grid mode

           \note \p in should be 1d array of size 3n or 2d array with (3 x n) or (n x 3) channels.

           \ingroup gfx_func_draw
         */
        AF_DEPRECATED("Use plot instead")
        void plot3(const array& in, const char* title=NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 2D or 3D plot to the window

           \param[in] in is an \ref array with the data points
           \param[in] title parameter is used when this function is called in grid mode

           \note \p in must be 2d and of the form [n, order], where order is either 2 or 3.
                 If order is 2, then chart is 2D and if order is 3, then chart is 3D.

           \ingroup gfx_func_draw
         */
        void plot(const array& in, const char* const title=NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 3D plot to the window

           \param[in] X is an \ref array with the x-axis data points
           \param[in] Y is an \ref array with the y-axis data points
           \param[in] Z is an \ref array with the z-axis data points
           \param[in] title parameter is used when this function is called in grid mode

           \note \p X, \p Y and \p Z should be vectors.

           \ingroup gfx_func_draw
         */
        void plot(const array& X, const array& Y, const array& Z, const char* const title=NULL);
#endif

        /**
           Renders the input arrays as a 2D plot to the window

           \param[in] X is an \ref array with the x-axis data points
           \param[in] Y is an \ref array with the y-axis data points
           \param[in] title parameter is used when this function is called in grid mode

           \note \p X and \p Y should be vectors.

           \ingroup gfx_func_draw
         */
        void plot(const array& X, const array& Y, const char* const title=NULL);

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 2D or 3D scatter-plot to the window

           \param[in] in is an \ref array with the data points
           \param[in] marker is an \ref markerType enum specifying which marker to use in the scatter plot
           \param[in] title parameter is used when this function is called in grid mode

           \note \p in must be 2d and of the form [n, order], where order is either 2 or 3.
                 If order is 2, then chart is 2D and if order is 3, then chart is 3D.

           \ingroup gfx_func_draw
         */
        void scatter(const array& in, const af::markerType marker = AF_MARKER_POINT,
                     const char* const title = NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 3D scatter-plot to the window

           \param[in] X is an \ref array with the x-axis data points
           \param[in] Y is an \ref array with the y-axis data points
           \param[in] Z is an \ref array with the z-axis data points
           \param[in] marker is an \ref markerType enum specifying which marker to use in the scatter plot
           \param[in] title parameter is used when this function is called in grid mode

           \note \p X, \p Y and \p Z should be vectors.

           \ingroup gfx_func_draw
         */
        void scatter(const array& X, const array& Y, const array& Z,
                     const af::markerType marker = AF_MARKER_POINT, const char* const title = NULL);
#endif

#if AF_API_VERSION >= 33
        /**
           Renders the input arrays as a 2D scatter-plot to the window

           \param[in] X is an \ref array with the x-axis data points
           \param[in] Y is an \ref array with the y-axis data points
           \param[in] marker is an \ref markerType enum specifying which marker to use in the scatter plot
           \param[in] title parameter is used when this function is called in grid mode

           \note \p X and \p Y should be vectors.

           \ingroup gfx_func_draw
         */
        void scatter(const array& X, const array& Y,
                     const af::markerType marker = AF_MARKER_POINT, const char* const title = NULL);
#endif

#if AF_API_VERSION >= 33
        /**
           Renders the input arrays as a 3D scatter-plot to the window

           \param[in] P is an \ref af_array or matrix with the xyz-values of the points
           \param[in] marker is an \ref markerType enum specifying which marker to use in the scatter plot
           \param[in] title parameter is used when this function is called in grid mode

           \ingroup gfx_func_draw
         */
        AF_DEPRECATED("Use scatter instead")
        void scatter3(const array& P, const af::markerType marker = AF_MARKER_POINT,
                      const char* const title = NULL);
#endif

        /**
           Renders the input array as a histogram to the window

           \param[in] X is the data frequency \ref array
           \param[in] minval is the value of the minimum data point of the array whose histogram(\p X) is going to be rendered.
           \param[in] maxval is the value of the maximum data point of the array whose histogram(\p X) is going to be rendered.
           \param[in] title parameter is used when this function is called in grid mode

           \note \p X should be a vector.

           \ingroup gfx_func_draw
         */
        void hist(const array& X, const double minval, const double maxval, const char* const title=NULL);

#if AF_API_VERSION >= 32
        /**
           Renders the input arrays as a 3D surface plot to the window

           \param[in] S is an \ref array with the z-axis data points
           \param[in] title parameter is used when this function is called in grid mode

           \note \p S should be a 2D array

           \ingroup gfx_func_draw
         */
        void surface(const array& S, const char* const title = NULL);
#endif

#if AF_API_VERSION >= 32
        /**
           Renders the input arrays as a 3D surface plot to the window

           \param[in] xVals is an \ref array with the x-axis data points
           \param[in] yVals is an \ref array with the y-axis data points
           \param[in] S is an \ref array with the z-axis data points
           \param[in] title parameter is used when this function is called in grid mode

           \note \p X and \p Y should be vectors or 2D arrays \p S should be s 2D array

           \ingroup gfx_func_draw
         */
        void surface(const array& xVals, const array& yVals, const array& S, const char* const title = NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 2D or 3D vector field plot to the window

           \param[in] points is an \ref array with the points
           \param[in] directions is an \ref array with the directions at the points
           \param[in] title parameter is used when this function is called in grid mode

           \note \p points and \p directions should have the same size and must
           be 2D.
           The number of rows (dim 0) determines are number of points and the
           number columns determines the type of plot. If the number of columns
           are 2, then the plot is 2D and if there are 3 columns, then the plot
           is 3D.

           \ingroup gfx_func_draw
         */
        void vectorField(const array& points, const array& directions, const char* const title = NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 3D vector field plot to the window

           \param[in] xPoints is an \ref array with the x-coordinate points
           \param[in] yPoints is an \ref array with the y-coordinate points
           \param[in] zPoints is an \ref array with the z-coordinate points
           \param[in] xDirs is an \ref array with the x-coordinate directions at the points
           \param[in] yDirs is an \ref array with the y-coordinate directions at the points
           \param[in] zDirs is an \ref array with the z-coordinate directions at the points
           \param[in] title parameter is used when this function is called in grid mode

           \note All the array inputs must be vectors and must have the size sizes.

           \ingroup gfx_func_draw
         */
        void vectorField(const array& xPoints, const array& yPoints, const array& zPoints,
                         const array& xDirs  , const array& yDirs  , const array& zDirs  ,
                         const char* const title = NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Renders the input arrays as a 2D vector field plot to the window

           \param[in] xPoints is an \ref array with the x-coordinate points
           \param[in] yPoints is an \ref array with the y-coordinate points
           \param[in] xDirs is an \ref array with the x-coordinate directions at the points
           \param[in] yDirs is an \ref array with the y-coordinate directions at the points
           \param[in] title parameter is used when this function is called in grid mode

           \note All the array inputs must be vectors and must have the size sizes.

           \ingroup gfx_func_draw
         */
        void vectorField(const array& xPoints, const array& yPoints,
                         const array& xDirs  , const array& yDirs  ,
                         const char* const title = NULL);
#endif

#if AF_API_VERSION >= 34
        /**
           Setup the axes limits for a 2D histogram/plot/vector field

           This function computes the minimum and maximum for each dimension

           \param[in] x the data to compute the limits for x-axis.
           \param[in] y the data to compute the limits for y-axis.
           \param[in] exact is for using the exact min/max values from \p x and \p y.
                      If exact is false then the most significant digit is rounded up
                      to next power of 2 and the magnitude remains the same.

           \ingroup gfx_func_window
        */
        void setAxesLimits(const array &x, const array &y, const bool exact = false);
#endif

#if AF_API_VERSION >= 34
        /**
           Setup the axes limits for a histogram/plot/surface/vector field

           This function computes the minimum and maximum for each dimension

           \param[in] x the data to compute the limits for x-axis.
           \param[in] y the data to compute the limits for y-axis.
           \param[in] z the data to compute the limits for z-axis.
           \param[in] exact is for using the exact min/max values from \p x and \p y.
                      If exact is false then the most significant digit is rounded up
                      to next power of 2 and the magnitude remains the same.

           \ingroup gfx_func_window
        */
        void setAxesLimits(const array &x, const array &y, const array &z,
                           const bool exact = false);
#endif

#if AF_API_VERSION >= 34
        /**
           Setup the axes limits for a histogram/plot/surface/vector field

           This function sets the axes limits to the ones provided by the user.

           \param[in] xmin is the minimum on x-axis
           \param[in] xmax is the maximum on x-axis
           \param[in] ymin is the minimum on y-axis
           \param[in] ymax is the maximum on y-axis
           \param[in] exact is for using the exact min/max values from \p x and \p y.
                      If exact is false then the most significant digit is rounded up
                      to next power of 2 and the magnitude remains the same.

           \ingroup gfx_func_window
        */
        void setAxesLimits(const float xmin, const float xmax,
                           const float ymin, const float ymax,
                           const bool exact = false);
#endif

#if AF_API_VERSION >= 34
        /**
           Setup the axes limits for a histogram/plot/surface/vector field

           This function sets the axes limits to the ones provided by the user.

           \param[in] xmin is the minimum on x-axis
           \param[in] xmax is the maximum on x-axis
           \param[in] ymin is the minimum on y-axis
           \param[in] ymax is the maximum on y-axis
           \param[in] zmin is the minimum on z-axis
           \param[in] zmax is the maximum on z-axis
           \param[in] exact is for using the exact min/max values from \p x, \p y and \p z.
                      If exact is false then the most significant digit is rounded up
                      to next power of 2 and the magnitude remains the same.

           \ingroup gfx_func_window
        */
        void setAxesLimits(const float xmin, const float xmax,
                           const float ymin, const float ymax,
                           const float zmin, const float zmax,
                           const bool exact = false);
#endif

#if AF_API_VERSION >= 34
        /**
           Setup the axes titles for a plot/surface/vector field

           This function creates the axis titles for a chart.

           \param[in] xtitle is the name of the x-axis
           \param[in] ytitle is the name of the y-axis
           \param[in] ztitle is the name of the z-axis

           \ingroup gfx_func_window
        */
        void setAxesTitles(const char * const xtitle = "X-Axis",
                           const char * const ytitle = "Y-Axis",
                           const char * const ztitle = NULL);
#endif

#if AF_API_VERSION >= 37
        /**
           Setup the axes label formats for charts

           \param[in] xformat is a printf-style format specifier for x-axis
           \param[in] yformat is a printf-style format specifier for y-axis
           \param[in] zformat is a printf-style format specifier for z-axis

           \ingroup gfx_func_window
        */
        void setAxesLabelFormat(const char *const xformat = "4.1%f",
                                const char *const yformat = "4.1%f",
                                const char *const zformat = NULL);
#endif

        /**
           Setup grid layout for multiview mode in a window

           \param[in]   rows is number of rows you want to divide the display area
           \param[in]   cols is number of coloumns you want to divide the display area

           \ingroup gfx_func_window
        */
        void grid(const int rows, const int cols);

        /**
           This function swaps the background buffer to current view
           and polls for any key strokes while the window was in focus

           \ingroup gfx_func_window
        */
        void show();

        /**
           Check if window is marked for close. This usually
           happens when user presses ESC key while the window is in focus.

           \return     \ref AF_SUCCESS if window show is successful, otherwise an appropriate error code
           is returned.

           \ingroup gfx_func_window
        */
        bool close();

#if AF_API_VERSION >= 33
        /**
           Hide/Show the window

           \param[in] isVisible indicates if the window is to be hidden or brought into focus

           \ingroup gfx_func_window
         */
        void setVisibility(const bool isVisible);
#endif

        /**
           This function is used to keep track of which cell in the grid mode is
           being currently rendered. When a user does Window(0,0), we internally
           store the cell coordinates and return a reference to the very object that
           called upon this function. This reference can be used later to issue
           draw calls using rendering functions.

           \param[in] r is row identifier where current object has to be rendered
           \param[in] c is column identifier where current object has to be rendered

           \return a reference to the object pointed by this
           to enable cascading this call with rendering functions.

           \ingroup gfx_window_func
         */
        inline Window& operator()(const int r, const int c) {
            _r = r; _c = c;
            return *this;
        }
};

}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   C Interface wrapper for creating a window

   \param[out]  out is the handle to the created window
   \param[in]   width is the width of the window that will be created
   \param[in]   height is the height of the window that will be created
   \param[in]   title is the window title

   \return     \ref AF_SUCCESS if window creation is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_create_window(af_window *out, const int width, const int height, const char* const title);

/**
   C Interface wrapper for setting the start position when window is displayed

   \param[in]   wind is the window handle
   \param[in]   x is horizontal start coordinate
   \param[in]   y is vertical start coordinate

   \return     \ref AF_SUCCESS if set position for window is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_position(const af_window wind, const unsigned x, const unsigned y);

/**
   C Interface wrapper for setting window title

   \param[in]   wind is the window handle
   \param[in]   title is title of the window

   \return     \ref AF_SUCCESS if set title for window is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_title(const af_window wind, const char* const title);

#if AF_API_VERSION >= 31
/**
   C Interface wrapper for setting window position

   \param[in]   wind is the window handle
   \param[in]   w is target width of the window
   \param[in]   h is target height of the window

   \return     \ref AF_SUCCESS if set size for window is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_size(const af_window wind, const unsigned w, const unsigned h);
#endif

/**
   C Interface wrapper for drawing an array as an image

   \param[in]   wind is the window handle
   \param[in]   in is an \ref af_array
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p in should be 2d array or 3d array with 3 channels.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_image(const af_window wind, const af_array in, const af_cell* const props);

/**
   C Interface wrapper for drawing an array as a plot

   \param[in]   wind is the window handle
   \param[in]   X is an \ref af_array with the x-axis data points
   \param[in]   Y is an \ref af_array with the y-axis data points
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X and \p Y should be vectors.

   \ingroup gfx_func_draw
*/
AF_DEPRECATED("Use af_draw_plot_nd or af_draw_plot_2d instead")
AFAPI af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y, const af_cell* const props);

#if AF_API_VERSION >= 32
/**
   C Interface wrapper for drawing an array as a plot

   \param[in]   wind is the window handle
   \param[in]   P is an \ref af_array or matrix with the xyz-values of the points
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p P should be a 3n x 1 vector or one of a 3xn or nx3 matrices.

   \ingroup gfx_func_draw
*/
AF_DEPRECATED("Use af_draw_plot_nd or af_draw_plot_3d instead")
AFAPI af_err af_draw_plot3(const af_window wind, const af_array P, const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing an array as a 2D or 3D plot

   \param[in]   wind is the window handle
   \param[in]   P is an \ref af_array or matrix with the xyz-values of the points
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p in must be 2d and of the form [n, order], where order is either 2 or 3.
         If order is 2, then chart is 2D and if order is 3, then chart is 3D.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_plot_nd(const af_window wind, const af_array P, const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing an array as a 2D plot

   \param[in]   wind is the window handle
   \param[in]   X is an \ref af_array with the x-axis data points
   \param[in]   Y is an \ref af_array with the y-axis data points
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X and \p Y should be vectors.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_plot_2d(const af_window wind, const af_array X, const af_array Y,
                             const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing an array as a 3D plot

   \param[in]   wind is the window handle
   \param[in]   X is an \ref af_array with the x-axis data points
   \param[in]   Y is an \ref af_array with the y-axis data points
   \param[in]   Z is an \ref af_array with the z-axis data points
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X, \p Y and \p Z should be vectors.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_plot_3d(const af_window wind,
                             const af_array X, const af_array Y, const af_array Z,
                             const af_cell* const props);
#endif

#if AF_API_VERSION >= 33
/**
   C Interface wrapper for drawing an array as a plot

   \param[in]   wind is the window handle
   \param[in]   X is an \ref af_array with the x-axis data points
   \param[in]   Y is an \ref af_array with the y-axis data points
   \param[in]   marker is an \ref af_marker_type enum specifying which marker to use in the scatter plot
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X and \p Y should be vectors.

   \ingroup gfx_func_draw
*/
AF_DEPRECATED("Use af_draw_scatter_nd or af_draw_scatter_2d instead")
AFAPI af_err af_draw_scatter(const af_window wind, const af_array X, const af_array Y,
                             const af_marker_type marker, const af_cell* const props);
#endif

#if AF_API_VERSION >= 33
/**
   C Interface wrapper for drawing an array as a plot

   \param[in]   wind is the window handle
   \param[in]   P is an \ref af_array or matrix with the xyz-values of the points
   \param[in]   marker is an \ref af_marker_type enum specifying which marker to use in the scatter plot
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_draw
*/
AF_DEPRECATED("Use af_draw_scatter_nd or af_draw_scatter_3d instead")
AFAPI af_err af_draw_scatter3(const af_window wind, const af_array P,
                              const af_marker_type marker, const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing an array as a plot

   \param[in]   wind is the window handle
   \param[in]   P is an \ref af_array or matrix with the xyz-values of the points
   \param[in]   marker is an \ref af_marker_type enum specifying which marker to use in the scatter plot
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p in must be 2d and of the form [n, order], where order is either 2 or 3.
         If order is 2, then chart is 2D and if order is 3, then chart is 3D.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_scatter_nd(const af_window wind, const af_array P,
                                const af_marker_type marker, const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing an array as a 2D plot

   \param[in]   wind is the window handle
   \param[in]   X is an \ref af_array with the x-axis data points
   \param[in]   Y is an \ref af_array with the y-axis data points
   \param[in]   marker is an \ref af_marker_type enum specifying which marker to use in the scatter plot
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X and \p Y should be vectors.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_scatter_2d(const af_window wind, const af_array X, const af_array Y,
                                const af_marker_type marker, const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing an array as a 3D plot

   \param[in]   wind is the window handle
   \param[in]   X is an \ref af_array with the x-axis data points
   \param[in]   Y is an \ref af_array with the y-axis data points
   \param[in]   Z is an \ref af_array with the z-axis data points
   \param[in]   marker is an \ref af_marker_type enum specifying which marker to use in the scatter plot
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X, \p Y and \p Z should be vectors.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_scatter_3d(const af_window wind,
                                const af_array X, const af_array Y, const af_array Z,
                                const af_marker_type marker, const af_cell* const props);
#endif

/**
   C Interface wrapper for drawing an array as a histogram

   \param[in]   wind is the window handle
   \param[in]   X is the data frequency \ref af_array
   \param[in]   minval is the value of the minimum data point of the array whose histogram(\p X) is going to be rendered.
   \param[in]   maxval is the value of the maximum data point of the array whose histogram(\p X) is going to be rendered.
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X should be a vector.

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_hist(const af_window wind, const af_array X, const double minval, const double maxval, const af_cell* const props);

#if AF_API_VERSION >= 32
/**
   C Interface wrapper for drawing array's as a surface

   \param[in]   wind is the window handle
   \param[in]   xVals is an \ref af_array with the x-axis data points
   \param[in]   yVals is an \ref af_array with the y-axis data points
   \param[in]   S is an \ref af_array with the z-axis data points
   \param[in]   props is structure \ref af_cell that has the properties that are used
   for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p X and \p Y should be vectors. \p S should be a 2D array

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_surface(const af_window wind, const af_array xVals, const af_array yVals, const af_array S, const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing array's as a 2D or 3D vector field

   \param[in]   wind is the window handle
   \param[in]   points is an \ref af_array with the points
   \param[in]   directions is an \ref af_array with the directions
   \param[in]   props is structure \ref af_cell that has the properties that
                are used for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note \p points and \p directions should have the same size and must
   be 2D.
   The number of rows (dim 0) determines are number of points and the
   number columns determines the type of plot. If the number of columns
   are 2, then the plot is 2D and if there are 3 columns, then the plot
   is 3D.

   \note all the \ref af_array inputs should be vectors and the same size

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_vector_field_nd(const af_window wind,
                const af_array points, const af_array directions,
                const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing array's as a 3D vector field

   \param[in]   wind is the window handle
   \param[in]   xPoints is an \ref af_array with the x-axis points
   \param[in]   yPoints is an \ref af_array with the y-axis points
   \param[in]   zPoints is an \ref af_array with the z-axis points
   \param[in]   xDirs is an \ref af_array with the x-axis directions
   \param[in]   yDirs is an \ref af_array with the y-axis directions
   \param[in]   zDirs is an \ref af_array with the z-axis directions
   \param[in]   props is structure \ref af_cell that has the properties that
                are used for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note all the \ref af_array inputs should be vectors and the same size

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_vector_field_3d(
                const af_window wind,
                const af_array xPoints, const af_array yPoints, const af_array zPoints,
                const af_array xDirs, const af_array yDirs, const af_array zDirs,
                const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for drawing array's as a 2D vector field

   \param[in]   wind is the window handle
   \param[in]   xPoints is an \ref af_array with the x-axis points
   \param[in]   yPoints is an \ref af_array with the y-axis points
   \param[in]   xDirs is an \ref af_array with the x-axis directions
   \param[in]   yDirs is an \ref af_array with the y-axis directions
   \param[in]   props is structure \ref af_cell that has the properties that
                are used for the current rendering.

   \return     \ref AF_SUCCESS if rendering is successful, otherwise an appropriate error code
   is returned.

   \note all the \ref af_array inputs should be vectors and the same size

   \ingroup gfx_func_draw
*/
AFAPI af_err af_draw_vector_field_2d(
                const af_window wind,
                const af_array xPoints, const af_array yPoints,
                const af_array xDirs, const af_array yDirs,
                const af_cell* const props);
#endif

/**
   C Interface wrapper for grid setup in a window

   \param[in]   wind is the window handle
   \param[in]   rows is number of rows you want to show in a window
   \param[in]   cols is number of coloumns you want to show in a window

   \return     \ref AF_SUCCESS if grid setup for window is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_grid(const af_window wind, const int rows, const int cols);

#if AF_API_VERSION >= 34
/**
   C Interface for setting axes limits for a histogram/plot/surface/vector field

   This function computes the minimum and maximum for each dimension

   \param[in] wind is the window handle
   \param[in] x the data to compute the limits for x-axis.
   \param[in] y the data to compute the limits for y-axis.
   \param[in] z the data to compute the limits for z-axis.
   \param[in] exact is for using the exact min/max values from \p x, \p y and \p z.
              If exact is false then the most significant digit is rounded up
              to next power of 2 and the magnitude remains the same.
   \param[in] props is structure \ref af_cell that has the properties that
              are used for the current rendering.

   \note Set \p to NULL if the chart is 2D.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_axes_limits_compute(const af_window wind,
                                        const af_array x, const af_array y, const af_array z,
                                        const bool exact,
                                        const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface for setting axes limits for a 2D histogram/plot/vector field

   This function sets the axes limits to the ones provided by the user.

   \param[in] wind is the window handle
   \param[in] xmin is the minimum on x-axis
   \param[in] xmax is the maximum on x-axis
   \param[in] ymin is the minimum on y-axis
   \param[in] ymax is the maximum on y-axis
   \param[in] exact is for using the exact min/max values from \p x, and \p y.
              If exact is false then the most significant digit is rounded up
              to next power of 2 and the magnitude remains the same.
   \param[in] props is structure \ref af_cell that has the properties that
              are used for the current rendering.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_axes_limits_2d(const af_window wind,
                                   const float xmin, const float xmax,
                                   const float ymin, const float ymax,
                                   const bool exact,
                                   const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface for setting axes limits for a 3D plot/surface/vector field

   This function sets the axes limits to the ones provided by the user.

   \param[in] wind is the window handle
   \param[in] xmin is the minimum on x-axis
   \param[in] xmax is the maximum on x-axis
   \param[in] ymin is the minimum on y-axis
   \param[in] ymax is the maximum on y-axis
   \param[in] zmin is the minimum on z-axis
   \param[in] zmax is the maximum on z-axis
   \param[in] exact is for using the exact min/max values from \p x, \p y and \p z.
              If exact is false then the most significant digit is rounded up
              to next power of 2 and the magnitude remains the same.
   \param[in] props is structure \ref af_cell that has the properties that
              are used for the current rendering.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_axes_limits_3d(const af_window wind,
                                   const float xmin, const float xmax,
                                   const float ymin, const float ymax,
                                   const float zmin, const float zmax,
                                   const bool exact,
                                   const af_cell* const props);
#endif

#if AF_API_VERSION >= 34
/**
   C Interface wrapper for setting axes titles for histogram/plot/surface/vector
   field

   Passing correct value to \p ztitle dictates the right behavior when it comes
   to setting the axes titles appropriately.  If the user is targeting a two
   dimensional chart on the window \p wind, then the user needs to pass NULL to
   \p ztitle so that internal caching mechanism understands this window requires
   a 2D chart. Any non NULL value passed to \p ztitle will result in ArrayFire
   thinking the \p wind intends to use a 3D chart.

   \param[in] wind is the window handle
   \param[in] xtitle is the name of the x-axis
   \param[in] ytitle is the name of the y-axis
   \param[in] ztitle is the name of the z-axis
   \param[in] props is structure \ref af_cell that has the properties that
              are used for the current rendering.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_axes_titles(const af_window wind,
                                const char * const xtitle,
                                const char * const ytitle,
                                const char * const ztitle,
                                const af_cell* const props);
#endif

#if AF_API_VERSION >= 37
/**
   C Interface wrapper for setting axes labels formats for charts

   Axes labels use printf style format specifiers. Default specifier for the
   data displayed as labels is `%4.1f`. This function lets the user change this
   label formatting to whichever format that fits their data range and precision.

   \param[in] wind is the window handle
   \param[in] xformat is a printf-style format specifier for x-axis
   \param[in] yformat is a printf-style format specifier for y-axis
   \param[in] zformat is a printf-style format specifier for z-axis
   \param[in] props is structure \ref af_cell that has the properties that
              are used for the current rendering.

   \note \p zformat can be NULL in which case ArrayFire understands that the
   label formats are meant for a 2D chart corresponding to this \p wind
   or a specific cell in multi-viewport mode (provided via \p props argument).
   A non NULL value to \p zformat means the label formats belong to a 3D chart.

   \ingroup gfx_func_window
*/
AFAPI af_err af_set_axes_label_format(const af_window wind,
                                      const char *const xformat,
                                      const char *const yformat,
                                      const char *const zformat,
                                      const af_cell *const props);
#endif

/**
   C Interface wrapper for showing a window

   \param[in] wind is the window handle

   \return \ref AF_SUCCESS if window show is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_show(const af_window wind);

/**
   C Interface wrapper for checking if window is marked for close

   \param[out]  out is a boolean which indicates whether window is marked for close. This usually
                happens when user presses ESC key while the window is in focus.
   \param[in]   wind is the window handle

   \return     \ref AF_SUCCESS if \p wind show is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_is_window_closed(bool *out, const af_window wind);

#if AF_API_VERSION >= 33
/**
   Hide/Show a window

   \param[in] wind is the window whose visibility is to be changed
   \param[in] is_visible indicates if the window is to be hidden or brought into focus

   \ingroup gfx_func_window
 */
AFAPI af_err af_set_visibility(const af_window wind, const bool is_visible);
#endif

/**
   C Interface wrapper for destroying a window handle

   \param[in]   wind is the window handle

   \return     \ref AF_SUCCESS if window destroy is successful, otherwise an appropriate error code
   is returned.

   \ingroup gfx_func_window
*/
AFAPI af_err af_destroy_window(const af_window wind);

#ifdef __cplusplus
}

#endif
