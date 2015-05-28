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

typedef unsigned long long af_window;

typedef struct {
    int row;
    int col;
    const char* title;
} af_cell;

#ifdef __cplusplus
namespace af
{

    // FIXME handle copying properly
class AFAPI Window {
    private:
        af_window wnd;
        /* below attributes are used to track which
         * cell in the grid is being rendered currently */
        int _r;
        int _c;

        void initWindow(const int width, const int height, const char* const title);

    public:
        Window();
        Window(const char* const title);
        Window(const int width, const int height, const char* const title="ArrayFire");
        Window(const af_window wnd);
        ~Window();

        af_window get() const { return wnd; }
        void setPos(const unsigned x, const unsigned y);
        void setTitle(const char* const title);

        void image(const array& in, const char* title=NULL);
        void plot(const array& X, const array& Y, const char* const title=NULL);
        void hist(const array& X, const double minval, const double maxval, const char* const title=NULL);
        void grid(const int rows, const int cols);
        void show();

        bool close();

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

AFAPI af_err af_create_window(af_window *out, const int width, const int height, const char* const title);

AFAPI af_err af_set_position(const af_window wind, const unsigned x, const unsigned y);

AFAPI af_err af_set_title(const af_window wind, const char* const title);

AFAPI af_err af_draw_image(const af_window wind, const af_array in, const af_cell* const props);

AFAPI af_err af_draw_plot(const af_window wind, const af_array X, const af_array Y, const af_cell* const props);

AFAPI af_err af_draw_hist(const af_window window, const af_array X, const double minval, const double maxval, const af_cell* const props);

AFAPI af_err af_grid(const af_window wind, const int rows, const int cols);

AFAPI af_err af_show(const af_window window);

AFAPI af_err af_is_window_closed(bool *out, const af_window window);

AFAPI af_err af_destroy_window(const af_window wind);

#ifdef __cplusplus
}

#endif
