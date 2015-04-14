/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#if defined (WITH_GRAPHICS)

#include <Array.hpp>
#include <graphics.hpp>
#include <err_cpu.hpp>
#include <cstdio>
#include <stdexcept>
#include <graphics_common.hpp>
#include <join.hpp>
#include <reduce.hpp>
#include <iostream>
#include <memory.hpp>

using namespace std;
using af::dim4;

namespace cpu
{
    template<typename T>
    void copy_plot(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot)
    {
        CheckGL("Before CopyArrayToVBO");
        const T *d_X = X.get();
        const T *d_Y = Y.get();

        T xmax = reduce_all<af_max_t,T, T>(X);
        T xmin = reduce_all<af_min_t,T, T>(X);
        T ymax = reduce_all<af_max_t,T, T>(Y);
        T ymin = reduce_all<af_min_t,T, T>(Y);


        // Plot size
        af::dim4 Xdim = X.dims();
        af::dim4 Ydim = Y.dims();

        T *Z = memAlloc<T>(Xdim[0] + Ydim[0]);

        for(int i=0; i < (int) Xdim[0]; i++){
            Z[2*i]   = d_X[i];
            Z[2*i+1] = d_Y[i];
        }

        glGenBuffers(1, &(plot->gl_vbo));
        glBindBuffer(GL_ARRAY_BUFFER, plot->gl_vbo);
        glBufferData(GL_ARRAY_BUFFER, ( X.elements()+Y.elements() ) * sizeof (T), Z, GL_STATIC_DRAW);

        CheckGL("In CopyArrayToVBO");

        fg_plot2d(plot, xmax, xmin, ymax, ymin, ( X.elements() ));
        memFree(Z);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    }

    #define INSTANTIATE(T)  \
        template void copy_plot<T>(const Array<T> &X, const Array<T> &Y, const fg_plot_handle plot);

    INSTANTIATE(float)
    INSTANTIATE(double)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)
}

#endif  // WITH_GRAPHICS
