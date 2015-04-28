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
#include <fghistogram1d.hpp>
#include <err_cpu.hpp>
#include <stdexcept>
#include <graphics_common.hpp>
#include <reduce.hpp>
#include <memory.hpp>

using af::dim4;

namespace cpu
{
    template<typename T>
    void copy_histogram(Array<uint> &P, const unsigned int nbins, const double minval, const double maxval, fg::Histogram* hist)
    {
        CheckGL("Before CopyArrayToVBO");

        glBindBuffer(GL_ARRAY_BUFFER, hist->vbo());

        struct point{
            float x;
            float y;
        };

        point graph[(nbins*4)];
        unsigned int* histData = P.get();
        float xmin = minval;
        float dx = (maxval - minval)/nbins;
        int count = 0;
        for (int i = 0; i < (nbins); i++){
            graph[count].x   = xmin;
            graph[count++].y = 0;
            graph[count].x   = xmin;
            graph[count++].y = histData[i];
            graph[count].x   = xmin+dx;
            graph[count++].y = histData[i];
            graph[count].x   = xmin+dx;
            graph[count++].y = 0;
            xmin = xmin + dx;
        }
        size_t bytes = sizeof(graph);
        if(bytes != hist->size()) {
            glBufferData(GL_ARRAY_BUFFER, sizeof(graph), graph, GL_STATIC_DRAW);
            hist->setVBOSize(sizeof(graph));
        } else {
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(graph), graph);
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        CheckGL("In CopyArrayToVBO");
    }

    #define INSTANTIATE(T)  \
        template void copy_histogram<T>(Array<uint> &P, const unsigned int nbins, const double minval, const double maxval, fg::Histogram* hist);

    INSTANTIATE(float)
    INSTANTIATE(int)
    INSTANTIATE(uint)
    INSTANTIATE(uchar)

}

#endif  // WITH_GRAPHICS
