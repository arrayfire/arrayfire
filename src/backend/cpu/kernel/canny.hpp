/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <array>
#include <cassert>
#include <list>

namespace arrayfire {
namespace cpu {
namespace kernel {
template<typename T>
void nonMaxSuppression(Param<T> output, CParam<T> magnitude, CParam<T> dxParam,
                       CParam<T> dyParam) {
    const af::dim4 dims    = magnitude.dims();
    const af::dim4 strides = magnitude.strides();

    T* out       = output.get();
    const T* mag = magnitude.get();
    const T* dX  = dxParam.get();
    const T* dY  = dyParam.get();

    for (dim_t b3 = 0; b3 < dims[3]; ++b3) {
        for (dim_t b2 = 0; b2 < dims[2]; ++b2) {
            dim_t offset;

            offset = dims[0] + 1;

            for (dim_t j = 2; j < dims[1]; ++j, offset += 2) {
                for (dim_t i = 2; i < dims[0]; ++i, ++offset) {
                    T curr = mag[offset];
                    if (curr == 0) {
                        out[offset] = (T)0;
                    } else {
                        const float se = mag[offset + dims[0] + 1];
                        const float nw = mag[offset - dims[0] - 1];
                        const float ea = mag[offset + 1];
                        const float we = mag[offset - 1];
                        const float ne = mag[offset - dims[0] + 1];
                        const float sw = mag[offset + dims[0] - 1];
                        const float no = mag[offset - dims[0]];
                        const float so = mag[offset + dims[0]];
                        const float dx = dX[offset];
                        const float dy = dY[offset];

                        float a1, a2, b1, b2, alpha;

                        if (dx >= 0) {
                            if (dy >= 0) {
                                const bool isDxMagGreater = (dx - dy) >= 0;

                                a1    = isDxMagGreater ? ea : so;
                                a2    = isDxMagGreater ? we : no;
                                b1    = se;
                                b2    = nw;
                                alpha = isDxMagGreater ? dy / dx : dx / dy;
                            } else {
                                const bool isDyMagGreater = (dx + dy) >= 0;

                                a1    = isDyMagGreater ? ea : no;
                                a2    = isDyMagGreater ? we : so;
                                b1    = ne;
                                b2    = sw;
                                alpha = isDyMagGreater ? -dy / dx : dx / -dy;
                            }
                        } else {
                            if (dy >= 0) {
                                const bool isDyMagGreater = (dx + dy) >= 0;

                                a1    = isDyMagGreater ? so : we;
                                a2    = isDyMagGreater ? no : ea;
                                b1    = sw;
                                b2    = ne;
                                alpha = isDyMagGreater ? -dx / dy : dy / -dx;
                            } else {
                                const bool isDxMagGreater = (-dx + dy) >= 0;

                                a1    = isDxMagGreater ? we : no;
                                a2    = isDxMagGreater ? ea : so;
                                b1    = nw;
                                b2    = se;
                                alpha = isDxMagGreater ? dy / dx : dx / dy;
                            }
                        }

                        float mag1 = (1.0f - alpha) * a1 + alpha * b1;
                        float mag2 = (1.0f - alpha) * a2 + alpha * b2;

                        if (curr > mag1 && curr > mag2) {
                            out[offset] = curr;
                        } else {
                            out[offset] = (T)0;
                        }
                    }
                }
            }

            out += strides[2];
            mag += strides[2];
            dX += strides[2];
            dY += strides[2];
        }
        out += strides[3];
        mag += strides[3];
        dX += strides[3];
        dY += strides[3];
    }
}

template<typename T>
void traceEdge(T* out, const T* strong, const T* weak, int t, int stride1) {
    if (!out || !strong || !weak) return;

    const T EDGE = 1;

    std::list<dim_t> edges;  // list of edges to be checked
    edges.push_back(t);

    do {
        t = edges.front();
        edges.pop_front();  // remove the last after read

        // get indices of 8 neighbours
        std::array<dim_t, 8> potentials;

        potentials[0] = t - stride1 - 1;    // north-west
        potentials[1] = potentials[0] + 1;  // north
        potentials[2] = potentials[1] + 1;  // north-east
        potentials[3] = t - 1;              // west
        potentials[4] = t + 1;              // east
        potentials[5] = t + stride1 - 1;    // south-west
        potentials[6] = potentials[5] + 1;  // south
        potentials[7] = potentials[6] + 1;  // south-east

        // test 8 neighbours and add them into edge
        // list only if they are also edges
        for (auto it : potentials) {
            if (weak[it] > 0 && out[it] != EDGE) {
                out[it] = EDGE;
                edges.emplace_back(it);
            }
        }
    } while (!edges.empty());
}

template<typename T>
void edgeTrackingHysteresis(Param<T> out, CParam<T> strong, CParam<T> weak) {
    const af::dim4 dims    = strong.dims();
    const dim_t batchCount = dims[2] * dims[3];
    const dim_t jMax       = dims[1] - 1;
    const dim_t iMax       = dims[0] - 1;

    const T* sptr = strong.get();
    const T* wptr = weak.get();
    T* optr       = out.get();

    for (dim_t batchId = 0; batchId < batchCount; ++batchId) {
        // Skip processing borders
        dim_t t = dims[0] + 1;

        for (dim_t j = 1; j <= jMax; ++j) {
            for (dim_t i = 1; i <= iMax; ++i, ++t) {
                // if current pixel(sptr) is part of a edge
                // and output doesn't have it marked already,
                // mark it and trace the pixels from here.
                if (sptr[t] > 0 && optr[t] != 1) {
                    optr[t] = 1;
                    traceEdge(optr, sptr, wptr, t, dims[0]);
                }
            }
        }
        optr += out.strides(2);
        sptr += strong.strides(2);
        wptr += weak.strides(2);
    }
}
}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
