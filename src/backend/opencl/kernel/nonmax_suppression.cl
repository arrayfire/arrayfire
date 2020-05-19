/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void nonMaxSuppressionKernel(global T* output, KParam oInfo,
                                      global const T* in, KParam inInfo,
                                      global const T* dx, KParam dxInfo,
                                      global const T* dy, KParam dyInfo,
                                      unsigned nBBS0, unsigned nBBS1) {
    // local thread indices
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx = get_local_size(0) * (get_group_id(0) - b2 * nBBS0) + lx;
    const int gy = get_local_size(1) * (get_group_id(1) - b3 * nBBS1) + ly;

    local T localMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];

    global const T* mag =
        in + (b2 * inInfo.strides[2] + b3 * inInfo.strides[3] + inInfo.offset);
    global const T* dX =
        dx + (b2 * dxInfo.strides[2] + b3 * dxInfo.strides[3] + dxInfo.offset) +
        dxInfo.strides[1] + 1;
    global const T* dY =
        dy + (b2 * dyInfo.strides[2] + b3 * dyInfo.strides[3] + dyInfo.offset) +
        dyInfo.strides[1] + 1;
    global T* out = output + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) +
                      oInfo.strides[1] + 1;

#pragma unroll
    for (int b = ly, gy2 = gy; b < SHRD_MEM_HEIGHT && gy2 < inInfo.dims[1];
         b += get_local_size(1), gy2 += get_local_size(1)) {
#pragma unroll
        for (int a = lx, gx2 = gx; a < SHRD_MEM_WIDTH && gx2 < inInfo.dims[0];
             a += get_local_size(0), gx2 += get_local_size(0)) {
            localMem[b][a] =
                mag[(gx2)*inInfo.strides[0] + (gy2)*inInfo.strides[1]];
        }
    }
    int i = lx + 1;
    int j = ly + 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < inInfo.dims[0] - 2 && gy < inInfo.dims[1] - 2) {
        int idx = gx * inInfo.strides[0] + gy * inInfo.strides[1];

        const float cmag = localMem[j][i];

        if (cmag == 0.0f)
            out[idx] = (T)0;
        else {
            const float dx = dX[idx];
            const float dy = dY[idx];
            const float se = localMem[j + 1][i + 1];
            const float nw = localMem[j - 1][i - 1];
            const float ea = localMem[j][i + 1];
            const float we = localMem[j][i - 1];
            const float ne = localMem[j - 1][i + 1];
            const float sw = localMem[j + 1][i - 1];
            const float no = localMem[j - 1][i];
            const float so = localMem[j + 1][i];

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
                    const bool isDxMagGreater = (dx + dy) >= 0;

                    a1    = isDxMagGreater ? so : we;
                    a2    = isDxMagGreater ? no : ea;
                    b1    = sw;
                    b2    = ne;
                    alpha = isDxMagGreater ? -dx / dy : dy / -dx;
                } else {
                    const bool isDyMagGreater = (-dx + dy) >= 0;

                    a1    = isDyMagGreater ? we : no;
                    a2    = isDyMagGreater ? ea : so;
                    b1    = nw;
                    b2    = se;
                    alpha = isDyMagGreater ? dy / dx : dx / dy;
                }
            }

            float mag1 = (1 - alpha) * a1 + alpha * b1;
            float mag2 = (1 - alpha) * a2 + alpha * b2;

            if (cmag > mag1 && cmag > mag2) {
                out[idx] = cmag;
            } else {
                out[idx] = (T)0;
            }
        }
    }
}
