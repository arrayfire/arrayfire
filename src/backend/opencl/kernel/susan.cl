/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define MAX_VAL(A,B) (A) < (B) ? (B) : (A)

#ifdef RESPONSE
kernel
void susan_responses(global T* out, global const T* in,
                     const unsigned idim0, const unsigned idim1,
                     const float t, const float g,
                     const unsigned edge)
{
    const int rSqrd   = RADIUS*RADIUS;
    const int windLen = 2*RADIUS+1;
    const int shrdLen = BLOCK_X + windLen-1;
    local T localMem[LOCAL_MEM_SIZE];

    const unsigned lx = get_local_id(0);
    const unsigned ly = get_local_id(1);
    const unsigned gx = get_global_id(0) + edge;
    const unsigned gy = get_global_id(1) + edge;

    const unsigned nucleusIdx = (ly+RADIUS)*shrdLen + lx+RADIUS;
    if (gx<idim0 && gy<idim1)
        localMem[nucleusIdx] = in[gy*idim0+gx];
    else
        localMem[nucleusIdx] = 0;
    T m_0 = localMem[nucleusIdx];

#pragma unroll
    for (int b=ly, gy2=gy; b<shrdLen; b+=BLOCK_Y, gy2+=BLOCK_Y) {
        int j = gy2-RADIUS;
#pragma unroll
        for (int a=lx, gx2=gx; a<shrdLen; a+=BLOCK_X, gx2+=BLOCK_X) {
            int i = gx2-RADIUS;
            if (i<idim0 && j<idim1)
                localMem[b*shrdLen+a] = in[i+idim0*j];
            else
                localMem[b*shrdLen+a] = m_0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < idim0 - edge && gy < idim1 - edge) {
        unsigned idx = gx + idim0 * gy;
        float nM  = 0.0f;
#pragma unroll
        for (int p=0; p<windLen; ++p) {
#pragma unroll
            for (int q=0; q<windLen; ++q) {
                int i = p - RADIUS;
                int j = q - RADIUS;
                int a = lx + RADIUS + i;
                int b = ly + RADIUS + j;
                if (i*i + j*j < rSqrd) {
                    float c = m_0;
                    float m = localMem[b * shrdLen + a];
                    float exp_pow = pow((m - c)/t, 6.0f);
                    float cM = exp(-exp_pow);
                    nM += cM;
                }
            }
        }
        out[idx] = nM < g ? g - nM : (T)0;
    }
}
#endif

#ifdef NONMAX
kernel
void non_maximal(global float* x_out, global float* y_out,
                 global float* resp_out, global unsigned* count,
                 const unsigned idim0, const unsigned idim1, global const T* resp_in,
                 const unsigned edge, const unsigned max_corners)
{
    // Responses on the border don't have 8-neighbors to compare, discard them
    const unsigned r = edge + 1;

    const unsigned gx = get_global_id(0) + r;
    const unsigned gy = get_global_id(1) + r;

    if (gx < idim0 - r && gy < idim1 - r) {
        const T v = resp_in[gy * idim0 + gx];

        // Find maximum neighborhood response
        T max_v;
        max_v = MAX_VAL(resp_in[(gy-1) * idim0 + gx-1], resp_in[gy * idim0 + gx-1]);
        max_v = MAX_VAL(max_v, resp_in[(gy+1) * idim0 + gx-1]);
        max_v = MAX_VAL(max_v, resp_in[(gy-1) * idim0 + gx  ]);
        max_v = MAX_VAL(max_v, resp_in[(gy+1) * idim0 + gx  ]);
        max_v = MAX_VAL(max_v, resp_in[(gy-1) * idim0 + gx+1]);
        max_v = MAX_VAL(max_v, resp_in[(gy)   * idim0 + gx+1]);
        max_v = MAX_VAL(max_v, resp_in[(gy+1) * idim0 + gx+1]);

        // Stores corner to {x,y,resp}_out if it's response is maximum compared
        // to its 8-neighborhood and greater or equal minimum response
        if (v > max_v) {
            const unsigned idx = atomic_inc(count);
            if (idx < max_corners) {
                x_out[idx]    = (float)gx;
                y_out[idx]    = (float)gy;
                resp_out[idx] = (float)v;
            }
        }
    }
}
#endif
