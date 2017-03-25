/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

constant int STRONG = 1;
constant int WEAK   = 2;
constant int NOEDGE = 0;

int lIdx(int x, int y, int stride0, int stride1)
{
    return (x*stride0 + y*stride1);
}


#if defined(NON_MAX_SUPPRESSION)
__kernel
void nonMaxSuppressionKernel(__global     T* output, KParam   oInfo,
                             __global const T*   in, KParam  inInfo,
                             __global const T*   dx, KParam  dxInfo,
                             __global const T*   dy, KParam  dyInfo,
                             unsigned nBBS0, unsigned nBBS1)
{
    // local thread indices
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    __local T localMem[SHRD_MEM_HEIGHT][SHRD_MEM_WIDTH];

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global const T* mag = in +
                              (b2 * inInfo.strides[2] + b3 * inInfo.strides[3] + inInfo.offset) +
                              inInfo.strides[1] + 1;
    __global const T* dX  = dx  +
                              (b2 * dxInfo.strides[2]  + b3 * dxInfo.strides[3] + dxInfo.offset) +
                              dxInfo.strides[1] + 1;
    __global const T* dY  = dy  +
                              (b2 * dyInfo.strides[2]  + b3 * dyInfo.strides[3] + dyInfo.offset) +
                              dyInfo.strides[1] + 1;
    __global     T*   out = output +
                              (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) + oInfo.strides[1] + 1;

    // pull image to local memory
#pragma unroll
    for (int b=ly, gy2=gy; b<SHRD_MEM_HEIGHT; b+=get_local_size(1), gy2+=get_local_size(1))
#pragma unroll
        for (int a=lx, gx2=gx; a<SHRD_MEM_WIDTH; a+=get_local_size(0), gx2+=get_local_size(0))
            localMem[b][a] = mag[ lIdx(gx2-1, gy2-1, inInfo.strides[0], inInfo.strides[1]) ];

    int i = lx + 1;
    int j = ly + 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx<inInfo.dims[0]-2 && gy<inInfo.dims[1]-2)
    {
        int idx = lIdx(gx, gy, inInfo.strides[0], inInfo.strides[1]);

        const float cmag = localMem[j][i];

        if (cmag == 0.0f)
            out[idx] = (T)0;
        else {
            const float dx = dX[idx];
            const float dy = dY[idx];
            const float se = localMem[j+1][i+1];
            const float nw = localMem[j-1][i-1];
            const float ea = localMem[j  ][i+1];
            const float we = localMem[j  ][i-1];
            const float ne = localMem[j-1][i+1];
            const float sw = localMem[j+1][i-1];
            const float no = localMem[j-1][i  ];
            const float so = localMem[j+1][i  ];

            float a1, a2, b1, b2, alpha;

            if (dx>=0) {
                if (dy>=0) {
                    const bool isTrue = (dx-dy)>=0;

                    a1    = isTrue ? ea : so;
                    a2    = isTrue ? we : no;
                    b1    = se;
                    b2    = nw;
                    alpha = isTrue ? dy/dx : dx/dy;
                } else {
                    const bool isTrue = (dx+dy)>=0;

                    a1    = isTrue ? ea : no;
                    a2    = isTrue ? we : so;
                    b1    = ne;
                    b2    = sw;
                    alpha = isTrue ? -dy/dx : dx/-dy;
                }
            } else {
                if (dy>=0) {
                    const bool isTrue = (dx+dy)>=0;

                    a1    = isTrue ? so : we;
                    a2    = isTrue ? no : ea;
                    b1    = sw;
                    b2    = ne;
                    alpha = isTrue ? -dx/dy : dy/-dx;
                } else {
                    const bool isTrue = (-dx+dy)>=0;

                    a1    = isTrue ? we : no;
                    a2    = isTrue ? ea : so;
                    b1    = nw;
                    b2    = se;
                    alpha = isTrue ? -dy/dx : dx/-dy;
                }
            }

            float mag1 = (1-alpha)*a1 + alpha*b1;
            float mag2 = (1-alpha)*a2 + alpha*b2;

            if (cmag>mag1 && cmag>mag2) {
                out[idx] = cmag;
            } else {
                out[idx] = (T)0;
            }
        }
    }
}
#endif

#if defined(INIT_EDGE_OUT)
__kernel
void initEdgeOutKernel(__global T*       output, KParam oInfo,
                       __global const T* strong, KParam sInfo,
                       __global const T*   weak, KParam wInfo,
                       unsigned nBBS0, unsigned nBBS1)
{
    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + get_local_id(0);
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + get_local_id(1);

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global const T* wPtr = weak   +
        (b2 * wInfo.strides[2] + b3 * wInfo.strides[3] + wInfo.offset) + wInfo.strides[1] + 1;

    __global const T* sPtr = strong +
        (b2 * sInfo.strides[2] + b3 * sInfo.strides[3] + sInfo.offset) + sInfo.strides[1] + 1;

    __global       T* oPtr = output +
        (b2 * oInfo.strides[2] + b3 * oInfo.strides[3] + oInfo.offset) + oInfo.strides[1] + 1;

    if (gx<(oInfo.dims[0]-2) && gy<(oInfo.dims[1]-2))
    {
        int idx   = lIdx(gx, gy, oInfo.strides[0], oInfo.strides[1]);
        oPtr[idx] = (sPtr[idx] > 0 ? STRONG : (wPtr[idx] > 0 ? WEAK : NOEDGE));
    }
}
#endif

#define VALID_BLOCK_IDX(j, i) ( (j)>0 && (j)<(SHRD_MEM_HEIGHT-1) && (i)>0 && (i)<(SHRD_MEM_WIDTH-1) )

#if defined(EDGE_TRACER)
__kernel
void edgeTrackKernel(__global T* output, KParam oInfo, unsigned nBBS0, unsigned nBBS1,
                    __global volatile int* hasChanged)
{
    // shared memory with 1 pixel border
    // strong and weak images are binary(char) images thus,
    // occupying only (16+2)*(16+2) = 324 bytes per shared memory tile
    __local int outMem  [ SHRD_MEM_HEIGHT ] [ SHRD_MEM_WIDTH ];
    __local int predicates[TOTAL_NUM_THREADS];

    // local thread indices
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + lx;
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + ly;

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global T* oPtr = output + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) + oInfo.strides[1] + 1;

    // pull image to local memory
#pragma unroll
    for (int b=ly, gy2=gy; b<SHRD_MEM_HEIGHT; b+=get_local_size(1), gy2+=get_local_size(1))
    {
#pragma unroll
        for (int a=lx, gx2=gx; a<SHRD_MEM_WIDTH; a+=get_local_size(0), gx2+=get_local_size(0))
        {
            int x = gx2-1;
            int y = gy2-1;
            if (x>=0 && x<oInfo.dims[0] && y>=0 && y<oInfo.dims[1])
                outMem[b][a] = oPtr[ lIdx(x, y, oInfo.strides[0], oInfo.strides[1]) ];
            else
                outMem[b][a] = NOEDGE;
        }
    }

    int i = lx + 1;
    int j = ly + 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    int tid = get_local_id(0) + get_local_size(0) * get_local_id(1);

    int continueIter = 1;

    while (continueIter)
    {
        int cu = outMem[j][i];
        int nw = outMem[j-1][i-1];
        int no = outMem[j-1][i  ];
        int ne = outMem[j-1][i+1];
        int ea = outMem[j  ][i+1];
        int se = outMem[j+1][i+1];
        int so = outMem[j+1][i  ];
        int sw = outMem[j+1][i-1];
        int we = outMem[j  ][i-1];

        bool hasStrongNeighbour = nw==STRONG || no==STRONG || ne==STRONG || ea==STRONG ||
                                  se==STRONG || so==STRONG || sw==STRONG || we==STRONG;

        if (cu==WEAK && hasStrongNeighbour)
            outMem[j][i] = STRONG;

        barrier(CLK_LOCAL_MEM_FENCE);

        cu = outMem[j][i];

        bool _nw = outMem[j-1][i-1] == WEAK  &&  VALID_BLOCK_IDX(j-1, i-1);
        bool _no = outMem[j-1][i  ] == WEAK  &&  VALID_BLOCK_IDX(j-1, i  );
        bool _ne = outMem[j-1][i+1] == WEAK  &&  VALID_BLOCK_IDX(j-1, i+1);
        bool _ea = outMem[j  ][i+1] == WEAK  &&  VALID_BLOCK_IDX(j  , i+1);
        bool _se = outMem[j+1][i+1] == WEAK  &&  VALID_BLOCK_IDX(j+1, i+1);
        bool _so = outMem[j+1][i  ] == WEAK  &&  VALID_BLOCK_IDX(j+1, i  );
        bool _sw = outMem[j+1][i-1] == WEAK  &&  VALID_BLOCK_IDX(j+1, i-1);
        bool _we = outMem[j  ][i-1] == WEAK  &&  VALID_BLOCK_IDX(j  , i-1);

        bool hasWeakNeighbour = _nw || _no || _ne || _ea || _se || _so || _sw || _we;

        // Following Block is equivalent of __syncthreads_or in CUDA
        predicates[tid] = cu==STRONG && hasWeakNeighbour;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int nt = TOTAL_NUM_THREADS/2;  nt>0; nt>>=1)
        {
            if (tid < nt)
                predicates[tid] = predicates[tid] || predicates[tid+nt];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        continueIter = predicates[0];
    };

    // Check if any 1-pixel border ring
    // has weak pixels with strong candidates
    // within the main region, then increment hasChanged.
    int cu = outMem[j][i];
    int nw = outMem[j-1][i-1];
    int no = outMem[j-1][i  ];
    int ne = outMem[j-1][i+1];
    int ea = outMem[j  ][i+1];
    int se = outMem[j+1][i+1];
    int so = outMem[j+1][i  ];
    int sw = outMem[j+1][i-1];
    int we = outMem[j  ][i-1];

    bool hasWeakNeighbour = nw==WEAK || no==WEAK || ne==WEAK || ea==WEAK ||
                            se==WEAK || so==WEAK || sw==WEAK || we==WEAK;

    // Following Block is equivalent of __syncthreads_or in CUDA
    predicates[tid] = cu==STRONG && hasWeakNeighbour;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nt = TOTAL_NUM_THREADS/2;  nt>0; nt>>=1)
    {
        if (tid < nt)
            predicates[tid] = predicates[tid] || predicates[tid+nt];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    continueIter = predicates[0];

    if (continueIter>0 && lx==0 && ly==0)
        atomic_add(hasChanged, 1);

    // Update output with shared memory result
    if (gx<(oInfo.dims[0]-2) && gy<(oInfo.dims[1]-2))
        oPtr[ lIdx(gx, gy, oInfo.strides[0], oInfo.strides[1]) ] = outMem[j][i];
}
#endif

#if defined(SUPPRESS_LEFT_OVER)
__kernel
void suppressLeftOverKernel(__global T* output, KParam oInfo, unsigned nBBS0, unsigned nBBS1)
{
    // batch offsets for 3rd and 4th dimension
    const unsigned b2 = get_group_id(0) / nBBS0;
    const unsigned b3 = get_group_id(1) / nBBS1;

    // global indices
    const int gx = get_local_size(0) * (get_group_id(0)-b2*nBBS0) + get_local_id(0);
    const int gy = get_local_size(1) * (get_group_id(1)-b3*nBBS1) + get_local_id(1);

    // Offset input and output pointers to second pixel of second coloumn/row
    // to skip the border
    __global T* oPtr = output + (b2 * oInfo.strides[2] + b3 * oInfo.strides[3]) + oInfo.strides[1] + 1;

    if (gx<(oInfo.dims[0]-2) && gy<(oInfo.dims[1]-2))
    {
        int idx = lIdx(gx, gy, oInfo.strides[0], oInfo.strides[1]);
        T val   = oPtr[idx];
        if (val==WEAK)
            oPtr[idx] = NOEDGE;
    }
}
#endif
