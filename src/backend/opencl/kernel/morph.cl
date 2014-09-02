#if T == double || U == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

struct Params {
    dim_type  windLen;
    dim_type     dim0;
    dim_type     dim1;
    dim_type     dim2;
    dim_type   offset;
    dim_type istride0;
    dim_type istride1;
    dim_type istride2;
    dim_type istride3;
    dim_type ostride0;
    dim_type ostride1;
    dim_type ostride2;
    dim_type ostride3;
};

dim_type lIdx(dim_type x, dim_type y,
        dim_type stride1, dim_type stride0)
{
    return (y*stride1 + x*stride0);
}

void load2LocalMem(__local T *  shrd,
        __global const T *      in,
        dim_type lx, dim_type ly, dim_type shrdStride,
        dim_type dim0, dim_type dim1,
        dim_type gx, dim_type gy,
        dim_type inStride1, dim_type inStride0)
{
    dim_type gx_  = clamp(gx, (long)0, dim0-1);
    dim_type gy_  = clamp(gy, (long)0, dim1-1);
    shrd[ lIdx(lx, ly, shrdStride, 1) ] = in[ lIdx(gx_, gy_, inStride1, inStride0) ];
}

//kernel assumes four dimensions
//doing this to reduce one uneccesary parameter
__kernel
void morph(__global T *              out,
           __global const T *        in,
           __constant const T *      d_filt,
           __local T *               localMem,
           __constant struct Params* params,
           dim_type nonBatchBlkSize)
{
    const dim_type se_len = params->windLen;
    const dim_type halo   = se_len/2;
    const dim_type padding= 2*halo;
    const dim_type shrdLen= get_local_size(0) + padding + 1;

    // gfor batch offsets
    dim_type batchId    = get_group_id(0) / nonBatchBlkSize;
    in  += (batchId * params->istride2 + params->offset);
    out += (batchId * params->ostride2);

    // local neighborhood indices
    const dim_type lx = get_local_id(0);
    const dim_type ly = get_local_id(1);

    // global indices
    dim_type gx = get_local_size(0) * (get_group_id(0)-batchId*nonBatchBlkSize) + lx;
    dim_type gy = get_local_size(1) * get_group_id(1) + ly;

    // offset values for pulling image to local memory
    dim_type lx2      = lx + get_local_size(0);
    dim_type ly2      = ly + get_local_size(1);
    dim_type gx2      = gx + get_local_size(0);
    dim_type gy2      = gy + get_local_size(1);

    // pull image to local memory
    load2LocalMem(localMem, in, lx, ly, shrdLen,
                  params->dim0, params->dim1,
                  gx-halo, gy-halo,
                  params->istride1, params->istride0);
    if (lx<padding) {
        load2LocalMem(localMem, in, lx2, ly, shrdLen,
                      params->dim0, params->dim1,
                      gx2-halo, gy-halo,
                      params->istride1, params->istride0);
    }
    if (ly<padding) {
        load2LocalMem(localMem, in, lx, ly2, shrdLen,
                      params->dim0, params->dim1,
                      gx-halo, gy2-halo,
                      params->istride1, params->istride0);
    }
    if (lx<padding && ly<padding) {
        load2LocalMem(localMem, in, lx2, ly2, shrdLen,
                      params->dim0, params->dim1,
                      gx2-halo, gy2-halo,
                      params->istride1, params->istride0);
    }

    dim_type i = lx + halo;
    dim_type j = ly + halo;
    __syncthreads();

    T acc = localMem[ lIdx(i, j, shrdLen, 1) ];
#pragma unroll
    for(dim_type wj=0; wj<params->windLen; ++wj) {
        dim_type joff   = wj*se_len;
        dim_type w_joff = (j+wj-halo)*shrdLen;
#pragma unroll
        for(dim_type wi=0; wi<params->windLen; ++wi) {
            T cur  = localMem[w_joff+i+wi-halo];
            if (d_filt[joff+wi]) {
                if (isDilation)
                    acc = max(acc, cur);
                else
                    acc = min(acc, cur);
            }
        }
    }

    if (gx<params->dim0 && gy<params->dim1) {
        dim_type outIdx = lIdx(gx, gy, params->ostride1, params->ostride0);
        out[outIdx] = acc;
    }
}
