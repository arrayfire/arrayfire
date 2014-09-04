#if T == double || U == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

struct Params {
    dim_type      offset;
    dim_type    idims[4];
    dim_type istrides[4];
    dim_type ostrides[4];
};

__kernel
void histogram(__global outType *         d_dst,
               __global const inType *    d_src,
               __global const float2 *    d_minmax,
               __constant struct Params * params,
               __local outType *          localMem,
               dim_type len, dim_type nbins, dim_type blk_x)
{
    // offset minmax array to account for batch ops
    __global const float2 * d_mnmx = d_minmax + get_group_id(1);

    // offset input and output to account for batch ops
    __global const inType *in = d_src + get_group_id(1) * params->istrides[2] + params->offset;
    __global outType * out    = d_dst + get_group_id(1) * params->ostrides[2];

    dim_type start = get_group_id(0) * THRD_LOAD * get_local_size(0) + get_local_id(0);
    dim_type end   = min((start + THRD_LOAD * get_local_size(0)), len);

    __local float minval;
    __local float dx;

    if (get_local_id(0) == 0) {
        float2 minmax = *d_mnmx;
        minval = minmax.s0;
        dx     = (minmax.s1-minmax.s0) / (float)nbins;
    }

    for (dim_type i = get_local_id(0); i < nbins; i += get_local_size(0))
        localMem[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int row = start; row < end; row += get_local_size(0)) {
        int bin = (int)(((float)in[row] - minval) / dx);
        bin     = max(bin, 0);
        bin     = min(bin, (int)nbins-1);
        atomic_inc((localMem + bin));
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (dim_type i = get_local_id(0); i < nbins; i += get_local_size(0)) {
        atomic_add((out + i), localMem[i]);
    }
}
