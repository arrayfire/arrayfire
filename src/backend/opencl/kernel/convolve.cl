#if T == double || U == double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

dim_type index(dim_type i, dim_type j, dim_type k, dim_type jstride, dim_type kstride)
{
    return i+j*jstride+k*kstride;
}

T readSrc(global T const *src, dim_type i, dim_type j, dim_type k, dim_type dims[], dim_type strides[])
{
    bool is_i = i>=0 && i<dims[0];
    bool is_j = j>=0 && j<dims[1];
    bool is_k = k>=0 && k<dims[2];
    if (is_i && is_j && is_k)
        return src[(i*strides[0] + j*strides[1] + k*strides[2])];
    else
        return (T)(0);
}

#if BASE_DIM==1
kernel
void convolve(global T *out, KParam oInfo,
              global T const *signal, KParam sInfo, local T *localMem,
              constant T const *impulse, KParam fInfo, dim_type nonBatchBlkSize,
              dim_type oStep, dim_type sStep)
{
    dim_type fLen    = fInfo.dims[0];
    dim_type padding  = fLen-1;
    dim_type shrdLen  = get_local_size(0) + 2*padding;
    unsigned batchId  = get_group_id(0)/nonBatchBlkSize;

    global T *dst = out + oStep +(batchId*oInfo.strides[1]);
    global T const *src = signal + sStep +(batchId*sInfo.strides[1]) + sInfo.offset;

    dim_type gx  = get_local_size(0)*(get_group_id(0)-batchId*nonBatchBlkSize) + get_local_id(0);

    for (dim_type i=0; i<shrdLen; i+=get_local_size(0)) {
        dim_type idx = gx-padding + i;
        dim_type lx  = get_local_id(0)+ i;
        if (lx<shrdLen)
            localMem[lx]  = (idx>=0 && idx<sInfo.dims[0]) ? src[idx*sInfo.strides[0]] : (T)(0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx>=0 && gx<oInfo.dims[0]) {
        dim_type lx   = get_local_id(0) + padding + (EXPAND ? 0 : fLen/2);
        accType accum = (accType)(0);
        for(dim_type f=0; f<fLen; ++f) {
            accum = accum + ((accType)localMem[lx-f] * (accType)impulse[f]);
        }
        dst[gx] = (T)accum;
    }
}
#endif

#if BASE_DIM==2
kernel
void convolve(global T *out, KParam oInfo,
              global T const *signal, KParam sInfo, local T *localMem,
              constant T const *impulse, KParam fInfo, dim_type nonBatchBlkSize,
              dim_type oStep, dim_type sStep)
{
    dim_type fLen0    = fInfo.dims[0];
    dim_type fLen1    = fInfo.dims[1];
    dim_type pad0     = fLen0-1;
    dim_type pad1     = fLen1-1;
    dim_type shrdLen0 = get_local_size(0) + 2*pad0;
    unsigned batchId  = get_group_id(0)/nonBatchBlkSize;

    global T *dst = out + oStep +(batchId*oInfo.strides[2]);
    global T const *src = signal + sStep +(batchId*sInfo.strides[2]) + sInfo.offset;

    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);
    dim_type gx = get_local_size(0) * (get_group_id(0)-batchId*nonBatchBlkSize) + lx;
    dim_type gy = get_local_size(1) * get_group_id(1) + ly;
    dim_type i = lx + pad0;
    dim_type j = ly + pad1;

    localMem[j*shrdLen0+i] = readSrc(src, gx, gy, 0, sInfo.dims, sInfo.strides);

    if (lx < pad0) {
        dim_type gx2 = gx + get_local_size(0);
        dim_type lx2 = i  + get_local_size(0);
        localMem[j*shrdLen0+ lx] = readSrc(src, gx-pad0, gy, 0, sInfo.dims, sInfo.strides);
        localMem[j*shrdLen0+lx2] = readSrc(src, gx2    , gy, 0, sInfo.dims, sInfo.strides);
    }
    if (ly < pad1) {
        dim_type gy2 = gy + get_local_size(1);
        dim_type ly2 = j  + get_local_size(1);
        localMem[ly*shrdLen0 +i] = readSrc(src, gx, gy-pad1, 0, sInfo.dims, sInfo.strides);
        localMem[ly2*shrdLen0+i] = readSrc(src, gx, gy2    , 0, sInfo.dims, sInfo.strides);
    }
    if (lx < pad0 && ly < pad1) {
        dim_type gx2 = gx + get_local_size(0);
        dim_type lx2 = i  + get_local_size(0);
        dim_type gy2 = gy + get_local_size(1);
        dim_type ly2 = j  + get_local_size(1);
        // 4 corner regions
        localMem[ly*shrdLen0+lx  ] = readSrc(src, gx-pad0, gy-pad1, 0, sInfo.dims, sInfo.strides);
        localMem[ly*shrdLen0+lx2 ] = readSrc(src, gx2    , gy-pad1, 0, sInfo.dims, sInfo.strides);
        localMem[ly2*shrdLen0+lx ] = readSrc(src, gx-pad0, gy2    , 0, sInfo.dims, sInfo.strides);
        localMem[ly2*shrdLen0+lx2] = readSrc(src, gx2    , gy2    , 0, sInfo.dims, sInfo.strides);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx>=0 && gx<oInfo.dims[0] && gy>=0 && gy<oInfo.dims[1]) {
        dim_type ci = i + (EXPAND ? 0 : fLen0/2);
        dim_type cj = j + (EXPAND ? 0 : fLen1/2);

        accType accum = (accType)(0);
        for(dim_type fj=0; fj<fLen1; ++fj) {
            for(dim_type fi=0; fi<fLen0; ++fi) {
                T f_val = impulse[fj*fLen0+fi];
                T s_val = localMem[(cj-fj)*shrdLen0+(ci-fi)];
                accum   = accum + ((accType)s_val*(accType)f_val);
            }
        }
        dst[gy*oInfo.strides[1]+gx] = (T)accum;
    }
}
#endif

#if BASE_DIM==3
kernel
void convolve(global T *out, KParam oInfo, global T const *signal,
              KParam sInfo, local T *localMem, constant T const *impulse,
              KParam fInfo, dim_type nonBatchBlkSize,
              dim_type oStep, dim_type sStep)
{
    dim_type fLen0    = fInfo.dims[0];
    dim_type fLen1    = fInfo.dims[1];
    dim_type fLen2    = fInfo.dims[2];
    dim_type pad0     = fLen0-1;
    dim_type pad1     = fLen1-1;
    dim_type pad2     = fLen2-1;
    dim_type shrdLen0 = get_local_size(0) + 2*pad0;
    dim_type skStride = shrdLen0 * (get_local_size(1) + 2*pad1);
    dim_type fStride  = fLen0 * fLen1;
    unsigned batchId  = get_group_id(0)/nonBatchBlkSize;

    global T *dst = out + oStep +(batchId*oInfo.strides[3]);
    global T const *src = signal + sStep +(batchId*sInfo.strides[3]) + sInfo.offset;

    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);
    dim_type lz = get_local_id(2);
    dim_type gx = get_local_size(0) * (get_group_id(0)-batchId*nonBatchBlkSize) + lx;
    dim_type gy = get_local_size(1) * get_group_id(1) + ly;
    dim_type gz = get_local_size(2) * get_group_id(2) + lz;
    dim_type i = lx + pad0;
    dim_type j = ly + pad1;
    dim_type k = lz + pad2;

    localMem[index(i, j, k, shrdLen0, skStride)] = readSrc(src, gx, gy, gz, sInfo.dims, sInfo.strides);

    { //in the hope of limiting scope of l*2 and g*2 variables
        dim_type gx2 = gx + get_local_size(0);
        dim_type lx2 = i  + get_local_size(0);
        dim_type gy2 = gy + get_local_size(1);
        dim_type ly2 = j  + get_local_size(1);
        dim_type gz2 = gz + get_local_size(2);
        dim_type lz2 = k  + get_local_size(2);
        if (lx < pad0) {
            localMem[index( lx, j, k, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy, gz, sInfo.dims, sInfo.strides);
            localMem[index(lx2, j, k, shrdLen0, skStride)] = readSrc(src, gx2    , gy, gz, sInfo.dims, sInfo.strides);
        }
        if (ly < pad1) {
            localMem[index(i,  ly, k, shrdLen0, skStride)] = readSrc(src, gx, gy-pad1, gz, sInfo.dims, sInfo.strides);
            localMem[index(i, ly2, k, shrdLen0, skStride)] = readSrc(src, gx, gy2    , gz, sInfo.dims, sInfo.strides);
        }
        if (lz < pad2) {
            localMem[index(i, j,  lz, shrdLen0, skStride)] = readSrc(src, gx, gy, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index(i, j, lz2, shrdLen0, skStride)] = readSrc(src, gx, gy, gz2    , sInfo.dims, sInfo.strides);
        }

        if (lx < pad0 && ly < pad1) {
            localMem[index( lx,  ly, k, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy-pad1, gz, sInfo.dims, sInfo.strides);
            localMem[index(lx2,  ly, k, shrdLen0, skStride)] = readSrc(src,     gx2, gy-pad1, gz, sInfo.dims, sInfo.strides);
            localMem[index( lx, ly2, k, shrdLen0, skStride)] = readSrc(src, gx-pad0,     gy2, gz, sInfo.dims, sInfo.strides);
            localMem[index(lx2, ly2, k, shrdLen0, skStride)] = readSrc(src,     gx2,     gy2, gz, sInfo.dims, sInfo.strides);
        }

        if (ly < pad1 && lz < pad2) {
            localMem[index(i,  ly,  lz, shrdLen0, skStride)] = readSrc(src, gx, gy-pad1, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index(i, ly2,  lz, shrdLen0, skStride)] = readSrc(src, gx,     gy2, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index(i,  ly, lz2, shrdLen0, skStride)] = readSrc(src, gx, gy-pad1,     gz2, sInfo.dims, sInfo.strides);
            localMem[index(i, ly2, lz2, shrdLen0, skStride)] = readSrc(src, gx,     gy2,     gz2, sInfo.dims, sInfo.strides);
        }

        if (lz < pad2 && lx < pad0) {
            localMem[index( lx, j,  lz, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index(lx2, j,  lz, shrdLen0, skStride)] = readSrc(src,     gx2, gy, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index( lx, j, lz2, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy,     gz2, sInfo.dims, sInfo.strides);
            localMem[index(lx2, j, lz2, shrdLen0, skStride)] = readSrc(src,     gx2, gy,     gz2, sInfo.dims, sInfo.strides);
        }

        if (lx < pad0 && ly < pad1 && lz < pad2) {
            localMem[index( lx,  ly, lz, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy-pad1, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index(lx2,  ly, lz, shrdLen0, skStride)] = readSrc(src, gx2    , gy-pad1, gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index( lx, ly2, lz, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy2    , gz-pad2, sInfo.dims, sInfo.strides);
            localMem[index(lx2, ly2, lz, shrdLen0, skStride)] = readSrc(src, gx2    , gy2    , gz-pad2, sInfo.dims, sInfo.strides);

            localMem[index( lx,  ly, lz2, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy-pad1, gz2, sInfo.dims, sInfo.strides);
            localMem[index(lx2,  ly, lz2, shrdLen0, skStride)] = readSrc(src, gx2    , gy-pad1, gz2, sInfo.dims, sInfo.strides);
            localMem[index( lx, ly2, lz2, shrdLen0, skStride)] = readSrc(src, gx-pad0, gy2    , gz2, sInfo.dims, sInfo.strides);
            localMem[index(lx2, ly2, lz2, shrdLen0, skStride)] = readSrc(src, gx2    , gy2    , gz2, sInfo.dims, sInfo.strides);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx>=0 && gx<oInfo.dims[0] && gy>=0 && gy<oInfo.dims[1] && gz>=0 && gz<oInfo.dims[2]) {
        dim_type ci = i + (EXPAND ? 0 : fLen0/2);
        dim_type cj = j + (EXPAND ? 0 : fLen1/2);
        dim_type ck = k + (EXPAND ? 0 : fLen2/2);

        accType accum = (accType)(0);
        for(dim_type fk=0; fk<fLen2; ++fk) {
            for(dim_type fj=0; fj<fLen1; ++fj) {
                for(dim_type fi=0; fi<fLen0; ++fi) {
                    T f_val = impulse[index(fi, fj, fk, fLen0, fStride)];
                    T s_val = localMem[index(ci-fi, cj-fj, ck-fk, shrdLen0, skStride)];
                    accum   = accum + ((accType)s_val*(accType)f_val);
                }
            }
        }
        dst[index(gx, gy, gz, oInfo.strides[1], oInfo.strides[2])] = (T)accum;
    }
}
#endif

#if defined(SEPARABLE_CONV)
T readSrc2(global T const *src, dim_type i, dim_type j, dim_type dims[], dim_type strides[])
{
    bool is_i = i>=0 && i<dims[0];
    bool is_j = j>=0 && j<dims[1];
    if (is_i && is_j)
        return src[i*strides[0] + j*strides[1]];
    else
        return (T)(0);
}

kernel
void convolve(global T *out, KParam oInfo, global T const *signal,
              KParam sInfo, local T *localMem, constant T const *impulse,
              dim_type fLen, dim_type nonBatchBlkSize)
{
    dim_type start    = (EXPAND ? 0 : fLen/2);
    dim_type pad      = fLen-1;
    dim_type shrdLen  = get_local_size(0);
    if (CONV_DIM==0) {
        shrdLen += 2*pad;
    }
    unsigned batchId  = get_group_id(0)/nonBatchBlkSize;
    global T *dst = out + (batchId*oInfo.strides[2]);
    global const T *src = signal + (batchId*sInfo.strides[2]) + sInfo.offset;

    dim_type lx = get_local_id(0);
    dim_type ly = get_local_id(1);
    dim_type gx = get_local_size(0) * (get_group_id(0)-batchId*nonBatchBlkSize) + lx;
    dim_type gy = get_local_size(1) * get_group_id(1) + ly;
    dim_type i  = (CONV_DIM==0 ? lx : ly) + pad;

    if (CONV_DIM==0) {
        localMem[ly*shrdLen+i] = readSrc2(src, gx+start, gy+start, sInfo.dims, sInfo.strides);
        if (lx < pad) {
            dim_type gx2 = gx + get_local_size(0);
            dim_type lx2 = i  + get_local_size(0);
            localMem[ly*shrdLen+ lx] = readSrc2(src, gx-pad+start, gy+start, sInfo.dims, sInfo.strides);
            localMem[ly*shrdLen+lx2] = readSrc2(src, gx2+start   , gy+start, sInfo.dims, sInfo.strides);
        }
    } else if (CONV_DIM==1) {
        localMem[i*shrdLen+lx] = readSrc2(src, gx+start, gy+start, sInfo.dims, sInfo.strides);
        if (ly < pad) {
            dim_type gy2 = gy + get_local_size(1);
            dim_type ly2 = i  + get_local_size(1);
            localMem[ ly*shrdLen+lx] = readSrc2(src, gx+start, gy-pad+start, sInfo.dims, sInfo.strides);
            localMem[ly2*shrdLen+lx] = readSrc2(src, gx+start,    gy2+start, sInfo.dims, sInfo.strides);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx>=0 && gx<oInfo.dims[0] && gy>=0 && gy<oInfo.dims[1]) {
        accType accum = (accType)(0);
        for(dim_type f=0; f<fLen; ++f) {
            T f_val = impulse[f];
            dim_type s_idx = (CONV_DIM==0 ? (ly*shrdLen+(i-f)) : ((i-f)*shrdLen+lx));
            T s_val = localMem[s_idx];
            accum   = accum + ((accType)s_val*(accType)f_val);
        }
        dst[gy*oInfo.strides[1]+gx] = (T)accum;
    }
}
#endif
