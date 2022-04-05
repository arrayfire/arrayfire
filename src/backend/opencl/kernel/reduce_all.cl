/*******************************************************
 * Copyright (c) 2021, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
// careful w/__threadfence substitution!
// http://www.whatmannerofburgeristhis.com/blog/opencl-vs-cuda-gpu-memory-fences/

kernel void reduce_all_kernel(global To *oData, KParam oInfo,
                              global int* retirementCount, global To *tmp, KParam tmpInfo,
                              const global Ti *iData, KParam iInfo,
                              uint groups_x, uint groups_y, uint repeat,
                              int change_nan, To nanval) {

    const uint tidx = get_local_id(0);
    const uint tidy = get_local_id(1);
    const uint tid  = tidy * DIMX + tidx;

    const uint zid       = get_group_id(0) / groups_x;
    const uint groupId_x = get_group_id(0) - (groups_x)*zid;
    const uint xid       = groupId_x * get_local_size(0) * repeat + tidx;

    const uint wid       = get_group_id(1) / groups_y;
    const uint groupId_y = get_group_id(1) - (groups_y)*wid;
    const uint yid       = groupId_y * get_local_size(1) + tidy;

    local To s_val[THREADS_PER_GROUP];
    local bool amLast;

    iData += wid * iInfo.strides[3] + zid * iInfo.strides[2] +
             yid * iInfo.strides[1] + iInfo.offset;

    bool cond =
        (yid < iInfo.dims[1]) && (zid < iInfo.dims[2]) && (wid < iInfo.dims[3]);


    int last   = (xid + repeat * DIMX);
    int lim    = last > iInfo.dims[0] ? iInfo.dims[0] : last;

    To out_val = init;
    for (int id = xid; cond && id < lim; id += DIMX) {
        To in_val = transform(iData[id]);
        if (change_nan) in_val = !IS_NAN(in_val) ? in_val : nanval;
        out_val = binOp(in_val, out_val);
    }

    s_val[tid] = out_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (THREADS_PER_GROUP == 256) {
        if (tid < 128) s_val[tid] = binOp(s_val[tid], s_val[tid + 128]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (THREADS_PER_GROUP >= 128) {
        if (tid < 64) s_val[tid] = binOp(s_val[tid], s_val[tid + 64]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (THREADS_PER_GROUP >= 64) {
        if (tid < 32) s_val[tid] = binOp(s_val[tid], s_val[tid + 32]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid < 16) s_val[tid] = binOp(s_val[tid], s_val[tid + 16]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8) s_val[tid] = binOp(s_val[tid], s_val[tid + 8]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 4) s_val[tid] = binOp(s_val[tid], s_val[tid + 4]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 2) s_val[tid] = binOp(s_val[tid], s_val[tid + 2]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 1) s_val[tid] = binOp(s_val[tid], s_val[tid + 1]);
    barrier(CLK_LOCAL_MEM_FENCE);


    const unsigned total_blocks = (get_num_groups(0) * get_num_groups(1) * get_num_groups(2));
    const int uubidx = (get_num_groups(0) * get_num_groups(1)) * get_group_id(2)
                       + (get_num_groups(0) * get_group_id(1)) + get_group_id(0);
    if (cond && tid == 0) {
        if(total_blocks != 1) {
            tmp[uubidx] = s_val[0];
        } else {
            oData[0] = s_val[0];
        }
    }

    // Last block to perform final reduction
    if (total_blocks > 1) {

        mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        // Thread 0 takes a ticket
        if (tid == 0) {
            unsigned int ticket = atomic_inc(retirementCount);
            // If the ticket ID == number of blocks, we are the last block
            amLast = (ticket == (total_blocks - 1));
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (amLast) {
            int i = tid;
            To fout_val = init;

            while (i < total_blocks) {
                To in_val = tmp[i];
                fout_val = binOp(in_val, fout_val);
                i += THREADS_PER_GROUP;
            }

            s_val[tid] = fout_val;
            barrier(CLK_LOCAL_MEM_FENCE);

            // reduce final block
            if (THREADS_PER_GROUP == 256) {
                if (tid < 128) s_val[tid] = binOp(s_val[tid], s_val[tid + 128]);
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (THREADS_PER_GROUP >= 128) {
                if (tid < 64) s_val[tid] = binOp(s_val[tid], s_val[tid + 64]);
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (THREADS_PER_GROUP >= 64) {
                if (tid < 32) s_val[tid] = binOp(s_val[tid], s_val[tid + 32]);
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (tid < 16) s_val[tid] = binOp(s_val[tid], s_val[tid + 16]);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < 8) s_val[tid] = binOp(s_val[tid], s_val[tid + 8]);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < 4) s_val[tid] = binOp(s_val[tid], s_val[tid + 4]);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < 2) s_val[tid] = binOp(s_val[tid], s_val[tid + 2]);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid < 1) s_val[tid] = binOp(s_val[tid], s_val[tid + 1]);
            barrier(CLK_LOCAL_MEM_FENCE);

            if (tid == 0) {
                oData[0] = s_val[0];

                // reset retirement count so that next run succeeds
                retirementCount[0] = 0;
            }
        }
    }
}
