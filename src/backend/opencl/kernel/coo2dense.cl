/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void coo2Dense(global T *oPtr, const KParam output, global const T *vPtr,
                      const KParam values, global const int *rPtr,
                      const KParam rowIdx, global const int *cPtr,
                      const KParam colIdx) {
    const int dimSize = get_local_size(0);

    for (int i = get_local_id(0); i < reps * dimSize; i += dimSize) {
        const int id = i + get_group_id(0) * dimSize * reps;
        if (id >= values.dims[0]) return;

        T v   = vPtr[id + values.offset];
        int r = rPtr[id + rowIdx.offset];
        int c = cPtr[id + colIdx.offset];

        int offset = r + c * output.strides[1];

        oPtr[offset] = v;
    }
}
