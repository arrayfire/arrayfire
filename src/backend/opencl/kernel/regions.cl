/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// The initial label kernel distinguishes between valid (nonzero)
// pixels and "background" (zero) pixels.
__kernel
void initial_label(global    T * equiv_map,
                   KParam        eInfo,
                   global char * bin,
                   KParam        bInfo)
{
    const int base_x = (get_group_id(0) * get_local_size(0) * N_PER_THREAD) + get_local_id(0);
    const int base_y = (get_group_id(1) * get_local_size(1) * N_PER_THREAD) + get_local_id(1);

    // If in bounds and a valid pixel, set the initial label.
    #pragma unroll
    for (int xb = 0; xb < N_PER_THREAD; ++xb) {
        #pragma unroll
        for (int yb = 0; yb < N_PER_THREAD; ++yb) {
            const int x = base_x + (xb * get_local_size(0));
            const int y = base_y + (yb * get_local_size(1));
            const int n = y * bInfo.dims[0] + x;
            if (x < bInfo.dims[0] && y < bInfo.dims[1]) {
                equiv_map[n] = (bin[n] > (char)0) ? n + 1 : 0;
            }
        }
    }
}

__kernel
void final_relabel(global       T    * equiv_map,
                   KParam              eInfo,
                   global       char * bin,
                   KParam              bInfo,
                   global const T    * d_tmp)
{
    const int base_x = (get_group_id(0) * get_local_size(0) * N_PER_THREAD) + get_local_id(0);
    const int base_y = (get_group_id(1) * get_local_size(1) * N_PER_THREAD) + get_local_id(1);

    // If in bounds and a valid pixel, set the initial label.
    #pragma unroll
    for (int xb = 0; xb < N_PER_THREAD; ++xb) {
        #pragma unroll
        for (int yb = 0; yb < N_PER_THREAD; ++yb) {
            const int x = base_x + (xb * get_local_size(0));
            const int y = base_y + (yb * get_local_size(1));
            const int n = y * bInfo.dims[0] + x;
            if (x < bInfo.dims[0] && y < bInfo.dims[1]) {
                equiv_map[n] = (bin[n] > (char)0) ? d_tmp[(int)equiv_map[n]] : (T)0;
            }
        }
    }
}

#define MIN(A,B) ((A < B) ? (A) : (B))

// When two labels are equivalent, choose the lower label, but
// do not choose zero, which indicates invalid.
//#if T == double
static inline T relabel(const T a, const T b) {
    return MIN((a + (LIMIT_MAX * (a == 0))),(b + (LIMIT_MAX * (b == 0))));
}

// The following kernel updates the equivalency map.  This kernel
// must be launched with a square block configuration with
// correctly instantiated template parameters.

// The following is the preferred configuration for Kepler:
// BLOCK_DIM = 16 // 256 threads per block
// NUM_WARPS = 8; // (Could compute this from block dim)
// Number of elements to handle per thread in each dimension
// N_PER_THREAD = 2; // 2x2 per thread = 4 total elems per thread
__kernel
void update_equiv(global T*   equiv_map,
                  KParam      eInfo,
                  global int* continue_flag)
{
    // Basic coordinates
    const int base_x = (get_group_id(0) * get_local_size(0) * N_PER_THREAD) + get_local_id(0);
    const int base_y = (get_group_id(1) * get_local_size(1) * N_PER_THREAD) + get_local_id(1);

    const int width  = eInfo.dims[0];
    const int height = eInfo.dims[1];

    // Per element write flags and label, initially 0
    char      write[N_PER_THREAD * N_PER_THREAD];
    T    best_label[N_PER_THREAD * N_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < N_PER_THREAD * N_PER_THREAD; ++i) {
        write[i]      = (char)0;
        best_label[i] = (T)0;
    }

    // Cached tile of the equivalency map
    __local T s_tile[N_PER_THREAD*BLOCK_DIM][(N_PER_THREAD*BLOCK_DIM)];

    // Space to track ballot funcs to track convergence
    __local int s_changed[NUM_WARPS];

    const int tn = (get_local_id(1) * get_local_size(0)) + get_local_id(0);

    const int warpSize = 32;
    const int warpIdx = tn / warpSize;
    s_changed[warpIdx] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    __local int tid_changed[NUM_WARPS];
    tid_changed[warpIdx] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int xb = 0; xb < N_PER_THREAD; ++xb) {
        #pragma unroll
        for (int yb = 0; yb < N_PER_THREAD; ++yb) {

            // Indexing variables
            const int x = base_x + (xb * get_local_size(0));
            const int y = base_y + (yb * get_local_size(1));
            const int tx = get_local_id(0) + (xb * get_local_size(0));
            const int ty = get_local_id(1) + (yb * get_local_size(1));
            const int tid_i = xb * N_PER_THREAD + yb;
            const int n = y * width + x;

            // Get the label for this pixel if we're  in bounds
            const T orig_label = (x < width && y < height) ?
                equiv_map[n] : (T)0;
            s_tile[ty][tx] = orig_label;

            // Find the lowest label of the nearest valid pixel
            // So far, all we know is that this pixel is valid.
            best_label[tid_i] = orig_label;

            if (orig_label != (T)0) {

                const int south_y = min(y, height-2) + 1;
                const int north_y = max(y, 1) - 1;
                const int east_x = min(x, width-2) + 1;
                const int west_x = max(x, 1) - 1;

                // Check bottom
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[(south_y) * width + x]);

                // Check right neighbor
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[y * width + east_x]);

                // Check left neighbor
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[y * width + west_x]);

                // Check top neighbor
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[(north_y) * width + x]);

#ifdef FULL_CONN
                // Check NW corner
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[(north_y) * width + west_x]);

                // Check NE corner
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[(north_y) * width + east_x]);

                // Check SW corner
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[(south_y) * width + west_x]);

                // Check SE corner
                best_label[tid_i] = relabel(best_label[tid_i],
                        equiv_map[(south_y) * width + east_x]);
#endif // if connectivity == 8
            } // if orig_label != 0

            // Process the equivalency list.
            T last_label = orig_label;
            T new_label  = best_label[tid_i];

            while (best_label[tid_i] != (T)0 && new_label < last_label) {
                last_label = new_label;
                new_label = equiv_map[(int)new_label - 1];
            }

            if (orig_label != new_label) {
                tid_changed[warpIdx] = 1;
                s_tile[ty][tx] = new_label;
                write[tid_i] = (char)1;
            }
            best_label[tid_i] = new_label;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Determine if any pixel changed
    unsigned int continue_iter = 0;
    s_changed[warpIdx] = tid_changed[warpIdx];
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int i = 0; i < NUM_WARPS; i++)
        continue_iter = continue_iter || (s_changed[i] != 0);

    // Iterate until no pixel in the tile changes
    while (continue_iter != 0) {

        // Reset whether or not this thread's pixels have changed.
        tid_changed[warpIdx] = 0;

        #pragma unroll
        for (int xb = 0; xb < N_PER_THREAD; ++xb) {
            #pragma unroll
            for (int yb = 0; yb < N_PER_THREAD; ++yb) {

                // Indexing
                const int tx = get_local_id(0) + (xb * get_local_size(0));
                const int ty = get_local_id(1) + (yb * get_local_size(1));
                const int tid_i = xb * N_PER_THREAD + yb;

                T last_label = best_label[tid_i];

                if (best_label[tid_i] != 0) {

                    const int north_y   = max(ty, 1) - 1;
                    const int south_y   = min(ty, N_PER_THREAD*BLOCK_DIM - 2) + 1;
                    const int east_x    = min(tx, N_PER_THREAD*BLOCK_DIM - 2) + 1;
                    const int west_x    = max(tx, 1) - 1;

                    // Check bottom
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[south_y][tx]);

                    // Check right neighbor
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[ty][east_x]);

                    // Check left neighbor
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[ty][west_x]);

                    // Check top neighbor
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[north_y][tx]);

#ifdef FULL_CONN
                    // Check NW corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[north_y][west_x]);

                    // Check NE corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[north_y][east_x]);

                    // Check SW corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[south_y][west_x]);

                    // Check SE corner
                    best_label[tid_i] = relabel(best_label[tid_i],
                                                s_tile[south_y][east_x]);
#endif
                }
                // This thread's value changed  this iteration if the
                // best label is not the same as the last label.
                const int changed = best_label[tid_i] != last_label;
                write[tid_i] |= changed;
                atomic_or(&tid_changed[warpIdx], changed);
            }
        }
        // Done looking at neighbors for this iteration
        barrier(CLK_LOCAL_MEM_FENCE);

        // Decide if we need to continue iterating
        s_changed[warpIdx] = tid_changed[warpIdx];
        barrier(CLK_LOCAL_MEM_FENCE);
        continue_iter = 0;
        #pragma unroll
        for (int i = 0; i < NUM_WARPS; i++)
            continue_iter |= (s_changed[i] != 0);

        // If we have to continue iterating, update the tile of the
        // equiv map in shared memory
        if (continue_iter != 0) {
            #pragma unroll
            for (int xb = 0; xb < N_PER_THREAD; ++xb) {
                #pragma unroll
                for (int yb = 0; yb < N_PER_THREAD; ++yb) {
                    const int tx = get_local_id(0) + (xb * get_local_size(0));
                    const int ty = get_local_id(1) + (yb * get_local_size(1));
                    const int tid_i = xb * N_PER_THREAD + yb;
                    // Update tile in shared memory
                    s_tile[ty][tx] = best_label[tid_i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    } // while (continue_iter)

    // Write out equiv_map
    #pragma unroll
    for (int xb = 0; xb < N_PER_THREAD; ++xb) {
        #pragma unroll
        for (int yb = 0; yb < N_PER_THREAD; ++yb) {
            const int x = base_x + (xb * get_local_size(0));
            const int y = base_y + (yb * get_local_size(1));
            const int n = y * width + x;
            const int tid_i = xb * N_PER_THREAD + yb;
            if (x < width && y < height && write[tid_i]) {
                equiv_map[n]  = best_label[tid_i];
                *continue_flag = 1;
            }
        }
    }
}
