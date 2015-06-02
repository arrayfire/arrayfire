/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef MAGMA_SYNC_H
#define MAGMA_SYNC_H

#ifndef check_error
#define check_error( err ) if (err != CL_SUCCESS) { printf ("OpenCL err: %d\n", err); throw cl::Error(err); }
#endif

static inline void
magma_event_sync( magma_event_t event )
{
    cl_int err = clWaitForEvents(1, &event);
    check_error(err);
}

static inline void
magma_queue_sync( magma_queue_t queue )
{
    cl_int err = clFinish( queue );
    check_error(err);
    err = clFlush( queue );
    check_error(err)
}

#endif
