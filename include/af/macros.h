/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <stdio.h>

///
/// Print a line on screen using printf syntax.
/// Usage: Uses same syntax and semantics as printf.
/// Output: \<filename\>:\<line number\>: \<message\>
///
#ifndef AF_MSG
#define AF_MSG(fmt,...) do {            \
        printf("%s:%d: " fmt "\n",      \
                 __FILE__, __LINE__, ##__VA_ARGS__);      \
        } while (0);
#endif

/**
 * AF_MEM_INFO macro can be used to print the current stats of ArrayFire's memory
 * manager.
 *
 * AF_MEM_INFO print 4 values:
 *
 * ---------------------------------------------------
 *  Name                    | Description
 * -------------------------|-------------------------
 *  Allocated Bytes         | Total number of bytes allocated by the memory manager
 *  Allocated Buffers       | Total number of buffers allocated
 *  Locked (In Use) Bytes   | Number of bytes that are in use by active arrays
 *  Locked (In Use) Buffers | Number of buffers that are in use by active arrays
 * ---------------------------------------------------
 *
 *  The `Allocated Bytes` is always a multiple of the memory step size. The
 *  default step size is 1024 bytes. This means when a buffer is to be
 *  allocated, the size is always rounded up to a multiple of the step size.
 *  You can use af::getMemStepSize() to check the current step size and
 *  af::setMemStepSize() to set a custom resolution size.
 *
 *  The `Allocated Buffers` is the number of buffers that use up the allocated
 *  bytes. This includes buffers currently in scope, as well as buffers marked
 *  as free, ie, from arrays gone out of scope. The free buffers are available
 *  for use by new arrays that might be created.
 *
 *  The `Locked Bytes` is the number of bytes in use that cannot be
 *  reallocated at the moment. The difference of Allocated Bytes and Locked
 *  Bytes is the total bytes available for reallocation.
 *
 *  The `Locked Buffers` is the number of buffer in use that cannot be
 *  reallocated at the moment. The difference of Allocated Buffers and Locked
 *  Buffers is the number of buffers available for reallocation.
 *
 * The AF_MEM_INFO macro can accept a string an argument that is printed to screen
 *
 * \param[in] msg (Optional) A message that is printed to screen
 *
 * \code
 *     AF_MEM_INFO("At start");
 * \endcode
 *
 * Output:
 *
 *     AF Memory at /workspace/myfile.cpp:41: At Start
 *     Allocated [ Bytes | Buffers ] = [ 4096 | 4 ]
 *     In Use    [ Bytes | Buffers ] = [ 2048 | 2 ]
 */
#define AF_MEM_INFO(msg) do {                                                           \
    size_t abytes = 0, abuffs = 0, lbytes = 0, lbuffs = 0;                              \
    af_err err = af_device_mem_info(&abytes, &abuffs, &lbytes, &lbuffs);                \
    if(err == AF_SUCCESS) {                                                             \
        printf("AF Memory at %s:%d: " msg "\n", __FILE__, __LINE__);                    \
        printf("Allocated [ Bytes | Buffers ] = [ %ld | %ld ]\n", abytes, abuffs);      \
        printf("In Use    [ Bytes | Buffers ] = [ %ld | %ld ]\n", lbytes, lbuffs);      \
    } else {                                                                            \
        fprintf(stderr, "AF Memory at %s:%d: " msg "\nAF Error %d\n",                   \
                __FILE__, __LINE__, err);                                               \
    }                                                                                   \
} while(0)
