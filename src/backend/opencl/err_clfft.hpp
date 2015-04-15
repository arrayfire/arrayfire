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
#include <err_common.hpp>
#include <clFFT.h>

static const char * _clfftGetResultString(clfftStatus st)
{
    switch (st)
    {
        case CLFFT_SUCCESS: return "Success";
        case CLFFT_DEVICE_NOT_FOUND: return "Device Not Found";
        case CLFFT_DEVICE_NOT_AVAILABLE: return "Device Not Available";
        case CLFFT_COMPILER_NOT_AVAILABLE: return "Compiler Not Available";
        case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory Object Allocation Failure";
        case CLFFT_OUT_OF_RESOURCES: return "Out of Resources";
        case CLFFT_OUT_OF_HOST_MEMORY: return "Out of Host Memory";
        case CLFFT_PROFILING_INFO_NOT_AVAILABLE: return "Profiling Information Not Available";
        case CLFFT_MEM_COPY_OVERLAP: return "Memory Copy Overlap";
        case CLFFT_IMAGE_FORMAT_MISMATCH: return "Image Format Mismatch";
        case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED: return "Image Format Not Supported";
        case CLFFT_BUILD_PROGRAM_FAILURE: return "Build Program Failure";
        case CLFFT_MAP_FAILURE: return "Map Failure";
        case CLFFT_INVALID_VALUE: return "Invalid Value";
        case CLFFT_INVALID_DEVICE_TYPE: return "Invalid Device Type";
        case CLFFT_INVALID_PLATFORM: return "Invalid Platform";
        case CLFFT_INVALID_DEVICE: return "Invalid Device";
        case CLFFT_INVALID_CONTEXT: return "Invalid Context";
        case CLFFT_INVALID_QUEUE_PROPERTIES: return "Invalid Queue Properties";
        case CLFFT_INVALID_COMMAND_QUEUE: return "Invalid Command Queue";
        case CLFFT_INVALID_HOST_PTR: return "Invalid Host Pointer";
        case CLFFT_INVALID_MEM_OBJECT: return "Invalid Memory Object";
        case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid Image Format Descriptor";
        case CLFFT_INVALID_IMAGE_SIZE: return "Invalid Image Size";
        case CLFFT_INVALID_SAMPLER: return "Invalid Sampler";
        case CLFFT_INVALID_BINARY: return "Invalid Binary";
        case CLFFT_INVALID_BUILD_OPTIONS: return "Invalid Build Options";
        case CLFFT_INVALID_PROGRAM: return "Invalid Program";
        case CLFFT_INVALID_PROGRAM_EXECUTABLE: return "Invalid Program Executable";
        case CLFFT_INVALID_KERNEL_NAME: return "Invalid Kernel Name";
        case CLFFT_INVALID_KERNEL_DEFINITION: return "Invalid Kernel Definition";
        case CLFFT_INVALID_KERNEL: return "Invalid Kernel";
        case CLFFT_INVALID_ARG_INDEX: return "Invalid Argument Index";
        case CLFFT_INVALID_ARG_VALUE: return "Invalid Argument Value";
        case CLFFT_INVALID_ARG_SIZE: return "Invalid Argument Size";
        case CLFFT_INVALID_KERNEL_ARGS: return "Invalid Kernel Arguments";
        case CLFFT_INVALID_WORK_DIMENSION: return "Invalid Work Dimension";
        case CLFFT_INVALID_WORK_GROUP_SIZE: return "Invalid Work Group Size";
        case CLFFT_INVALID_WORK_ITEM_SIZE: return "Invalid Work Item Size";
        case CLFFT_INVALID_GLOBAL_OFFSET: return "Invalid Global Offset";
        case CLFFT_INVALID_EVENT_WAIT_LIST: return "Invalid Event Wait List";
        case CLFFT_INVALID_EVENT: return "Invalid Event";
        case CLFFT_INVALID_OPERATION: return "Invalid Operation";
        case CLFFT_INVALID_GL_OBJECT: return "Invalid GL Object";
        case CLFFT_INVALID_BUFFER_SIZE: return "Invalid Buffer Size";
        case CLFFT_INVALID_MIP_LEVEL: return "Invalid MIP Level";
        case CLFFT_INVALID_GLOBAL_WORK_SIZE: return "Invalid Global Work Size";
        case CLFFT_BUGCHECK: return "Bugcheck";
        case CLFFT_NOTIMPLEMENTED: return "Not implemented";
        case CLFFT_TRANSPOSED_NOTIMPLEMENTED: return "Transpose not implemented for this transformation";
        case CLFFT_FILE_NOT_FOUND: return "File not found";
        case CLFFT_FILE_CREATE_FAILURE: return "File creation failed";
        case CLFFT_VERSION_MISMATCH: return "Version mismatch";
        case CLFFT_INVALID_PLAN: return "Invalid plan";
        case CLFFT_DEVICE_NO_DOUBLE: return "Device does not support double precision";
        case CLFFT_DEVICE_MISMATCH: return "Plan device mismatch";
        case CLFFT_ENDSTATUS: return "End status";
    }

    return "Unknown error";
}

#define CLFFT_CHECK(fn) do {                    \
        clfftStatus _clfft_st = fn;             \
        if (_clfft_st != CLFFT_SUCCESS) {       \
            garbageCollect();                   \
            _clfft_st = (fn);                   \
        }                                       \
        if (_clfft_st != CLFFT_SUCCESS) {       \
            char clfft_st_msg[1024];            \
            snprintf(clfft_st_msg,              \
                     sizeof(clfft_st_msg),      \
                     "clFFT Error (%d): %s\n",  \
                     (int)(_clfft_st),          \
                     _clfftGetResultString(     \
                         _clfft_st));           \
                                                \
            AF_ERROR(clfft_st_msg,              \
                     AF_ERR_INTERNAL);          \
        }                                       \
    } while(0)

