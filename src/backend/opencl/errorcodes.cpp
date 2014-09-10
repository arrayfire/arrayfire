#include <errorcodes.hpp>
#include <cl.hpp>

std::string getErrorMessage(int error_code)
{
    switch(error_code)
    {
        case CL_SUCCESS                                  : return std::string("CL_SUCCESS");
        case CL_DEVICE_NOT_FOUND                         : return std::string("CL_DEVICE_NOT_FOUND");
        case CL_DEVICE_NOT_AVAILABLE                     : return std::string("CL_DEVICE_NOT_AVAILABLE");
        case CL_COMPILER_NOT_AVAILABLE                   : return std::string("CL_COMPILER_NOT_AVAILABLE");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE            : return std::string("CL_MEM_OBJECT_ALLOCATION_FAILURE");
        case CL_OUT_OF_RESOURCES                         : return std::string("CL_OUT_OF_RESOURCES");
        case CL_OUT_OF_HOST_MEMORY                       : return std::string("CL_OUT_OF_HOST_MEMORY");
        case CL_PROFILING_INFO_NOT_AVAILABLE             : return std::string("CL_PROFILING_INFO_NOT_AVAILABLE");
        case CL_MEM_COPY_OVERLAP                         : return std::string("CL_MEM_COPY_OVERLAP");
        case CL_IMAGE_FORMAT_MISMATCH                    : return std::string("CL_IMAGE_FORMAT_MISMATCH");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED               : return std::string("CL_IMAGE_FORMAT_NOT_SUPPORTED");
        case CL_BUILD_PROGRAM_FAILURE                    : return std::string("CL_BUILD_PROGRAM_FAILURE");
        case CL_MAP_FAILURE                              : return std::string("CL_MAP_FAILURE");
        case CL_MISALIGNED_SUB_BUFFER_OFFSET             : return std::string("CL_MISALIGNED_SUB_BUFFER_OFFSET");
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return std::string("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
        case CL_INVALID_VALUE                            : return std::string("CL_INVALID_VALUE");
        case CL_INVALID_DEVICE_TYPE                      : return std::string("CL_INVALID_DEVICE_TYPE");
        case CL_INVALID_PLATFORM                         : return std::string("CL_INVALID_PLATFORM");
        case CL_INVALID_DEVICE                           : return std::string("CL_INVALID_DEVICE");
        case CL_INVALID_CONTEXT                          : return std::string("CL_INVALID_CONTEXT");
        case CL_INVALID_QUEUE_PROPERTIES                 : return std::string("CL_INVALID_QUEUE_PROPERTIES");
        case CL_INVALID_COMMAND_QUEUE                    : return std::string("CL_INVALID_COMMAND_QUEUE");
        case CL_INVALID_HOST_PTR                         : return std::string("CL_INVALID_HOST_PTR");
        case CL_INVALID_MEM_OBJECT                       : return std::string("CL_INVALID_MEM_OBJECT");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          : return std::string("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
        case CL_INVALID_IMAGE_SIZE                       : return std::string("CL_INVALID_IMAGE_SIZE");
        case CL_INVALID_SAMPLER                          : return std::string("CL_INVALID_SAMPLER");
        case CL_INVALID_BINARY                           : return std::string("CL_INVALID_BINARY");
        case CL_INVALID_BUILD_OPTIONS                    : return std::string("CL_INVALID_BUILD_OPTIONS");
        case CL_INVALID_PROGRAM                          : return std::string("CL_INVALID_PROGRAM");
        case CL_INVALID_PROGRAM_EXECUTABLE               : return std::string("CL_INVALID_PROGRAM_EXECUTABLE");
        case CL_INVALID_KERNEL_NAME                      : return std::string("CL_INVALID_KERNEL_NAME");
        case CL_INVALID_KERNEL_DEFINITION                : return std::string("CL_INVALID_KERNEL_DEFINITION");
        case CL_INVALID_KERNEL                           : return std::string("CL_INVALID_KERNEL");
        case CL_INVALID_ARG_INDEX                        : return std::string("CL_INVALID_ARG_INDEX");
        case CL_INVALID_ARG_VALUE                        : return std::string("CL_INVALID_ARG_VALUE");
        case CL_INVALID_ARG_SIZE                         : return std::string("CL_INVALID_ARG_SIZE");
        case CL_INVALID_KERNEL_ARGS                      : return std::string("CL_INVALID_KERNEL_ARGS");
        case CL_INVALID_WORK_DIMENSION                   : return std::string("CL_INVALID_WORK_DIMENSION");
        case CL_INVALID_WORK_GROUP_SIZE                  : return std::string("CL_INVALID_WORK_GROUP_SIZE");
        case CL_INVALID_WORK_ITEM_SIZE                   : return std::string("CL_INVALID_WORK_ITEM_SIZE");
        case CL_INVALID_GLOBAL_OFFSET                    : return std::string("CL_INVALID_GLOBAL_OFFSET");
        case CL_INVALID_EVENT_WAIT_LIST                  : return std::string("CL_INVALID_EVENT_WAIT_LIST");
        case CL_INVALID_EVENT                            : return std::string("CL_INVALID_EVENT");
        case CL_INVALID_OPERATION                        : return std::string("CL_INVALID_OPERATION");
        case CL_INVALID_GL_OBJECT                        : return std::string("CL_INVALID_GL_OBJECT");
        case CL_INVALID_BUFFER_SIZE                      : return std::string("CL_INVALID_BUFFER_SIZE");
        case CL_INVALID_MIP_LEVEL                        : return std::string("CL_INVALID_MIP_LEVEL");
        case CL_INVALID_GLOBAL_WORK_SIZE                 : return std::string("CL_INVALID_GLOBAL_WORK_SIZE");
        case CL_INVALID_PROPERTY                         : return std::string("CL_INVALID_PROPERTY");
        default                                          : return std::string("Unkown error code");
    }
}
