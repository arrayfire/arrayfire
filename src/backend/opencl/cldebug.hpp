// FIXME: Add a special flag for debug
#ifndef NDEBUG
#include <iostream>
#include <stdio.h>
#include <errorcodes.hpp>

#define CL_FINISH(Q) Q.finish()
#define SHOW_CL_ERROR(ERR) std::cout << ERR.what() << ": " << getErrorMessage(ERR.err()) << std::endl;

#define SHOW_BUILD_INFO(PROG) do {                              \
    std::cout << PROG.getBuildInfo<CL_PROGRAM_BUILD_LOG>(       \
        PROG.getInfo<CL_PROGRAM_DEVICES>()[0]) << std::endl;    \
                                                                \
    std::cout << PROG.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(   \
        PROG.getInfo<CL_PROGRAM_DEVICES>()[0]) << std::endl;    \
    } while(0)                                                  \

#else

#define CL_FINISH(Q)
#define SHOW_CL_ERROR(ERR)
#define SHOW_BUILD_INFO(PROG)

#endif
