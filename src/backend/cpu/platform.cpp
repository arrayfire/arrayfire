#include <af/version.h>
#include <platform.hpp>
#include <iostream>
#include <sstream>

using namespace std;

namespace cpu {
    static const char *get_system(void)
    {
        return
    #if defined(ARCH_32)
        "32-bit "
    #elif defined(ARCH_64)
        "64-bit "
    #endif

    #if defined(OS_LNX)
        "Linux";
    #elif defined(OS_WIN)
        "Windows";
    #elif defined(OS_MAC)
        "Mac OSX";
    #endif
    }

    string getInfo()
    {
        ostringstream info;
        info << "ArrayFire v" << AF_VERSION << AF_VERSION_MINOR
             << " (CPU, " << get_system() << ", build " << REVISION << ")" << std::endl;
        return info.str();
    }

    int getDeviceCount()
    {
        return 1;
    }

    int setDevice(int device)
    {
        static bool flag;
        if(!flag) {
            std::cout << "WARNING: af_set_device not supported for CPU" << std::endl;
            flag = 1;
        }
        return 1;
    }
}
