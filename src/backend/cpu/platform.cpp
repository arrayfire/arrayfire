#include <platform.hpp>
#include <iostream>

using namespace std;

namespace cpu {
    string getInfo()
    {
        string info("Using ArrayFire Open Source for CPU\n");
        return info;
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
