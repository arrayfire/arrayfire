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
        return -1;
    }
}
