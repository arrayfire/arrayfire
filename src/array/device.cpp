#include <af/device.h>
#include "error.hpp"

namespace af
{
    void info()
    {
        AF_THROW(af_info());
    }

    int getDeviceCount()
    {
        int devices = -1;
        AF_THROW(af_get_device_count(&devices));
        return devices;
    }

    void setDevice(const int device)
    {
        AF_THROW(af_set_device(device));
    }
}
