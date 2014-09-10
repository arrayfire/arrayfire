#pragma once
#include <af/defines.h>
#include <cl.hpp>

#ifdef __cplusplus
extern "C" {
#endif

    AFAPI af_err af_info();

    AFAPI af_err af_get_device_count(int *num_of_devices);

    AFAPI af_err af_set_device(int device);

#ifdef __cplusplus
}
#endif
