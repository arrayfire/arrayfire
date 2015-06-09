/*
* Copyright (C) 2011 Florian Rathgeber, florian.rathgeber@gmail.com
*
* This code is licensed under the MIT License.  See the FindCUDA.cmake script
* for the text of the license.
*
* Based on code by Christopher Bruns published on Stack Overflow (CC-BY):
* http://stackoverflow.com/questions/2285185
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <iterator>
#include <set>

int main() {
    int deviceCount;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess)
    {
        printf("Couldn't get device count: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    std::set<int> computes;
    typedef std::set<int>::iterator iter;

    // machines with no GPUs can still report one emulation device
    for (int device = 0; device < deviceCount; ++device) {
        int major = 9999, minor = 9999;
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) { // 9999 means emulation only
            ++gpuDeviceCount;
            major = properties.major;
            minor = properties.minor;
            if ((major == 2 && minor == 1)) {
                // There is no --arch compute_21 flag for nvcc, so force minor to 0
                minor = 0;
            }
            computes.insert(10 * major + minor);
        }
    }
    int i = 0;
    for(iter it = computes.begin(); it != computes.end(); it++, i++) {
        if(i > 0) {
            printf(" ");
        }
        printf("%d", *it);
    }
    /* don't just return the number of gpus, because other runtime cuda
    errors can also yield non-zero return values */
    if (gpuDeviceCount <= 0 || computes.size() <= 0) {
        return 1; // failure
    }
    return 0; // success
}
