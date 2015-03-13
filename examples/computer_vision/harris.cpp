/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cstdio>
#include <arrayfire.h>
#include <cstdlib>

using namespace af;

static void harris_demo(bool console)
{
    // Load image
    array img = loadImage(ASSETS_DIR"/examples/images/square.png", false);

    // Calculate image gradients
    array ix, iy;
    grad(ix, iy, img);

    // Compute second-order derivatives
    array ixx = ix * ix;
    array ixy = ix * iy;
    array iyy = iy * iy;

    // Compute a Gaussian kernel with standard deviation of 1.0 and length of 5 pixels
    // These values can be changed to use a smaller or larger window
    array gauss_filt = gaussiankernel(5, 5, 1.0, 1.0);

    // Filter second-order derivatives with Gaussian kernel computed previously
    ixx = convolve(ixx, gauss_filt);
    ixy = convolve(ixy, gauss_filt);
    iyy = convolve(iyy, gauss_filt);

    // Calculate trace
    array tr = ixx + iyy;
    // Calculate determinant
    array det = ixx * iyy - ixy * ixy;

    // Calculate Harris response
    array response = det - 0.04f * (tr * tr);

    // Gets maximum response for each 3x3 neighborhood
    //array max_resp = maxfilt(response, 3, 3);
    array mask = constant(1,3,3);
    array max_resp = dilate(response, mask);

    // Discard responses that are not greater than threshold
    array corners = response > 1e5f;
    corners = corners * response;

    // Discard responses that are not equal to maximum neighborhood response,
    // scale them to original response value
    corners = (corners == max_resp) * corners;

    // Find corner indexes in the image as 1D indexes
    array idx = where(corners);

    // Calculate 2D corner indexes
    array corners_x = idx / corners.dims()[0];
    array corners_y = idx % corners.dims()[0];

    const int good_corners = corners_x.dims()[0];
    std::cout << "Corners found: " << good_corners << std::endl << std::endl;

    af_print(corners_x);
    af_print(corners_y);
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::deviceset(device);
        af::info();
        std::cout << "** ArrayFire Harris Corner Detector Demo **" << std::endl << std::endl;
        harris_demo(console);

    } catch (af::exception& ae) {
        std::cout << ae.what() << std::endl;
        throw;
    }

    if (!console) {
        printf("hit [enter]...");
        getchar();
    }
    return 0;
}
