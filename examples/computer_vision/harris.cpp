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
    af::Window wnd("Harris Corner Detector");

    // Load image
    array img_color;
    if (console)
        img_color = loadImage(ASSETS_DIR "/examples/images/square.png", true);
    else
        img_color = loadImage(ASSETS_DIR "/examples/images/lena.ppm", true);
    // Convert the image from RGB to gray-scale
    array img = colorSpace(img_color, AF_GRAY, AF_RGB);
    // For visualization in ArrayFire, color images must be in the [0.0f-1.0f] interval
    img_color /= 255.f;

    // Calculate image gradients
    array ix, iy;
    grad(ix, iy, img);

    // Compute second-order derivatives
    array ixx = ix * ix;
    array ixy = ix * iy;
    array iyy = iy * iy;

    // Compute a Gaussian kernel with standard deviation of 1.0 and length of 5 pixels
    // These values can be changed to use a smaller or larger window
    array gauss_filt = gaussianKernel(5, 5, 1.0, 1.0);

    // Filter second-order derivatives with Gaussian kernel computed previously
    ixx = convolve(ixx, gauss_filt);
    ixy = convolve(ixy, gauss_filt);
    iyy = convolve(iyy, gauss_filt);

    // Calculate trace
    array itr = ixx + iyy;
    // Calculate determinant
    array idet = ixx * iyy - ixy * ixy;

    // Calculate Harris response
    array response = idet - 0.04f * (itr * itr);

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

    // Gets host pointer to response data
    float* h_corners = corners.host<float>();

    unsigned good_corners = 0;

    // Draw draw_len x draw_len crosshairs where the corners are
    const int draw_len = 3;
    for (int y = draw_len; y < img_color.dims(0) - draw_len; y++) {
        for (int x = draw_len; x < img_color.dims(1) - draw_len; x++) {
            // Only draws crosshair if is a corner
            if (h_corners[x * corners.dims(0) + y] > 1e5f) {
                // Draw horizontal line of (draw_len * 2 + 1) pixels centered on the corner
                // Set only the first channel to 1 (green lines)
                img_color(y, seq(x-draw_len, x+draw_len), 0) = 0.f;
                img_color(y, seq(x-draw_len, x+draw_len), 1) = 1.f;
                img_color(y, seq(x-draw_len, x+draw_len), 2) = 0.f;

                // Draw vertical line of (draw_len * 2 + 1) pixels centered on  the corner
                // Set only the first channel to 1 (green lines)
                img_color(seq(y-draw_len, y+draw_len), x, 0) = 0.f;
                img_color(seq(y-draw_len, y+draw_len), x, 1) = 1.f;
                img_color(seq(y-draw_len, y+draw_len), x, 2) = 0.f;

                good_corners++;
            }
        }
    }

    printf("Corners found: %u\n", good_corners);

    if (!console) {
        // Previews color image with green crosshairs
        while(!wnd.close())
            wnd.image(img_color);
    } else {
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
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::setDevice(device);
        af::info();
        std::cout << "** ArrayFire Harris Corner Detector Demo **" << std::endl << std::endl;
        harris_demo(console);

    } catch (af::exception& ae) {
        std::cerr << ae.what() << std::endl;
        throw;
    }

    return 0;
}
