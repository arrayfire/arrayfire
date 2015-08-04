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

array normalize(array a)
{
    float mx = af::max<float>(a);
    float mn = af::min<float>(a);
    return (a-mn)/(mx-mn);
}

void drawRectangle(array &out, unsigned x, unsigned y, unsigned dim0, unsigned dim1)
{
    printf("\nMatching patch origin = (%u, %u)\n\n", x, y);
    seq col_span(x, x+dim0, 1);
    seq row_span(y, y+dim1, 1);
    //edge on left
    out(col_span, y       , 0) = 0.f;
    out(col_span, y       , 1) = 0.f;
    out(col_span, y       , 2) = 1.f;
    //edge on right
    out(col_span, y+dim1  , 0) = 0.f;
    out(col_span, y+dim1  , 1) = 0.f;
    out(col_span, y+dim1  , 2) = 1.f;
    //edge on top
    out(x       , row_span, 0) = 0.f;
    out(x       , row_span, 1) = 0.f;
    out(x       , row_span, 2) = 1.f;
    //edge on bottom
    out(x+dim0  , row_span, 0) = 0.f;
    out(x+dim0  , row_span, 1) = 0.f;
    out(x+dim0  , row_span, 2) = 1.f;
}

static void templateMatchingDemo(bool console)
{
    // Load image
    array img_color;
    if (console)
        img_color = loadImage(ASSETS_DIR "/examples/images/square.png", true);
    else
        img_color = loadImage(ASSETS_DIR "/examples/images/man.jpg", true);

    // Convert the image from RGB to gray-scale
    array img = colorSpace(img_color, AF_GRAY, AF_RGB);
    dim4 iDims = img.dims();
    std::cout<<"Input image dimensions: " << iDims << std::endl << std::endl;
    // For visualization in ArrayFire, color images must be in the [0.0f-1.0f] interval

    // extract a patch from input image
    unsigned patch_size = 100;
    array tmp_img = img(seq(100, 100+patch_size, 1.0), seq(100, 100+patch_size, 1.0));
    array result  = matchTemplate(img, tmp_img); // Default disparity metric is
                                                 // Sum of Absolute differences (SAD)
                                                 // Currently supported metrics are
                                                 // AF_SAD, AF_ZSAD, AF_LSAD, AF_SSD,
                                                 // AF_ZSSD, ASF_LSSD
    array disp_img = img/255.0f;
    array disp_tmp = tmp_img/255.0f;
    array disp_res = normalize(result);

    unsigned minLoc;
    float    minVal;
    min<float>(&minVal, &minLoc, disp_res);
    std::cout<< "Location(linear index) of minimum disparity value = " << minLoc << std::endl;

    if (!console) {
        // Draw a rectangle on input image where the template matches
        array marked_res = tile(disp_img, 1, 1, 3);
        drawRectangle(marked_res, minLoc%iDims[0], minLoc/iDims[0], patch_size, patch_size);

        std::cout<<"Note: Based on the disparity metric option provided to matchTemplate function\n"
            "either minimum or maximum disparity location is the starting corner\n"
            "of our best matching patch to template image in the search image"<< std::endl;

        af::Window wnd("Template Matching Demo");

        // Previews color image with green crosshairs
        while(!wnd.close()) {
            wnd.setColorMap(AF_COLORMAP_DEFAULT);
            wnd.grid(2, 2);
            wnd(0, 0).image(disp_img  , "Search Image"    );
            wnd(0, 1).image(disp_tmp  , "Template Patch"  );
            wnd(1, 0).image(marked_res, "Best Match"      );
            wnd.setColorMap(AF_COLORMAP_HEAT);
            wnd(1, 1).image(disp_res  , "Disparity values");
            wnd.show();
        }
    }
}

int main(int argc, char** argv)
{
    int device = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;

    try {
        af::setDevice(device);
        af::info();
        std::cout << "** ArrayFire template matching Demo **" << std::endl << std::endl;
        templateMatchingDemo(console);

    } catch (af::exception& ae) {
        std::cerr << ae.what() << std::endl;
        throw;
    }

    return 0;
}
