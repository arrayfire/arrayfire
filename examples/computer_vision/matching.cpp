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
    // after input image is indexed, .copy() is required because
    // displaying the indexed array doesn't seem to render correctly
    array tmp_img = img(seq(100, 100+patch_size, 1.0), seq(100, 100+patch_size, 1.0)).copy();
    array result  = matchTemplate(img, tmp_img); // Default disparity metric is
                                                 // Sum of Absolute differences (SAD)
                                                 // Currently supported metrics are
                                                 // AF_SAD, AF_ZSAD, AF_LSAD, AF_SSD,
                                                 // AF_ZSSD, ASF_LSSD
    array disp_img = img/255.0f;
    array disp_tmp = tmp_img/255.0f;
    array disp_res = normalize(result);

    unsigned minLoc = where(disp_res==0.0f).scalar<unsigned>();
    std::cout<< "Location(linear index) of minimum disparity value = " <<
        minLoc << std::endl;

    if (!console) {
        // Draw a rectangle on input image where the template matches
        array marked_res = tile(disp_img, 1, 1, 3);

        unsigned x = minLoc%iDims[0];
        unsigned y = minLoc/iDims[0];
        printf("\nMatching patch origin = (%u, %u)\n\n", x, y);
        seq col_span(x, x+patch_size, 1);
        seq row_span(y, y+patch_size, 1);
        //edge on left
        marked_res(col_span , y        , 0) = 0.f;
        marked_res(col_span , y        , 1) = 0.f;
        marked_res(col_span , y        , 2) = 1.f;
        //edge on right
        marked_res(col_span , y+100    , 0) = 0.f;
        marked_res(col_span , y+100    , 1) = 0.f;
        marked_res(col_span , y+100    , 2) = 1.f;
        //edge on top
        marked_res(x        , row_span , 0) = 0.f;
        marked_res(x        , row_span , 1) = 0.f;
        marked_res(x        , row_span , 2) = 1.f;
        //edge on bottom
        marked_res(x+100    , row_span , 0) = 0.f;
        marked_res(x+100    , row_span , 1) = 0.f;
        marked_res(x+100    , row_span , 2) = 1.f;

        std::cout<<"Note: Based on the disparity metric option provided to matchTemplate function\n"
            "either minimum or maximum disparity location is the starting corner\n"
            "of our best matching path to template image in the search image"<< std::endl;

        af::Window wnd("Template Matching Demo");

        // Previews color image with green crosshairs
        while(!wnd.close()) {
            wnd.grid(2, 2);
            wnd(0, 0).image(disp_img  , "Search Image"    );
            wnd(0, 1).image(disp_tmp  , "Template Patch"  );
            wnd(1, 0).image(disp_res  , "Disparity values");
            wnd(1, 1).image(marked_res, "Best Match"      );
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
