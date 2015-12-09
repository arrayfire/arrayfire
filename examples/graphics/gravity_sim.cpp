/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <iostream>
#include <cstdio>

using namespace af;
using namespace std;

static const int width = 512, height = 512;


void simulate(af::array &parts, af::array &vels, af::array &forces){
    parts += vels;

    //calculate distance to center
    float center_coors[2]   = { width / 2, height / 2 };
    af::array col  = tile(af::array(1, 2, center_coors), parts.dims(0));
    af::array diff = parts - col;
    af::array dist = sqrt( diff.col(0)*diff.col(0) + diff.col(1)*diff.col(1) );

    forces = -1 * diff;
    forces.col(0) /= dist; //normalize force vectors
    forces.col(1) /= dist; //normalize force vectors

    //update velocities from forces
    vels += forces;

}

void collisions(af::array &parts, af::array &vels){
    //clamp particles inside screen border
    parts.col(0) = min(width, max(0, parts.col(0)));
    parts.col(1) = min(height - 1, max(0, parts.col(1)));

    //calculate distance to center
    float center_coors[2]   = { width / 2, height / 2 };
    af::array col  = tile(af::array(1, 2, center_coors), parts.dims(0));
    af::array diff = parts - col;
    af::array dist = sqrt( diff.col(0)*diff.col(0) + diff.col(1)*diff.col(1) );

    /*
    //collide with center sphere
    int radius = 50;
    af::array col_ids = dist(dist<radius);
    if(col_ids.dims(0) > 0) {
        //vels(col_ids, span) += -1 * parts(col_ids, span);
        vels(col_ids, span) = 0;
    }
    */

}

int main(int argc, char *argv[])
{
    try {
        const static int total_particles=200;
        static const int reset = 500;

        af::info();

        af::Window myWindow(width, height, "Gravity Simulation using ArrayFire");

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array draw_kernel = gaussianKernel(3, 3);

        // Generate a random starting state
        af::array particles = af::randu(total_particles,2);
        particles.col(0) *= width;
        particles.col(1) *= height;

        af::array velocities = af::randn(total_particles, 2);
        af::array forces = af::randn(total_particles, 2);

        af::array image = af::constant(0, width, height);
        af::array ids(total_particles, u32);

        while(!myWindow.close()) {

            ids = (particles.col(0).as(u32) * height) + particles.col(1).as(u32);
            image(ids) += 255;
            image = convolve2(image, draw_kernel);
            myWindow.image(image);
            image(span, span) = 0;
            frame_count++;

            // Generate a random starting state
            if(frame_count % reset == 0) {
                particles = af::randu(total_particles,2);
                particles.col(0) *= width;
                particles.col(1) *= height;

                velocities = af::randn(total_particles, 2);
            }

            //run force simulation and update particles
            simulate(particles, velocities, forces);

            //check for collisions and adjust velocities accordingly
            collisions(particles, velocities);

        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    #ifdef WIN32 // pause in Windows
    if (!(argc == 2 && argv[1][0] == '-')) {
        printf("hit [enter]...");
        fflush(stdout);
        getchar();
    }
    #endif
    return 0;
}

