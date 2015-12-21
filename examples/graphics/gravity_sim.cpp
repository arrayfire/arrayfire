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
static const int pixels_per_unit = 20;

void simulate(af::array *pos, af::array *vels, af::array *forces, float dt){
    pos[0] += vels[0] * pixels_per_unit * dt;
    pos[1] += vels[1] * pixels_per_unit * dt;

    //calculate distance to center
    af::array diff_x = pos[0] - width/2;
    af::array diff_y = pos[1] - height/2;
    af::array dist = sqrt( diff_x*diff_x + diff_y*diff_y );

    //calculate normalised force vectors
    forces[0] = -1 * diff_x / dist;
    forces[1] = -1 * diff_y / dist;
    //update force scaled to time and magnitude constant
    forces[0] *= pixels_per_unit * dt;
    forces[1] *= pixels_per_unit * dt;

    //dampening
    vels[0] *= 1 - (0.005*dt);
    vels[1] *= 1 - (0.005*dt);

    //update velocities from forces
    vels[0] += forces[0];
    vels[1] += forces[1];

}

void collisions(af::array *pos, af::array *vels){
    //clamp particles inside screen border
    af::array projected_px = min(width, max(0, pos[0]));
    af::array projected_py = min(height - 1, max(0, pos[1]));

    //calculate distance to center
    af::array diff_x = projected_px - width/2;
    af::array diff_y = projected_py - height/2;
    af::array dist = sqrt( diff_x*diff_x + diff_y*diff_y );

    //collide with center sphere
    const int radius = 50;
    const float elastic_constant = 0.91f;
    if(sum<int>(dist<radius) > 0) {
        vels[0](dist<radius) = -elastic_constant * vels[0](dist<radius);
        vels[1](dist<radius) = -elastic_constant * vels[1](dist<radius);

        //normalize diff vector
        diff_x /= dist;
        diff_y /= dist;
        //place all particle colliding with sphere on surface
        pos[0](dist<radius) = width/2 + diff_x(dist<radius) * radius;
        pos[1](dist<radius) = height/2 +  diff_y(dist<radius) * radius;
    }
}


int main(int argc, char *argv[])
{
    try {
        const static int total_particles = 1000;
        static const int reset = 500;

        af::info();

        af::Window myWindow(width, height, "Gravity Simulation using ArrayFire");

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array draw_kernel = gaussianKernel(3, 3);

        af::array pos[2];
        af::array vels[2];
        af::array forces[2];

        // Generate a random starting state
        pos[0] = af::randu(total_particles) * width;
        pos[1] = af::randu(total_particles) * height;

        vels[0] = af::randn(total_particles);
        vels[1] = af::randn(total_particles);

        forces[0] = af::randn(total_particles);
        forces[1] = af::randn(total_particles);

        af::array image = af::constant(0, width, height);
        af::array ids(total_particles, u32);

        af::timer timer = af::timer::start();
        while(!myWindow.close()) {
            float dt = af::timer::stop(timer);
            timer = af::timer::start();

            ids = (pos[0].as(u32) * height) + pos[1].as(u32);
            image(ids) += 255;
            image = convolve2(image, draw_kernel);
            myWindow.image(image);
            image = af::constant(0, image.dims());
            frame_count++;

            // Generate a random starting state
            if(frame_count % reset == 0) {
                pos[0] = af::randu(total_particles) * width;
                pos[1] = af::randu(total_particles) * height;

                vels[0] = af::randn(total_particles);
                vels[1] = af::randn(total_particles);
            }

            //check for collisions and adjust positions/velocities accordingly
            collisions(pos, vels);

            //run force simulation and update particles
            simulate(pos, vels, forces, dt);

        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}

