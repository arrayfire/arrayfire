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
#include <vector>

using namespace af;
using namespace std;

static const bool is3D = false; const static int total_particles = 2500;
static const int reset = 5000;
static const int width = 768, height = 768, depth = 768;
static const float eps = 10.f;
static const int gravity_constant = 9000;

void initial_conditions_rand(vector<af::array> &pos, vector<af::array> &vels, vector<af::array> &forces) {
    for(int i=0; i<pos.size(); ++i) {
        pos[i]    = af::randn(total_particles) * width + width;
        vels[i]   = 0 * af::randu(total_particles) - 0.5;
        forces[i] = af::constant(0, total_particles);
    }
}

void simulate(vector<af::array> &pos, vector<af::array> &vels, vector<af::array> &forces, float dt) {
    for(int i=0; i<pos.size(); ++i) {
        pos[i] += vels[i] * dt;
        pos[i].eval();
    }

    //calculate forces to each particle
    vector<af::array> diff(pos.size());
    af::array dist = af::constant(0, pos[0].dims(0),pos[0].dims(0));

    for(int i=0; i<pos.size(); ++i) {
        diff[i] = tile(pos[i], 1, pos[i].dims(0)) - transpose(tile(pos[i], 1, pos[i].dims(0)));
        dist += (diff[i]*diff[i]);
    }

    dist = sqrt(dist);
    dist = af::max(20, dist);
    dist *= dist * dist;

    for(int i=0; i<pos.size(); ++i) {
        //calculate force vectors
        forces[i] = diff[i] / dist;
        forces[i].eval();

        //af::array idx = af::where(af::isNaN(forces[i]));
        //if(idx.elements() > 0)
        //    forces[i](idx) = 0;
        forces[i] = sum(forces[i]).T();

        //update force scaled to time, magnitude constant
        forces[i] *= (gravity_constant);
        forces[i].eval();

        //update velocities from forces
        vels[i] += forces[i] * dt;
        vels[i].eval();

        //noise
        //forces[i] += 0.1 * af::randn(forces[i].dims(0));

        //dampening
        //vels[i] *= 1 - (0.005*dt);
    }
}

void collisions(vector<af::array> &pos, vector<af::array> &vels, bool is3D) {
    //clamp particles inside screen border
    //af::array invalid_x = -2 * (pos[0] > width-1 || pos[0] < 0) + 1;
    //af::array invalid_y = -2 * (pos[1] > height-1 || pos[1] < 0) + 1;
    af::array invalid_x = (pos[0] < width-1 || pos[0] > 0);
    af::array invalid_y = (pos[1] < height-1 || pos[1] > 0);
    vels[0]= invalid_x * vels[0] ;
    vels[1]= invalid_y * vels[1] ;

    af::array projected_px = min(width-1, max(0, pos[0]));
    af::array projected_py = min(height - 1, max(0, pos[1]));
    pos[0] = projected_px;
    pos[1] = projected_py;

    if(is3D){
        af::array invalid_z = -2 * (pos[2] > depth-1 || pos[2] < 0) + 1;
        vels[2]= invalid_z * vels[2] ;
        af::array projected_pz = min(depth - 1, max(0, pos[2]));
        pos[2] = projected_pz;
    }
}


int main(int argc, char *argv[])
{
    try {

        af::info();

        af::Window myWindow(width, height, "Gravity Simulation using ArrayFire");
        myWindow.setColorMap(AF_COLORMAP_HEAT);

        int frame_count = 0;

        // Initialize the kernel array just once
        const af::array draw_kernel = gaussianKernel(5, 5);

        const int dims = (is3D)? 3 : 2;

        vector<af::array> pos(dims);
        vector<af::array> vels(dims);
        vector<af::array> forces(dims);

        // Generate a random starting state
        initial_conditions_rand(pos, vels, forces);

        af::array image = af::constant(0, width, height);
        af::array ids(total_particles, u32);

        af::timer timer = af::timer::start();
        while(!myWindow.close()) {
            float dt = af::timer::stop(timer);
            timer = af::timer::start();

            //if(is3D) {
                //array Pts = join(1, pos[0], pos[1], pos[2]);
                //myWindow.scatter3(Pts);
            //} else {
                ids = (pos[0].as(u32) * height) + pos[1].as(u32);
                image(ids) += 15.f;
                image = convolve(image, draw_kernel);
                myWindow.image(image);
                image = af::constant(0, image.dims());
                /*
                myWindow.scatter(pos[0], pos[1]);
                */
            //}

            frame_count++;

            // Generate a random starting state
            if(frame_count % reset == 0) {
                initial_conditions_rand(pos, vels, forces);
            }

            //simulate
            simulate(pos, vels, forces, dt);

            //check for collisions and adjust positions/velocities accordingly
            collisions(pos, vels, is3D);

        }
    } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}

