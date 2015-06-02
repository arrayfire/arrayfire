/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <af/util.h>
#include <math.h>
#include "mnist_common.h"

using namespace af;
using std::vector;

float accuracy(const array& predicted, const array& target)
{
    array val, plabels, tlabels;
    max(val, tlabels, target, 1);
    max(val, plabels, predicted, 1);
    return 100 * count<float>(plabels == tlabels) / tlabels.elements();
}

// Activation function
array sigmoid(const array &val)
{
    return 1 / (1 + exp(-val));
}

// Derivative of the activation function
array deriv(const array &out)
{
    return out * (1 - out);
}

// Cost function
double error(const array &out,
             const array &pred)
{
    array dif = (out - pred);
    return sqrt((double)(sum<float>(dif * dif)));
}

class ann {

private:
    int num_layers;
    vector<array> weights;

    // Add bias input to the output from previous layer
    array add_bias(const array &in);

    vector<array> forward_propagate(const array& input);

    void back_propagate(const vector<array> signal,
                        const array &pred,
                        const double &alpha);
public:

    // Create a network with given parameters
    ann(vector<int> layers, double range=0.05);

    // Output after single pass of forward propagation
    array predict(const array &input);

    // Method to trian the neural net
    double train(const array &input, const array &target,
                 double alpha = 1.0,
                 int max_epochs = 300,
                 int batch_size = 100,
                 double maxerr = 1.0,
                 bool verbose = false);
};

array ann::add_bias(const array &in)
{
    // Bias input is added on top of given input
    return join(1, constant(1, in.dims(0), 1), in);
}

vector<array> ann::forward_propagate(const array& input)
{
    // Get activations at each layer
    vector<array> signal(num_layers);
    signal[0] = input;

    for (int i = 0; i < num_layers - 1; i++) {
        array in = add_bias(signal[i]);
        array out = matmul(in, weights[i]);
        signal[i + 1] = sigmoid(out);
    }

    return signal;
}

void ann::back_propagate(const vector<array> signal,
                         const array &target,
                         const double &alpha)
{

    // Get error for output layer
    array out = signal[num_layers  - 1];
    array err = (out - target);
    int m = target.dims(0);

    for (int i = num_layers - 2; i >= 0; i--) {
        array in = add_bias(signal[i]);
        array delta = (deriv(out) * err).T();

        // Adjust weights
        array grad = -(alpha * matmul(delta, in)) / m;
        weights[i] += grad.T();

        // Input to current layer is output of previous
        out = signal[i];
        err = matmulTT(delta, weights[i]);

        // Remove the error of bias and propagate backward
        err = err(span, seq(1, out.dims(1)));
    }
}

ann::ann(vector<int> layers, double range) :
    num_layers(layers.size()),
    weights(layers.size() - 1)
{
    // Generate uniformly distributed random numbers between [-range/2,range/2]
    for (int i = 0; i < num_layers - 1; i++) {
        weights[i] = range * randu(layers[i] + 1, layers[i + 1]) - range/2;
    }
}

array ann::predict(const array &input)
{
    vector<array> signal = forward_propagate(input);
    array out = signal[num_layers - 1];
    return out;
}

double ann::train(const array &input, const array &target,
                  double alpha, int max_epochs, int batch_size,
                  double maxerr, bool verbose)
{

    const int num_samples = input.dims(0);
    const int num_batches = num_samples / batch_size;

    double err = 0;

    // Training the entire network
    for (int i = 0; i < max_epochs; i++) {

        for (int j = 0; j < num_batches - 1; j++) {

            int st = j * batch_size;
            int en = st + batch_size;

            array x = input(seq(st, en), span);
            array y = target(seq(st, en), span);

            // Propagate the inputs forward
            vector<array> signals = forward_propagate(x);
            array out = signals[num_layers - 1];


            // Propagate the error backward
            back_propagate(signals, y, alpha);
        }

        // Validate with last batch
        int st = (num_batches - 1) * batch_size;
        int en = num_samples - 1;
        array out = predict(input(seq(st, en), span));
        err = error(out, target(seq(st, en), span));

        // Check if convergence criteria has been met
        if (err < maxerr) {
            printf("Converged on Epoch: %4d\n", i + 1);
            return err;
        }

        if (verbose) {
            if ((i + 1) % 10 == 0) printf("Epoch: %4d, Error: %0.4f\n", i+1, err);
        }
    }
    return err;
}

int ann_demo(bool console, int perc)
{
    printf("** ArrayFire ANN Demo **\n\n");

    array train_images, test_images;
    array train_target, test_target;
    int num_classes, num_train, num_test;

    // Load mnist data
    float frac = (float)(perc) / 100.0;
    setup_mnist<true>(&num_classes, &num_train, &num_test,
                      train_images, test_images, train_target, test_target, frac);

    int feature_size = train_images.elements() / num_train;

    // Reshape images into feature vectors
    array train_feats = moddims(train_images, feature_size, num_train).T();
    array test_feats  = moddims(test_images , feature_size, num_test ).T();

    train_target = train_target.T();
    test_target  = test_target.T();

    // Network parameters
    vector<int> layers;
    layers.push_back(train_feats.dims(1));
    layers.push_back(100);
    layers.push_back(50);
    layers.push_back(num_classes);

    // Create network
    ann network(layers);

    // Train network
    timer::start();
    network.train(train_feats, train_target,
                  2.0, // learning rate / alpha
                  250, // max epochs
                  100, // batch size
                  0.5, // max error
                  true); // verbose
    af::sync();
    double train_time = timer::stop();

    // Run the trained network and test accuracy.
    array train_output = network.predict(train_feats);
    array test_output  = network.predict(test_feats );


    // Benchmark prediction
    af::sync();
    timer::start();
    for (int i = 0; i < 100; i++) {
        network.predict(test_feats);
    }
    af::sync();
    double test_time = timer::stop() / 100;

    printf("\nTraining set:\n");
    printf("Accuracy on training data: %2.2f\n",
           accuracy(train_output, train_target));

    printf("\nTest set:\n");
    printf("Accuracy on testing  data: %2.2f\n",
           accuracy(test_output , test_target ));

    printf("\nTraining time: %4.4lf s\n", train_time);
    printf("Prediction time: %4.4lf s\n\n", test_time);

    if (!console) {
        // Get 20 random test images.
        test_output = test_output.T();
        display_results<true>(test_images, test_output, test_target.T(), 20);
    }

    return 0;
}

int main(int argc, char** argv)
{
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 60;

    try {

        af::setDevice(device);
        af::info();
        return ann_demo(console, perc);

    } catch (af::exception &ae) {
        std::cerr << ae.what() << std::endl;
    }

}
