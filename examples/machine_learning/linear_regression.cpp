
/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// Warning : Not using the original dataset from  https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
// but the normalized : the x variables have been standardized to have mean 0 and squared length = 1 (sum(x^2)=1). 
// Each of these 10 feature variables have been mean centered and scaled by the standard deviation times n_samples (i.e. the sum of squares of each column totals 1).
// https://www4.stat.ncsu.edu/~boos/var.select/diabetes.read.rdata.out.txt
// scikit-learn:
// https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
// plot  "diabetes.tab.txt" using 1:11 with points

#include <arrayfire.h>
#include <math.h>

#include <stdio.h>
#include <af/util.h>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace af;

void parseFloatLine(std::ifstream& ifs, std::vector<float>& v) {
    v.clear();
    std::string l;
    getline(ifs, l);
    std::stringstream ss(l);
    float f;
    while(!ss.eof()) {
        ss >> f;
	v.push_back(f);
    }
}

// Demo of ordinary least squares Linear Regression
// fits a linear model with coefficients w = (w1, …, wp) 
// to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.
// The slope of the fitted line is equal to the correlation between y and x 
// corrected by the ratio of standard deviations of these variables. 
// The intercept of the fitted line is such that the line passes through the center of mass (x, y) of the data points. 
// Expected results:
// Reg coeffs: Slope:  938.23786125  Intercept:  152.91886182616167
// Mean squared error: 2548.07
// Coefficient of determination (r2): 0.47 (1 is perfect prediction)
int demo(bool console, const int perc) {
    std::ifstream ifs("../examples/machine_learning/diabetes.read.rdata.out.txt");
    if (!ifs.good()) {
        std::cerr << "Cannot open diabetes.tab.txt" << std::endl;
        return -1;
    }
    std::string line;
    std::getline(ifs, line); // just ignore the header line

    ifs.unsetf(std::ios_base::skipws);
    const unsigned N = std::count(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), '\n');
    std::cout << "Found " << N << " lines of data" << std::endl;
    if (N < 1)
        return -1;

    ifs.clear();
    ifs.seekg(0, std::ios::beg); // rewind
    std::getline(ifs, line); // just ignore the header line

    std::vector<float> trainX, testX; // patients ages
    std::vector<float> trainY, testY; // diabete disease progression: "quantitative measure of disease progression one year after baseline"
    std::vector<float> row;
    unsigned n = 0;
    float frac = (float)(perc) / 100.0; // fraction of N to be used for training vs testing
    while (!ifs.eof()) {
	parseFloatLine(ifs, row);
	if (row.size() < 11)
            break;
	++n;
        // column 3 is the bmi, col 11 is Y (diabete prog)  
        if (n < frac*N) {
	    trainX.push_back(row[3]);
	    trainY.push_back(row[11]);
        } else {
	    testX.push_back(row[3]);
	    testY.push_back(row[11]);
        }
    }

    if (trainY.size() + testY.size() != N) {
        std::cerr << "Unexpected number of datapoints: " << trainY.size() << " + " << testY.size()  << " vs " << N << std::endl;
        return -1;
    }
    std::cout << N << " data points extracted : " << trainX.size() << " for training, " << testX.size() << " for testing.\n";
    std::cout << "First point: " << trainX[0] << ":" << trainY[0] << "...\n";

    af::array trainXarray(trainX.size(), trainX.data());
    af::array trainYarray(trainY.size(), trainY.data());
    //af_print(trainYarray);
    af::array testXarray(testX.size(), testX.data());
    af::array testYarray(testY.size(), testY.data());


    float meanX = af::mean<float>(trainXarray);
    float meanY = af::mean<float>(trainYarray);
    float meanX2 = af::mean<float>(trainXarray*trainXarray);
    float meanY2 = af::mean<float>(trainYarray*trainYarray);
    float meanXY = af::mean<float>(trainXarray*trainYarray);
    std::cout << "meanX=" << meanX << " meanY=" << meanY << " meanXY=" << meanXY << std::endl;

    // https://en.wikipedia.org/wiki/Simple_linear_regression
    float rxy = 0; // sample correlation coefficient 
    rxy = (meanXY - meanX*meanY) / sqrt((meanX2-meanX*meanX)*(meanY2-meanY*meanY));
    std::cout << "rxy = " << rxy << std::endl;

    // https://www.geeksforgeeks.org/linear-regression-python-implementation/
    // slope = B1 = SSxy / SSxx
    // SSxy = Sigma((X-meanX)(Y-meanY))
    float SSxy = af::sum<float>((trainXarray-meanX)*(trainYarray-meanY));
    // SSxx = Sigma((X-meanX)^2)
    float SSxx = af::sum<float>(af::pow(trainXarray-meanX, 2)); // dont use af::pow2
    float slope = SSxy / SSxx;
    std::cout << "Slope SSxy/SSxx = " << slope << std::endl;
    // intercept = B0 = meanY - B1*meanX
    float intercept = meanY - slope*meanX;
    std::cout << "Intercept = " << intercept << std::endl;

    // R2 = 1 - SSE / SST 
    // with SSE = "sum of squared errors" = Sigma((Y-fittedY)^2) 
    // with SST = "sum of squared total" = Sigma((Y-meanY)^2)
    
    float r2 = 0; 
    std::cout << " r2=" << r2 << std::endl;
    return 0;
}

int main(int argc, char **argv) {
    int device   = argc > 1 ? atoi(argv[1]) : 0;
    bool console = argc > 2 ? argv[2][0] == '-' : false;
    int perc     = argc > 3 ? atoi(argv[3]) : 95; // percentage training/test
    std::cout << argv[0] << std::endl;

    try {
        af::setDevice(device);
        af::info();
        return demo(console, perc);
    } catch (af::exception &ae) { 
        std::cerr << ae.what() << std::endl; 
    }

    return 0;
}

