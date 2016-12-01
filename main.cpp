//
//  main.cpp
//  Backpropagation_rewrite
//
//  Created by 徐鹏 on 11/2/16.
//  Copyright © 2016 徐鹏. All rights reserved.
//
#include <iostream>
#include <fstream>
#include "NeuralNetwork.hpp"
using namespace std;

int main()
{
    double weight_0[4] = { 0.15, 0.2, 0.25, 0.3 };
    double weight_1[4] = { 0.4, 0.45, 0.5, 0.55 };
    double bias[2] = { 0.35, 0.6 };
    double traininput[2]  = {  0.05, 0.1 };
    double traintarget[2] = {  0.01, 0.99};
    NeuralNetwork workNet(3, 2, 2, 2, weight_0, weight_1, traininput, traintarget, bias);
    int error_flag = 1;
    double error = 1.;
    int train_num = 0;
    while (error_flag)
    {
        error = workNet.calculate_total_error();
        workNet.train();
        train_num++;
        cout << "train_num : = " << train_num << "----"<<
        "train_error : " << error << endl;
        if (error < 0.000001) break;
    }
	vector<double>inputs;
	inputs.push_back(0.05);
	inputs.push_back(0.1);
	vector<double>result = workNet.predict(inputs);
	cout << result[0] <<"----"<<result[1]<< endl;
    return 0;
}
