#pragma once
#include <vector>
#include <iostream>
#include "Layer.h"
#include "Neuron.h"
class NeuralNetwork
{
public:
	NeuralNetwork();
	~NeuralNetwork();
	//初始化网络的层数，权值系数，输入输出
	NeuralNetwork(int layers_num, int inputlayer_neurons_num, int hiddenlayer_neurons_num, int outputlayer_neurons_num,
				  double* hidden_layer_weights, double* output_layer_weights,
				  double*  net_inputs, double* target_outs,double* bias);
	int train();
	void forward_pass();
	void back_propagate();
	double calculate_total_error();
	double learning_rate = 0.5;

private:
	int inputlayer_neurons_num_, hiddenlayer_neurons_num_, outputlayer_neurons_num_, layers_num_;
	Layer** layers_;

	double * net_inputs_;
	double * target_outs_;
	
	double* hidden_layer_weights_;
	double* output_layer_weights_;
	double* bias_;

	double total_error_;
};

