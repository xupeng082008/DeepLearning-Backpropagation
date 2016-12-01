//
//  NeuralNetwork.cpp
//  DeepLearning
//
//  Created by 徐鹏 on 11/2/16.
//  Copyright © 2016 徐鹏. All rights reserved.
//

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork()
{
}
NeuralNetwork::~NeuralNetwork()
{
}
NeuralNetwork::NeuralNetwork(int layers_num, int inputlayer_num, int hiddenlayer_num,
                             int outputlayer_num,
                             double* hidden_layer_weights, double* output_layer_weights,
                             double* net_inputs, double*  target_outs,double* bias) :
                                layers_num_(layers_num),
                                inputlayer_neurons_num_(inputlayer_num),
                                hiddenlayer_neurons_num_(hiddenlayer_num),
                                outputlayer_neurons_num_(outputlayer_num)
{
    layers_[0].neurons_num_ = inputlayer_neurons_num_;
    for (int i = 0; i < layers_[0].neurons_num_; i++)
    {
        layers_[0].neurons_[i].set_neuron_value(net_inputs[i]);
    }
    layers_[1].neurons_num_ = hiddenlayer_neurons_num_;
    layers_[2].neurons_num_ = outputlayer_neurons_num_;
    for (size_t i = 0; i < 2; i++)
    {
        bias_[i] = bias[i];
        net_inputs_[i] = net_inputs[i];
        target_outs_[i] = target_outs[i];
    }
    net_inputs_[inputlayer_neurons_num_] = 1.;//bias
    for (int i = 0; i < 4; i++)
    {
        hidden_layer_weights_[i] = hidden_layer_weights[i];
        output_layer_weights_[i] = output_layer_weights[i];
    }
    hidden_layer_weights_[4] = bias_[0];
    output_layer_weights_[4] = bias_[1];
    
    //assgin weights for every hiddenlayer neuron
    for (int i = 0,k = 0; i < layers_[1].neurons_num_; i++)
    {
//        layers_[1].neurons_[i].neuron_inputs_[i] = net_inputs_[i];
        for (int j = 0; j < layers_[0].neurons_num_; j++)
            layers_[1].neurons_[i].neuron_weights_[j] = hidden_layer_weights_[k++];
        layers_[1].neurons_[i].neuron_weights_[2] = bias_[0];
    }

    for (int i = 0, k = 0; i < layers_[2].neurons_num_; i++)
    {
        for (int j = 0; j < layers_[1].neurons_num_; j++)
            layers_[2].neurons_[i].neuron_weights_[j] = output_layer_weights_[k++];
        layers_[2].neurons_[i].neuron_weights_[2] = bias_[1];
    }
}
int NeuralNetwork::train()
{
    forward_pass();
    back_propagate();//calculate deltas; update weights
    return 0;
}

void NeuralNetwork::forward_pass()
{
    //set hidden inputs
    for (int i=0; i<layers_[1].neurons_num_; i++) {
        for (int j=0; j<layers_[0].neurons_num_+1; j++) {
            layers_[1].neurons_[i].neuron_inputs_[j]=net_inputs_[j];
        }
    }
    //calculate hidden_layer_output
    for (int i = 0; i < layers_[1].neurons_num_; i++)
    {
        layers_[1].neurons_[i].neuron_net_input_ =
        layers_[1].neurons_[i].calculate_total_net_input(
                    layers_[1].neurons_[i].neuron_weights_,
                    layers_[1].neurons_[i].neuron_inputs_,
                    layers_[0].neurons_num_);
        layers_[1].neurons_[i].neuron_net_output_ =
        layers_[1].neurons_[i].activite_fun(layers_[1].neurons_[i].neuron_net_input_);
    }
    
    //hiddenlayer_outoput used as  outputlayer's input
    for (int i = 0; i < layers_[2].neurons_num_; i++)
    {
        for (int j = 0; j < layers_[1].neurons_num_; j++)
        {
            layers_[2].neurons_[i].neuron_inputs_[j] = 
				layers_[1].neurons_[j].neuron_net_output_;
        }
        layers_[2].neurons_[i].neuron_inputs_[layers_[1].neurons_num_] = 1.;
    }
    //calculate output_layer's output
    for (int i = 0; i < layers_[2].neurons_num_; i++)
    {
        layers_[2].neurons_[i].neuron_net_input_ =
        layers_[2].neurons_[i].calculate_total_net_input(
                            layers_[2].neurons_[i].neuron_weights_,
                            layers_[2].neurons_[i].neuron_inputs_,
                            layers_[1].neurons_num_);
        layers_[2].neurons_[i].neuron_net_output_=
        layers_[2].neurons_[i].activite_fun(layers_[2].neurons_[i].neuron_net_input_);
    }
}

void NeuralNetwork::back_propagate()
{
    // Output_layer's neruon deltas :
    for (size_t i = 0; i < layers_[2].neurons_num_; i++)
    {
        layers_[2].neurons_[i].deltas_ =
        -(target_outs_[i] - layers_[2].neurons_[i].neuron_net_output_)
        * layers_[2].neurons_[i].calculate_activite_fun_diff(
                        layers_[2].neurons_[i].neuron_net_output_);
    }
    //Hiddenn_layer's neruon deltas :
    for (size_t i = 0; i < layers_[1].neurons_num_; i++)
    {
        layers_[1].neurons_[i].deltas_ = 0.;
        for (size_t j = 0; j < layers_[2].neurons_num_; j++)
        {
            //deltas[l-1] = output[l-1]*deltas[l]
            layers_[1].neurons_[i].deltas_ +=
            layers_[2].neurons_[j].neuron_weights_[i]
            * layers_[2].neurons_[j].deltas_;
        }
        layers_[1].neurons_[i].deltas_ *= 
			layers_[1].neurons_[i].calculate_activite_fun_diff
								(layers_[1].neurons_[i].neuron_net_output_);
    }
    
    //update weights
    //update output_layer's weights
    for (size_t i = 0; i < layers_[2].neurons_num_; i++)
    {
        //update w
        for (size_t j = 0; j < layers_[2].neurons_num_; j++)
        {
            layers_[2].neurons_[i].neuron_weights_[j] -=learning_rate
            *layers_[1].neurons_[j].neuron_net_output_
            *layers_[2].neurons_[i].deltas_;
        }
        //update bias
        layers_[2].neurons_[i].neuron_weights_[2] -= learning_rate*layers_[2].neurons_[i].deltas_;
        
    }
    //update hidden_layer's weights
    for (size_t i = 0; i < layers_[1].neurons_num_; i++)
    {
        //update w
        for (size_t j = 0; j < layers_[0].neurons_num_; j++)
        {
            //loss_diff_of_w[l] = output[l-1]*deltas[l]
            layers_[1].neurons_[i].neuron_weights_[j] -= learning_rate
            *layers_[0].neurons_[j].neuron_net_input_
            *layers_[1].neurons_[i].deltas_;
        }
        //update bias
        layers_[1].neurons_[i].neuron_weights_[2] -= learning_rate*layers_[1].neurons_[i].deltas_;
    }
}

double NeuralNetwork::calculate_total_error()
{
    total_error_ = 0.;
    forward_pass();
    for (size_t i = 0; i < layers_[2].neurons_num_; i++)
    {
        total_error_ += 0.5*(target_outs_[i] - layers_[2].neurons_[i].neuron_net_output_)*
        (target_outs_[i] - layers_[2].neurons_[i].neuron_net_output_);
    }
    return total_error_;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>innputs)
{
	std::vector<double>outputs;
	for (size_t i = 0; i < layers_[2].neurons_num_; i++)
	{
		outputs.push_back(layers_[2].neurons_[i].neuron_net_output_);
	}
	return outputs;
}

