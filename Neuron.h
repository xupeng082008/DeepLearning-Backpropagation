#pragma once
#include <stdlib.h>
class Neuron
{
public:
	Neuron();
	~Neuron();
	Neuron(double input);
	Neuron(int neurons_input_num);
	
	double calculate_total_net_input(double* layer_weights, double* neuron_inputs, int neuron_input_num);
	double calculate_output(double net_inputs);
	double activite_fun(double net_input);
	double calculate_activite_fun_diff(double fun_value);
	double get_neuron_value(double input);
	void set_neuron_value(double input);

	double neuron_net_input_;
	double neuron_net_output_;

//private:
	int neuron_input_num_;
	double* neuron_inputs_;
	double* neuron_weights_;
	double deltas_;

};

