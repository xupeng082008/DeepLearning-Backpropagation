#pragma once
#include "Neuron.h"
class Layer
{
public:
	Layer();
	Layer(int neurons_num);
	~Layer();

	Neuron *neurons_;
	int neurons_num_;
private:
	
	
};

