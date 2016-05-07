#include "NN.h"
#include "GenAlg.h"
#include <math.h>
#include <iostream>

string IntToAlph(int thisInt);
vector<double> getTargetVals(int n);
int getMaxPos(vector<double> array);
void dispImgMat(vector <double>imgArray);
class Computer
{
public:
	double fitness;
	Network* thisNetwork;
	vector<unsigned> topology;
	vector<double>weights;
	public:

	Computer(vector<unsigned> topology)
	{
		setNetwork(topology);
	}
	void setNetwork(vector<unsigned> top)
	{
		thisNetwork = new Network(top);
	}
	void BackPropagate(vector<double> targetVals)
	{
		thisNetwork->backPropagate(targetVals);
	}
	Network* getNetwork()
	{
		return thisNetwork;
	}
	double GetFitness()
	{
		return fitness;
	}
	vector<double> GetWeights()
	{
		weights = thisNetwork->GetWeights();
		return weights;
	}
	void feedforward(vector<double> inputs)
	{
		thisNetwork->feedForward(inputs);
	}
	vector<double> GetResult()
	{
		vector<double> resultVals;
		thisNetwork->getResults(resultVals);
		return resultVals;
	}
	void SetFitness(double n)
	{
		fitness = n;
	}
	void SetWeights(vector<double> weights)
	{
		thisNetwork->PutWeights(weights);
	}
};

vector<Computer> compPop;
vector<Computer> newCompPop;
vector<double> rouletteWheel(vector<Computer> CompPop, double fitnessSum);