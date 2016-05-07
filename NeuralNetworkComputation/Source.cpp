#include "stdio.h"
#include "NN.h"
#include "GenAlg.h"
#include <vector>
#include "Source.h"
#include <math.h>
#include <cmath>
#include<string>
#include "CImg.h"
#include <conio.h>
#include <iomanip>
#include<iostream>
using namespace std;
using namespace cimg_library;



vector <double> imgToArray(CImg <unsigned> image)
{
	double val;
	double floatval;
	vector<double> matrix;
	for (int i = 0; i < 20; i++)
	for (int j = 0; j < 20; j++)
	{
		val = (254-(image(i, j, 0, 0)));
		floatval = val/ 255;
		if (floatval>0.1)
			matrix.push_back(floatval);
		else
			matrix.push_back(floatval);

	}
//	dispImgMat(matrix);
	return matrix;
}

bool comparatorFunc(Computer a, Computer b){
	if(a.GetFitness() > b.GetFitness())
		return true;
	return false;
}
void main()
{
	int v;
	double fSum = 0;
	int generationNum = 0;
	vector<double> imgArray;
	int imgChar = 0;
	double thisError;
	double thisResult;
	string hyph;
	hyph = "-0";

	vector<double>c1, c2;

	Computer *thisComputer;
	string thisAlph;
	vector <double> resultVals;
	vector<unsigned> topology;
	topology.push_back(400); topology.push_back(400);
//	topology.push_back(400);
	//topology.push_back(100);
	topology.push_back(26);
	int x = 36; int jaq;
	string dirfold;
	dirfold = "letters/img0";
	string zeroes;
	int userGenNum;
	bool GAbool;
	string GAboolstring;

	Computer *temp = new Computer(topology);
	Computer *holder = new Computer(topology);

	int num_computers;

	cout << "Enter Population size:";
	cin >> num_computers;

	//Generate population of num_computers
	for (int i = 0; i < num_computers; i++)
	{
		compPop.push_back(Computer(topology));
	}

		cout << "Please enter number of Generations: ";
		cin >> userGenNum;
		cout << "Do you want to use GA? Please type 'yes' for yes. 'no' for no\n";
		cin >> GAboolstring;
		if (GAboolstring == "yes")
			GAbool = true;
		else
			GAbool = false;
		//While loop keeps the generations going. This is the learning loop.
		while (generationNum<userGenNum)
		{
			fSum = 0;
			for (int i = 0; i < compPop.size(); i++)
			{
				compPop[i].fitness = 0;
					//for each letter
					for (int j = 1; j <= 26; j++)
					{
						jaq = x + j;
						thisAlph = IntToAlph(j);
						//for each pic
						for (int k = 1; k < 3; k++)
						{
							if (k < 10)
								zeroes = "0";
							else
								zeroes = "";
						//Getting the image and converting to array
						string tmp(dirfold+to_string(jaq)+hyph+zeroes+ to_string(k) + ".bmp");
						char *c = const_cast<char*>(tmp.c_str());
						CImg<unsigned char> image(c);
						imgArray = imgToArray(image);

						//Feed image array into neural network.
						compPop[i].feedforward(imgArray);

						//Obtain result.
						resultVals.clear();
						compPop[i].thisNetwork->getResults(resultVals);
						
						v = getMaxPos(resultVals);
				//		v = 0;
				//		for (int a = 0; a < 26; a++)
				//			v += resultVals[a];

						cout << "\nGeneration: " << generationNum << "\nComputer: " << i << " \nRealLetter: " <<thisAlph << " \nPicno: " << k << "\nFitness:" << compPop[i].fitness << "\nResultLetter: "<<IntToAlph(v+1)<<endl;
				//		for (int a = 0; a < resultVals.size(); a++)
				//			cout << resultVals[a] << endl;

						//obtain imgChar and backPropagate
						imgChar = j-1;
						compPop[i].BackPropagate(getTargetVals(imgChar));

						if (v == imgChar)
							compPop[i].fitness++;
					}
				}
				fSum += compPop[i].fitness;
			}
			//Rank computers based on error. Using bubble sort atm. will fix later.
			for (int j = 0; j <= compPop.size()-1; j++)
			{
				for (int k = 0; k < j; k++)
				{
					if (compPop[k].GetFitness()<compPop[k + 1].GetFitness())
					{
						temp = &compPop[k];
						compPop[k] = compPop[k + 1];
						compPop[k + 1] = *temp;
					}
				}
			}
		    //Use roulette wheel to obtain children.
			newCompPop.clear();
			for (int i = 0; i < compPop.size(); i++)
			{
				vector<double> parent1 = rouletteWheel(compPop, fSum);
				vector<double> parent2 = rouletteWheel(compPop, fSum);
				Children newChild = Crossover(parent1, parent2);
				vector<double>c1, c2;
				c1 = newChild.childOne;
				c2 = newChild.childTwo;
				if (newCompPop.size() < compPop.size())
				{
					holder->SetWeights(c1);
					newCompPop.push_back(*holder);
				}
				else
					break;
				if (newCompPop.size() < compPop.size())
				{
					holder->SetWeights(c2);
					newCompPop.push_back(*holder);
				}
				else
					break;
			}   
			for (int i = 0; i < compPop.size(); i++){
				if (GAbool)
					compPop[i].SetWeights(newCompPop[i].GetWeights());
				compPop[i].fitness = 0;
			}
			cout << "\nFitness Sum: " << fSum<<endl;
			generationNum++;
			fSum = 0; 
	}
	vector<double> testingWeights = compPop[0].getNetwork()->GetWeights();
	vector<double> thisImgArray;
	//Post Training
	/////////////////////
	cout << "Testing:" << endl;
	string addr("TestLetters/img0");
	string dotBmp("-001.bmp");
	string testAddr;
	for (int g = 37; g < 42; g++)
	{
		testAddr = addr + to_string(g) + dotBmp;
		char *tchar = const_cast<char*>(testAddr.c_str());
		CImg<unsigned char> nimage(tchar);
		thisImgArray = imgToArray(nimage);
		Computer testComputer(topology);
		testComputer.SetWeights(testingWeights);
		testComputer.feedforward(thisImgArray);
		resultVals.clear();
		testComputer.getNetwork()->getResults(resultVals);
		v = getMaxPos(resultVals);
		cout << IntToAlph(v + 1)<<endl;
	}
	getch();
}
//roulette wheel algorithm. I'll be assigning fitnesses depending on distance from character.
vector <double> rouletteWheel(vector<Computer> CompPop, double fitnessSum)
{
	int fInt = int(fitnessSum);
	int r;
	if (fInt == 0)
		r = rand() % CompPop.size() - 1;
	else
		r = rand() % fInt;
	int iteratedSum = 0;
	for (int i = 0; i < CompPop.size(); i++)
	{
		iteratedSum += CompPop[i].GetFitness();
		if (iteratedSum >= r)
			return CompPop[i].GetWeights();
	}
	return CompPop[r].GetWeights();
}
vector<double> getTargetVals(int n)
{
	vector<double> targetVals;
	for (int i = 0; i < 26; i++)
	{
		if (i != (n))
			targetVals.push_back(-1);
		else
			targetVals.push_back(1);
	}
	return targetVals;
}
string IntToAlph(int thisInt)
{
	switch (thisInt){
	case 1: return "a"; break;
	case 2: return "b"; break;
	case 3:return "c"; break;
	case 4:return "d"; break;
	case 5:return "e"; break;
	case 6: return "f"; break;
	case 7: return "g"; break;
	case 8: return "h"; break;
	case 9: return "i"; break;
	case 10: return "j"; break;
	case 11: return "k"; break;
	case 12: return "l"; break;
	case 13: return "m"; break;
	case 14: return "n"; break;
	case 15: return "o"; break;
	case 16: return "p"; break;
	case 17: return "q"; break;
	case 18: return "r"; break;
	case 19: return "s"; break;
	case 20: return "t"; break;
	case 21: return "u"; break;
	case 22: return "v"; break;
	case 23: return "w"; break;
	case 24: return "x"; break;
	case 25: return "y"; break;
	case 26: return "z"; break;
	};
}
int getMaxPos(vector<double> array)
{
	double max = 0; int maxpos = 0;
	for (int i = 0; i < array.size();i++)
		if (array[i]>max)
		{
			max = array[i];
			maxpos = i;
		}
		return maxpos;
}
void dispImgMat(vector <double>imgArray)
{
	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			if (imgArray[i] >= 100)
				cout << imgArray[i+20*j] << " ";
			else
				cout << imgArray[i+20*j] << "  ";
		}
		cout << endl;
	}
}