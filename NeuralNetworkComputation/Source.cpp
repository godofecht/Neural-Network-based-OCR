#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

// tinyML includes
#include "Network.h"

// CImg for image processing
#include "CImg.h"

using namespace std;
using namespace cimg_library;
using namespace ML;

// Image preprocessing: convert 20x20 image to normalized vector
vector<double> imgToArray(const CImg<unsigned>& image) {
    vector<double> matrix;
    matrix.reserve(400);
    
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            double val = (254.0 - image(i, j, 0, 0)) / 255.0;
            matrix.push_back(val);
        }
    }
    return matrix;
}

// Genetic Algorithm: Individual (Computer)
struct Computer {
    unique_ptr<Network> net;
    double fitness = 0.0;
    
    Computer(const vector<unsigned>& topology) 
        : net(make_unique<Network>(topology)) {}
    
    Computer(const Computer& other)
        : net(make_unique<Network>(*other.net)), fitness(other.fitness) {}
    
    Computer(Computer&& other) = default;
    Computer& operator=(Computer&& other) = default;

    Computer& operator=(const Computer& other) {
        if (this != &other) {
            net = make_unique<Network>(*other.net);
            fitness = other.fitness;
        }
        return *this;
    }
    
    double getFitness() const { return fitness; }
    void setFitness(double f) { fitness = f; }
};

// GA Selection: compare by fitness
bool comparatorFunc(const Computer& a, const Computer& b) {
    return a.getFitness() > b.getFitness();
}

int main() {
    try {
        int populationSize = 0;
        int generations = 0;
        string useGAString;
        
        cout << "=== Neural Network OCR Trainer (Powered by tinyML) ===" << endl << endl;
        
        cout << "Enter Population size: ";
        cin >> populationSize;
        
        if (populationSize <= 0) {
            cerr << "Population size must be positive!" << endl;
            return 1;
        }
        
        cout << "Enter number of Generations: ";
        cin >> generations;
        
        if (generations <= 0) {
            cerr << "Number of generations must be positive!" << endl;
            return 1;
        }
        
        cout << "Use Genetic Algorithm? (yes/no): ";
        cin >> useGAString;
        bool useGA = (useGAString == "yes");
        
        // Network topology: 400 inputs (20x20), 400 hidden, 26 outputs (A-Z)
        vector<unsigned> topology = {400, 400, 26};
        
        cout << "\nNetwork Topology: ";
        for (size_t i = 0; i < topology.size(); ++i) {
            if (i > 0) cout << " -> ";
            cout << topology[i];
        }
        cout << endl;
        
        // Initialize population
        vector<Computer> population;
        population.reserve(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            population.emplace_back(topology);
        }
        
        cout << "\nStarting training..." << endl;
        cout << "Using " << (useGA ? "Genetic Algorithm" : "Backpropagation") << " optimization" << endl << endl;
        
        // Training loop
        for (int gen = 0; gen < generations; ++gen) {
            double sumError = 0.0;
            int sampleCount = 0;
            
            // For each individual in population
            for (auto& computer : population) {
                double trainingError = 0.0;
                
                // Simulate training on dataset (placeholder)
                // In a real scenario, you would load actual training data
                for (int sample = 0; sample < 10; ++sample) {
                    vector<double> inputVals(400, static_cast<double>(sample) / 10.0);
                    vector<double> targetVals(26, 0.0);
                    targetVals[sample % 26] = 1.0;
                    
                    computer.net->feedForward(inputVals);
                    computer.net->backPropagate(targetVals);
                    
                    trainingError += computer.net->getRecentAverageError();
                }
                
                // Set fitness (inverse of error)
                double avgError = trainingError / 10.0;
                computer.setFitness(1.0 / (1.0 + avgError));
                
                sumError += avgError;
                sampleCount++;
            }
            
            // Sort by fitness
            sort(population.begin(), population.end(), comparatorFunc);
            
            double avgError = sumError / sampleCount;
            cout << "Generation " << (gen + 1) << "/" << generations 
                 << " | Avg Error: " << avgError 
                 << " | Best Fitness: " << population[0].getFitness() << endl;
            
            // GA Evolution (optional)
            if (useGA && gen < generations - 1) {
                // Keep top 20% of population
                int eliteSize = max(1, populationSize / 5);
                
                // Breed new individuals
                vector<Computer> newPopulation;
                for (int i = 0; i < eliteSize; ++i) {
                    newPopulation.push_back(population[i]);
                }
                
                // Fill rest with mutated copies
                while (newPopulation.size() < population.size()) {
                    Computer child = newPopulation[rand() % eliteSize];
                    
                    // Mutation: add small random perturbation to weights
                    auto weights = child.net->getWeights();
                    for (auto& w : weights) {
                        w += ((rand() % 100) / 1000.0) - 0.05;
                    }
                    child.net->putWeights(weights);
                    
                    newPopulation.push_back(child);
                }
                
                population = newPopulation;
            }
        }
        
        cout << "\nTraining Complete!" << endl;
        cout << "Best Network Fitness: " << population[0].getFitness() << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
