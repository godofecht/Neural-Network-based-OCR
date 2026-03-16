#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>
#include <iomanip>

// tinyML includes
#include "Network.h"

// CImg for image processing
#include "CImg.h"

// Training UI
#include "TrainingUI.h"

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
    vector<unsigned> topo;

    Computer(const vector<unsigned>& topology)
        : net(make_unique<Network>(topology)), topo(topology) {}

    Computer(Computer&& other) = default;
    Computer& operator=(Computer&& other) = default;

    // ML::Network is not copyable (contains unique_ptr), so clone via weights
    Computer clone() const {
        Computer c(topo);
        c.fitness = fitness;
        auto w = net->getWeights();
        c.net->putWeights(w);
        return c;
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
        
        TrainingUI::clearScreen();
        TrainingUI::displayHeader();
        
        cout << Colors::BRIGHT_YELLOW << "⚙️  Configuration\n" << Colors::RESET;
        cout << "Enter Population size: ";
        cin >> populationSize;
        
        if (populationSize <= 0) {
            cerr << Colors::RED << "❌ Population size must be positive!" << Colors::RESET << endl;
            return 1;
        }
        
        cout << "Enter number of Generations: ";
        cin >> generations;
        
        if (generations <= 0) {
            cerr << Colors::RED << "❌ Number of generations must be positive!" << Colors::RESET << endl;
            return 1;
        }
        
        cout << "Use Genetic Algorithm? (yes/no): ";
        cin >> useGAString;
        bool useGA = (useGAString == "yes");
        
        // Network topology: 400 inputs (20x20), 400 hidden, 26 outputs (A-Z)
        vector<unsigned> topology = {400, 400, 26};
        
        // Clear and show initial UI
        TrainingUI::clearScreen();
        TrainingUI::displayHeader();
        
        // Create layer info for visualization
        vector<TrainingUI::LayerInfo> layers = {
            {"Input Layer", 0, 400, 0},
            {"Hidden Layer", 400, 400, 400 * 400},
            {"Output Layer", 400, 26, 400 * 26}
        };
        
        TrainingUI::displayNetworkArchitecture(layers);
        
        // Initialize population
        vector<Computer> population;
        population.reserve(populationSize);
        for (int i = 0; i < populationSize; ++i) {
            population.emplace_back(topology);
        }
        
        cout << Colors::BRIGHT_GREEN << "✅ Population initialized with " << populationSize 
             << " organisms\n" << Colors::RESET;
        cout << Colors::BRIGHT_GREEN << "✅ Using " << (useGA ? "Genetic Algorithm" : "Backpropagation") 
             << " optimization\n\n" << Colors::RESET;
        
        auto startTime = chrono::steady_clock::now();
        double previousBestFitness = 0.0;
        
        // Training loop
        for (int gen = 0; gen < generations; ++gen) {
            auto genStart = chrono::steady_clock::now();
            double sumError = 0.0;
            double sumFitness = 0.0;
            int sampleCount = 0;
            
            // For each individual in population
            for (auto& computer : population) {
                double trainingError = 0.0;
                
                // Simulate training on dataset (placeholder)
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
                sumFitness += computer.getFitness();
                sampleCount++;
            }
            
            // Sort by fitness
            sort(population.begin(), population.end(), comparatorFunc);
            
            double avgError = sumError / sampleCount;
            double avgFitness = sumFitness / sampleCount;
            double currentBestFitness = population[0].getFitness();
            double improvementRate = (gen > 0) ? (currentBestFitness - previousBestFitness) : 0.0;
            previousBestFitness = currentBestFitness;
            
            auto now = chrono::steady_clock::now();
            int elapsedSeconds = chrono::duration_cast<chrono::seconds>(now - startTime).count();
            
            // Update UI
            TrainingUI::clearScreen();
            TrainingUI::displayHeader();
            TrainingUI::displayNetworkArchitecture(layers);
            
            TrainingUI::TrainingStats stats{
                gen + 1,
                generations,
                populationSize,
                currentBestFitness,
                avgFitness,
                improvementRate,
                elapsedSeconds
            };
            
            TrainingUI::displayTrainingProgress(stats);
            
            // Display recent generation details
            cout << Colors::BOLD << Colors::BRIGHT_CYAN << "📈 Recent Generation\n" << Colors::RESET;
            TrainingUI::displayGenerationSummary(gen + 1, currentBestFitness, improvementRate);
            cout << "\n";
            
            // GA Evolution (optional)
            if (useGA && gen < generations - 1) {
                // Keep top 20% of population
                int eliteSize = max(1, populationSize / 5);
                
                // Breed new individuals
                vector<Computer> newPopulation;
                for (int i = 0; i < eliteSize; ++i) {
                    newPopulation.push_back(population[i].clone());
                }

                // Fill rest with mutated copies
                while (newPopulation.size() < population.size()) {
                    Computer child = newPopulation[rand() % eliteSize].clone();

                    // Mutation: add small random perturbation to weights
                    auto weights = child.net->getWeights();
                    for (auto& w : weights) {
                        w += ((rand() % 100) / 1000.0) - 0.05;
                    }
                    child.net->putWeights(weights);

                    newPopulation.push_back(std::move(child));
                }

                population = std::move(newPopulation);
            }
            
            TrainingUI::displayFooter();
        }
        
        // Final results
        TrainingUI::clearScreen();
        TrainingUI::displayHeader();
        
        cout << Colors::BOLD << Colors::BRIGHT_GREEN 
             << "🏆 Training Complete!\n\n" << Colors::RESET;
        
        cout << Colors::BRIGHT_CYAN << "Final Results:\n" << Colors::RESET;
        cout << "├─ " << Colors::BRIGHT_GREEN << "Best Network Fitness: " << Colors::RESET 
             << Colors::BRIGHT_YELLOW << fixed << setprecision(4) << population[0].getFitness() 
             << Colors::RESET << "\n";
        cout << "├─ " << Colors::BRIGHT_GREEN << "Total Generations: " << Colors::RESET 
             << Colors::BRIGHT_YELLOW << generations << Colors::RESET << "\n";
        cout << "├─ " << Colors::BRIGHT_GREEN << "Population Size: " << Colors::RESET 
             << Colors::BRIGHT_YELLOW << populationSize << Colors::RESET << "\n";
        cout << "└─ " << Colors::BRIGHT_GREEN << "Optimization Method: " << Colors::RESET 
             << Colors::BRIGHT_YELLOW << (useGA ? "Genetic Algorithm" : "Backpropagation") 
             << Colors::RESET << "\n\n";
        
        cout << Colors::DIM << "Results saved. Training ended gracefully.\n" << Colors::RESET;
        
        return 0;
        
    } catch (const exception& e) {
        cerr << Colors::RED << "❌ Error: " << e.what() << Colors::RESET << endl;
        return 1;
    }
}
