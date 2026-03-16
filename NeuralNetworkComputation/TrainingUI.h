#ifndef TRAINING_UI_H
#define TRAINING_UI_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

using namespace std;

// ANSI color codes
namespace Colors {
    const string RESET = "\033[0m";
    const string BOLD = "\033[1m";
    const string DIM = "\033[2m";
    
    // Foreground colors
    const string BLACK = "\033[30m";
    const string RED = "\033[31m";
    const string GREEN = "\033[32m";
    const string YELLOW = "\033[33m";
    const string BLUE = "\033[34m";
    const string MAGENTA = "\033[35m";
    const string CYAN = "\033[36m";
    const string WHITE = "\033[37m";
    
    // Bright colors
    const string BRIGHT_GREEN = "\033[92m";
    const string BRIGHT_YELLOW = "\033[93m";
    const string BRIGHT_BLUE = "\033[94m";
    const string BRIGHT_MAGENTA = "\033[95m";
    const string BRIGHT_CYAN = "\033[96m";
}

class TrainingUI {
public:
    struct LayerInfo {
        string name;
        size_t inputSize;
        size_t outputSize;
        size_t parameters;
    };
    
    struct TrainingStats {
        int generation;
        int totalGenerations;
        int populationSize;
        double bestFitness;
        double avgFitness;
        double improvementRate;
        int elapsedSeconds;
    };

    static void clearScreen() {
        cout << "\033[2J\033[H";
    }

    static void displayHeader() {
        cout << Colors::BRIGHT_CYAN << Colors::BOLD 
             << "╔════════════════════════════════════════════════════════════════════════════════╗" << endl
             << "║         Neural Network OCR Trainer (Powered by tinyML v1.0.0)          ║" << endl
             << "╚════════════════════════════════════════════════════════════════════════════════╝"
             << Colors::RESET << endl << endl;
    }

    static void displayNetworkArchitecture(const vector<LayerInfo>& layers) {
        cout << Colors::BOLD << Colors::BRIGHT_BLUE 
             << "Network Architecture" << endl
             << Colors::RESET;
        
        // Top border
        cout << "┌";
        for (int i = 0; i < 82; ++i) cout << "─";
        cout << "┐" << endl;
        
        size_t totalParams = 0;
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            totalParams += layer.parameters;
            
            cout << "│ ";
            cout << Colors::BRIGHT_GREEN << setw(3) << (i + 1) << Colors::RESET << ". ";
            cout << setw(15) << left << layer.name;
            
            cout << Colors::CYAN << "  │  ";
            cout << "Input: " << setw(5) << layer.inputSize << " ";
            cout << "=> Output: " << setw(5) << layer.outputSize << " ";
            cout << Colors::RESET << "│ ";
            
            cout << Colors::YELLOW << "Params: " << setw(7) << layer.parameters 
                 << Colors::RESET << endl;
        }
        
        // Middle border
        cout << "├";
        for (int i = 0; i < 82; ++i) cout << "─";
        cout << "┤" << endl;
        cout << "│ " << Colors::BOLD << "Total Parameters: " << Colors::RESET 
             << Colors::BRIGHT_YELLOW << setw(10) << totalParams 
             << Colors::RESET;
        for (int i = 0; i < 52; ++i) cout << " ";
        cout << " │" << endl;
        
        // Bottom border
        cout << "└";
        for (int i = 0; i < 82; ++i) cout << "─";
        cout << "┘" << endl << endl;
    }

    static void displayTrainingProgress(const TrainingStats& stats) {
        cout << Colors::BOLD << Colors::BRIGHT_MAGENTA 
             << "Training Progress" << endl
             << Colors::RESET;
        
        // Progress bar
        int barWidth = 60;
        int progress = (stats.generation * barWidth) / stats.totalGenerations;
        
        cout << "Generation [";
        cout << Colors::BRIGHT_GREEN;
        for (int i = 0; i < progress; ++i) cout << "#";
        cout << Colors::DIM;
        for (int i = progress; i < barWidth; ++i) cout << "-";
        cout << Colors::RESET << "] ";
        
        cout << Colors::BOLD << setw(3) << stats.generation << "/" << stats.totalGenerations 
             << Colors::RESET << endl << endl;
        
        // Stats grid
        cout << "┌─────────────────────────────────┬─────────────────────────────────┐" << endl;
        
        // Population Size
        cout << "│ " << Colors::CYAN << "Population Size" << Colors::RESET;
        for (int i = 0; i < 16; ++i) cout << " ";
        cout << "│ ";
        cout << Colors::BRIGHT_YELLOW << setw(5) << stats.populationSize 
             << Colors::RESET << " organisms";
        for (int i = 0; i < 18; ++i) cout << " ";
        cout << "│" << endl;
        
        // Best Fitness
        cout << "│ " << Colors::CYAN << "Best Fitness" << Colors::RESET;
        for (int i = 0; i < 19; ++i) cout << " ";
        cout << "│ ";
        cout << Colors::BRIGHT_GREEN << fixed << setprecision(4) 
             << setw(8) << stats.bestFitness << Colors::RESET;
        for (int i = 0; i < 20; ++i) cout << " ";
        cout << "│" << endl;
        
        // Average Fitness
        cout << "│ " << Colors::CYAN << "Average Fitness" << Colors::RESET;
        for (int i = 0; i < 16; ++i) cout << " ";
        cout << "│ ";
        cout << Colors::BRIGHT_YELLOW << fixed << setprecision(4) 
             << setw(8) << stats.avgFitness << Colors::RESET;
        for (int i = 0; i < 20; ++i) cout << " ";
        cout << "│" << endl;
        
        // Improvement Rate
        cout << "│ " << Colors::CYAN << "Improvement Rate" << Colors::RESET;
        for (int i = 0; i < 15; ++i) cout << " ";
        cout << "│ ";
        cout << (stats.improvementRate > 0 ? Colors::BRIGHT_GREEN : Colors::RED);
        cout << fixed << setprecision(2) << setw(7) 
             << (stats.improvementRate * 100) << "%" << Colors::RESET;
        for (int i = 0; i < 17; ++i) cout << " ";
        cout << "│" << endl;
        
        // Elapsed Time
        cout << "│ " << Colors::CYAN << "Elapsed Time" << Colors::RESET;
        for (int i = 0; i < 19; ++i) cout << " ";
        cout << "│ ";
        int hours = stats.elapsedSeconds / 3600;
        int minutes = (stats.elapsedSeconds % 3600) / 60;
        int seconds = stats.elapsedSeconds % 60;
        cout << Colors::BRIGHT_BLUE;
        if (hours > 0) cout << hours << "h ";
        cout << setfill('0') << setw(2) << minutes << "m "
             << setw(2) << seconds << "s" << Colors::RESET;
        for (int i = 0; i < 15; ++i) cout << " ";
        cout << "│" << endl;
        
        cout << "└─────────────────────────────────┴─────────────────────────────────┘" << endl << endl;
    }

    static void displayGenerationSummary(int generation, double fitness, double deltaFitness) {
        cout << Colors::DIM << "Generation " << Colors::RESET 
             << Colors::BRIGHT_YELLOW << generation << Colors::RESET 
             << Colors::DIM << " - Fitness: " << Colors::RESET;
        
        cout << Colors::BRIGHT_GREEN << fixed << setprecision(4) 
             << fitness << Colors::RESET;
        
        if (deltaFitness > 0) {
            cout << Colors::BRIGHT_GREEN << " UP +" << deltaFitness << Colors::RESET;
        } else if (deltaFitness < 0) {
            cout << Colors::RED << " DOWN " << deltaFitness << Colors::RESET;
        } else {
            cout << Colors::DIM << " STABLE" << Colors::RESET;
        }
        cout << endl;
    }

    static void displayFooter() {
        cout << endl << Colors::DIM 
             << "───────────────────────────────────────────────────────────────────────────────────" << endl
             << "Press Ctrl+C to stop training" << endl
             << Colors::RESET;
    }
};

#endif // TRAINING_UI_H
