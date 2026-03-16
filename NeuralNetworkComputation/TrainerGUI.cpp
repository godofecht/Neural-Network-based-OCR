#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>
#include <thread>

// ImGui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// GLFW
#include <GLFW/glfw3.h>

// tinyML
#include "Network.h"

// CImg for image processing
#include "CImg.h"

using namespace std;
using namespace cimg_library;
using namespace ML;

struct TrainingState {
    bool isTraining = false;
    bool isPaused = false;
    int populationSize = 50;
    int totalGenerations = 10;
    bool useGA = true;
    
    int currentGeneration = 0;
    double bestFitness = 0.0;
    double avgFitness = 0.0;
    double prevBestFitness = 0.0;
    
    chrono::steady_clock::time_point startTime;
    int elapsedSeconds = 0;
    
    vector<float> fitnessHistory;
};

void trainingThreadFunc(TrainingState& state) {
    state.startTime = chrono::steady_clock::now();
    state.currentGeneration = 0;
    state.fitnessHistory.clear();
    
    for (int gen = 0; gen < state.totalGenerations && state.isTraining; ++gen) {
        while (state.isPaused && state.isTraining) {
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        
        if (!state.isTraining) break;
        
        state.currentGeneration = gen + 1;
        state.prevBestFitness = state.bestFitness;
        
        // Simulate fitness calculation
        state.bestFitness = 0.3 + gen * 0.03 + (rand() % 100) / 1000.0;
        state.avgFitness = state.bestFitness * 0.8;
        state.fitnessHistory.push_back((float)state.bestFitness);
        
        // Simulate training iteration
        this_thread::sleep_for(chrono::milliseconds(800));
        
        // Update elapsed time
        auto now = chrono::steady_clock::now();
        state.elapsedSeconds = chrono::duration_cast<chrono::seconds>(now - state.startTime).count();
    }
    
    state.isTraining = false;
}

int main() {
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        return 1;
    }

    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1400, 900, 
        "Neural Network OCR Trainer (ImGui + tinyML v1.0.0)", nullptr, nullptr);
    if (!window) {
        cerr << "Failed to create GLFW window" << endl;
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    TrainingState state;
    thread trainingThread;
    ImVec4 clear_color = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_FirstUseEver);
        
        ImGui::Begin("Trainer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), 
                "Neural Network OCR Trainer");
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), 
                "Powered by tinyML v1.0.0");
            ImGui::Separator();

            if (!state.isTraining) {
                ImGui::Text("Configuration");
                ImGui::Separator();
                ImGui::SliderInt("Population Size##config", &state.populationSize, 10, 200);
                ImGui::SliderInt("Generations##config", &state.totalGenerations, 5, 100);
                ImGui::Checkbox("Use Genetic Algorithm##config", &state.useGA);
                
                ImGui::Spacing();
                ImGui::Spacing();
                
                if (ImGui::Button("START TRAINING", ImVec2(250, 40))) {
                    state.isTraining = true;
                    state.isPaused = false;
                    state.fitnessHistory.clear();
                    state.currentGeneration = 0;
                    state.bestFitness = 0.0;
                    if (trainingThread.joinable()) trainingThread.join();
                    trainingThread = thread(trainingThreadFunc, ref(state));
                }
            } else {
                ImGui::Text("Training in Progress");
                ImGui::Separator();
                
                if (ImGui::Button(state.isPaused ? "RESUME" : "PAUSE", ImVec2(120, 40))) {
                    state.isPaused = !state.isPaused;
                }
                ImGui::SameLine();
                if (ImGui::Button("STOP", ImVec2(120, 40))) {
                    state.isTraining = false;
                    if (trainingThread.joinable()) trainingThread.join();
                }
            }

            ImGui::Spacing();
            ImGui::Separator();

            ImGui::Text("Network Architecture");
            ImGui::Columns(4, "network");
            ImGui::Text("Layer"); ImGui::NextColumn();
            ImGui::Text("Input"); ImGui::NextColumn();
            ImGui::Text("Output"); ImGui::NextColumn();
            ImGui::Text("Parameters"); ImGui::NextColumn();
            ImGui::Separator();

            ImGui::Text("Input"); ImGui::NextColumn();
            ImGui::Text("0"); ImGui::NextColumn();
            ImGui::Text("400"); ImGui::NextColumn();
            ImGui::Text("0"); ImGui::NextColumn();

            ImGui::Text("Hidden"); ImGui::NextColumn();
            ImGui::Text("400"); ImGui::NextColumn();
            ImGui::Text("400"); ImGui::NextColumn();
            ImGui::Text("160,000"); ImGui::NextColumn();

            ImGui::Text("Output"); ImGui::NextColumn();
            ImGui::Text("400"); ImGui::NextColumn();
            ImGui::Text("26"); ImGui::NextColumn();
            ImGui::Text("10,400"); ImGui::NextColumn();

            ImGui::Separator();
            ImGui::Text("Total Parameters: 170,400");
            ImGui::Columns(1);

            ImGui::Spacing();
            ImGui::Separator();

            if (state.isTraining || state.fitnessHistory.size() > 0) {
                ImGui::Text("Training Progress");
                ImGui::Separator();
                
                float progress = state.totalGenerations > 0 ? 
                    (float)state.currentGeneration / state.totalGenerations : 0.0f;
                ImGui::ProgressBar(progress, ImVec2(-1, 30));
                
                ImGui::Text("Generation: %d / %d", state.currentGeneration, state.totalGenerations);
                ImGui::Text("Population: %d organisms", state.populationSize);
                
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), 
                    "Best Fitness: %.6f", state.bestFitness);
                ImGui::Text("Average Fitness: %.6f", state.avgFitness);
                
                double improvementRate = state.prevBestFitness > 0 ? 
                    ((state.bestFitness - state.prevBestFitness) / state.prevBestFitness) * 100 : 0;
                ImGui::TextColored(
                    improvementRate > 0 ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.2f, 0.2f, 1.0f),
                    "Improvement: %.2f%%", improvementRate);
                
                int minutes = state.elapsedSeconds / 60;
                int seconds = state.elapsedSeconds % 60;
                ImGui::TextColored(ImVec4(0.2f, 0.6f, 1.0f, 1.0f),
                    "Elapsed: %02dm %02ds", minutes, seconds);

                if (state.fitnessHistory.size() > 1) {
                    ImGui::Spacing();
                    ImGui::PlotLines("Fitness Over Time", 
                        state.fitnessHistory.data(), 
                        (int)state.fitnessHistory.size(),
                        0, nullptr, 0.0f, 1.0f, ImVec2(-1, 150));
                }
            }

            ImGui::End();
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    if (state.isTraining) {
        state.isTraining = false;
        if (trainingThread.joinable()) trainingThread.join();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
