#include "experiments/training_logger.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

void TrainingLogger::ensureResultsDirs() {
    fs::create_directories("results");
    fs::create_directories("results/epochs");
}

void TrainingLogger::logEpochLoss(const std::string &runName, int epoch, float avgLoss) {
    ensureResultsDirs();

    const std::string path = "results/epochs/" + runName + ".csv";
    const bool exists = fs::exists(path);

    std::ofstream out(path, std::ios::app);
    if (!exists) {
        out << "epoch,avg_loss\n";
    }

    out << epoch << "," << avgLoss << "\n";
}

void TrainingLogger::logSummary(const RunConfig& config, float finalAccuracy) {
    ensureResultsDirs();

    const std::string path = "results/summary.csv";
    const bool exists = fs::exists(path);

    std::ofstream out(path, std::ios::app);
    if (!exists) {
        out << "run_name,train_limit,test_limit,learning_rate,batch_size,epochs,final_accuracy\n";
    }

    out << config.runName << ","
        << config.trainLimit << ","
        << config.testLimit << ","
        << config.learningRate << ","
        << config.batchSize << ","
        << config.epochs << ","
        << finalAccuracy << "\n";
}
