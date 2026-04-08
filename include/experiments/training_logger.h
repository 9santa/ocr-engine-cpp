#pragma once

#include <string>

struct RunConfig {
    std::string runName;
    int trainLimit = 0;
    int testLimit = 0;
    float learningRate = 0.0f;
    int batchSize = 0;
    int epochs = 0;
};

class TrainingLogger {
public:
    static void ensureResultsDirs();

    static void logEpochLoss(
        const std::string& runName,
        int epoch,
        float avgLoss);

    static void logSummary(
        const RunConfig& config,
        float finalAccuracy);

};
