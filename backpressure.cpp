#include "backpressure.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

// --- PIDController 实现 ---
PIDController::PIDController(double kp, double ki, double kd, double dt)
    : kp_(kp), ki_(ki), kd_(kd), dt_(dt), prev_error_(0.0), integral_(0.0) {}

double PIDController::update(double error) {
    integral_ += error * dt_;
    double derivative = (error - prev_error_) / dt_;
    double output = kp_ * error + ki_ * integral_ + kd_ * derivative;
    prev_error_ = error;
    return output;
}

// --- MemoryGovernor 实现 ---

MemoryGovernor::MemoryGovernor(size_t maxMemoryBytes)
    : maxMemoryBytes_(maxMemoryBytes) {}

// 这里示范Linux方式获取RSS内存使用，也可扩展Windows版本
size_t MemoryGovernor::getCurrentMemoryUsage() const {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss * 1024; // ru_maxrss单位KB，转bytes
    }
    return 0;
#endif
}

bool MemoryGovernor::isMemoryPressureHigh() const {
    size_t usage = getCurrentMemoryUsage();
    return usage > maxMemoryBytes_ * 0.7;
}

bool MemoryGovernor::isMemoryPressureCritical() const {
    size_t usage = getCurrentMemoryUsage();
    return usage > maxMemoryBytes_ * 0.9;
}

// --- AdaptiveBackpressure 实现 ---

AdaptiveBackpressure::AdaptiveBackpressure(size_t maxMemoryBytes, size_t maxQueueLength)
    : memGovernor_(maxMemoryBytes),
      pidController_(0.5, 0.1, 0.05, 1.0), // PID 参数可调
      currentQueueLength_(0),
      currentMemoryUsage_(0),
      state_(BackpressureState::NORMAL),
      baseDelayMs_(5),
      maxDelayMs_(200) {}

void AdaptiveBackpressure::updateStats(size_t currentQueueLength, size_t currentMemoryUsage) {
    currentQueueLength_.store(currentQueueLength, std::memory_order_relaxed);
    currentMemoryUsage_.store(currentMemoryUsage, std::memory_order_relaxed);
    updateState();
}

void AdaptiveBackpressure::updateState() {
    size_t memUse = currentMemoryUsage_.load(std::memory_order_relaxed);
    size_t queueLen = currentQueueLength_.load(std::memory_order_relaxed);

    // 根据内存和队列长度阈值判定状态
    if (memUse > memGovernor_.maxMemoryBytes_ * 0.9 || queueLen > 10000) {
        state_ = BackpressureState::STOP_PRODUCTION;
    }
    else if (memUse > memGovernor_.maxMemoryBytes_ * 0.8 || queueLen > 5000) {
        state_ = BackpressureState::HEAVY_PRESSURE;
    }
    else if (memUse > memGovernor_.maxMemoryBytes_ * 0.7 || queueLen > 1000) {
        state_ = BackpressureState::LIGHT_PRESSURE;
    }
    else {
        state_ = BackpressureState::NORMAL;
    }
}

int AdaptiveBackpressure::calculateDelay() {
    double targetDelay = 0.0;
    switch (state_) {
    case BackpressureState::NORMAL:
        targetDelay = baseDelayMs_ * 0.5;
        break;
    case BackpressureState::LIGHT_PRESSURE:
        targetDelay = baseDelayMs_ * 5;
        break;
    case BackpressureState::HEAVY_PRESSURE:
        targetDelay = maxDelayMs_ * 0.7;
        break;
    case BackpressureState::STOP_PRODUCTION:
        targetDelay = maxDelayMs_;
        break;
    }

    // 误差 = 目标delay - 当前delay，利用PID调节
    static double lastDelay = 0;
    double error = targetDelay - lastDelay;
    double adjusted = pidController_.update(error);
    double newDelay = lastDelay + adjusted;

    // 限制范围
    newDelay = std::max(0.0, std::min(static_cast<double>(maxDelayMs_), newDelay));
    lastDelay = newDelay;

    return static_cast<int>(newDelay);
}

int AdaptiveBackpressure::getDelayMs() {
    return calculateDelay();
}

BackpressureState AdaptiveBackpressure::getState() const {
    return state_;
}
