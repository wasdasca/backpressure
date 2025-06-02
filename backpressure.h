#ifndef BACKPRESSURE_H
#define BACKPRESSURE_H

#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>

// PID 控制器，负责计算 delay 调节值
class PIDController {
public:
    PIDController(double kp, double ki, double kd, double dt);

    // 根据当前误差计算新的控制值
    double update(double error);

private:
    double kp_, ki_, kd_;
    double dt_;
    double prev_error_;
    double integral_;
};

// 内存监控管理器，实时获取内存使用量，和总内存阈值判断
class MemoryGovernor {
public:
    MemoryGovernor(size_t maxMemoryBytes);

    // 返回当前内存使用量 (单位: bytes)
    size_t getCurrentMemoryUsage() const;

    // 是否达到高内存使用阈值
    bool isMemoryPressureHigh() const;

    // 是否达到极限内存使用阈值
    bool isMemoryPressureCritical() const;

private:
    size_t maxMemoryBytes_;
};

// 背压状态机，管理状态及delay时长调整
enum class BackpressureState {
    NORMAL,
    LIGHT_PRESSURE,
    HEAVY_PRESSURE,
    STOP_PRODUCTION
};

// 统一背压控制器，结合内存和队列长，管理delay，状态机，PID
class AdaptiveBackpressure {
public:
    AdaptiveBackpressure(size_t maxMemoryBytes, size_t maxQueueLength);

    // 外部调用，更新当前队列长度和内存占用
    void updateStats(size_t currentQueueLength, size_t currentMemoryUsage);

    // 返回本次任务创建前需要sleep的延迟时间（毫秒）
    int getDelayMs();

    // 查询当前背压状态
    BackpressureState getState() const;

private:
    void updateState();
    int calculateDelay();

    MemoryGovernor memGovernor_;
    PIDController pidController_;

    std::atomic<size_t> currentQueueLength_;
    std::atomic<size_t> currentMemoryUsage_;

    BackpressureState state_;

    // delay 基础值
    int baseDelayMs_;
    int maxDelayMs_;
};

#endif // BACKPRESSURE_H
