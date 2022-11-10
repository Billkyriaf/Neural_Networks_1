#ifndef KNN_CLASSIFIER_TIMER_H
#define KNN_CLASSIFIER_TIMER_H

#include <ctime>
#include <string>

class Timer {
public:
    // Constructors
    Timer() = default;

    // Destructor
    ~Timer() = default;

    // Getters

    // Setters

    // Functions
    void startTimer();
    void stopTimer();
    void displayElapsed() const;

private:
    struct timespec start;
    struct timespec stop;
    long elapsed_sec {0};
    long elapsed_ms {0};
    long elapsed_us {0};
    long elapsed_ns {0};

    void elapsedTime();
};


#endif
