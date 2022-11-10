#include <iostream>
#include "Timer.h"


/**
 * Calculates the elapsed time using the start and stop timespec struct of the timer
 *
 * @param timer  The timer struct
 */
void Timer::elapsedTime() {
    elapsed_sec = stop.tv_sec - start.tv_sec;  // Get the hole number of seconds

    long ns = 0;

    if (stop.tv_nsec < start.tv_nsec){  // If a hole second has not passed (eg 5.2s to 6.1s)
        elapsed_sec--; // decrease the seconds
        ns = 1000000000 - start.tv_nsec + stop.tv_nsec;  // and get the nanoseconds passed

    } else if (stop.tv_nsec > start.tv_nsec) { // Else if a hole second has passed (eg 5.2s to 6.3s)
        ns = stop.tv_nsec - start.tv_nsec;  // get the nanoseconds elapsed
    }

    // Extract the micro and milli seconds from the nano seconds

    /*
     * Every 1000ns is 1us
     * Every 1000us is 1ms
     * Every 1000ms is 1s
     */
    if (ns < 1000) {
        elapsed_ms = 0;
        elapsed_us = 0;
        elapsed_ns = ns;

    } else {
        long us = ns / 1000;
        elapsed_ns = ns % 1000;

        if (us < 1000) {
            elapsed_ms = 0;
            elapsed_us = us;

        } else {
            elapsed_ms = us / 1000;
            elapsed_us = us % 1000;
        }
    }
}

/**
 * Store the starting moment in the start struct
 * @param timer  The timer struct
 */
void Timer::startTimer() {
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
}

/**
 * Store the stopping moment in the stop struct
 * @param timer  The timer struct
 */
void Timer::stopTimer() {
    clock_gettime(CLOCK_MONOTONIC_RAW, &stop);

    elapsedTime();
}

/**
 * Print the elapsed time
 * @param timer  The timer struct
 */
void Timer::displayElapsed() const {
    std::cout << elapsed_sec << "s " << elapsed_ms << "ms " << elapsed_us
              << "us " << elapsed_ns << "ns\n" << std::endl;
}
