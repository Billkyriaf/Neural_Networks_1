#include "Print_Progress.h"

#include <iostream>


/**
 * Constructor for Print_Progress
 *
 * @param threads   The number of threads to print progress for
 * @param total     The total number of images to process
 */
Print_Progress::Print_Progress() : n_threads(16), progress_max(625){
    progress.resize(n_threads);
    progress_increment.resize(n_threads);

    // init progress array
    for (int i = 0; i < 16; i++) {
        progress[i] = 0;
        progress_increment[i] = ">________";
    }

    printTableHeader();
    printProgress();
}


/**
 * Set the progress of a thread
 *
 * @param id                The id of the thread
 * @param thread_progress   The progress of the thread
 */
Print_Progress::Print_Progress(int threads, int max_progress) : n_threads(threads), progress_max(max_progress) {
    progress.resize(n_threads);
    progress_increment.resize(n_threads);

    // init progress array
    for (int i = 0; i < n_threads; i++) {
        progress.at(i) = 0;
        progress_increment.at(i) = ">________";
    }

    printTableHeader();
    printProgress();
}


/**
 * Print the progress table header
 */
void Print_Progress::printTableHeader() {
    std::string header = "|    id    |";
    int dash_count = 12;
    for (int i = 0; i < n_threads; i++) {
        if (i < 10) {
            header += "   t " + std::to_string(i) + "   |";
        } else {
            header += "  t " + std::to_string(i) + "   |";
        }
        dash_count += 10;
    }
    std::cout << header << std::endl;
    for (int i = 0; i < dash_count; i++) {
        std::cout << "-";
    }
    std::cout << std::endl;

}

/**
 * Print the updated the progress line
 */
void Print_Progress::printProgress(){
    std::string progress_string = "\r| progress |";
    for (int i = 0; i < n_threads; i++) {
        progress_string += progress_increment[i] + "|";
    }
    std::cout << progress_string << std::flush;
}


/**
 * Set the progress of a thread
 *
 * @param caller_id The id of the thread that called this function
 */
void Print_Progress::updateProgress(int caller_id) {
    // normalize the progress
    progress[caller_id] = (progress[caller_id] * 100) / progress_max;


    if (progress[caller_id] < 1 * 100 / 8){
        progress_increment[caller_id] = "=>_______";
    } else if (progress[caller_id] < 2 * 100 / 8){
        progress_increment[caller_id] = "==>______";
    } else if (progress[caller_id] < 3 * 100 / 8){
        progress_increment[caller_id] = "===>_____";
    } else if (progress[caller_id] < 4 * 100 / 8){
        progress_increment[caller_id] = "====>____";
    } else if (progress[caller_id] < 5 * 100 / 8){
        progress_increment[caller_id] = "=====>___";
    } else if (progress[caller_id] < 6 * 100 / 8){
        progress_increment[caller_id] = "======>__";
    } else if (progress[caller_id] < 7 * 100 / 8){
        progress_increment[caller_id] = "=======>_";
    } else if (progress[caller_id] < 8 * 100 / 8){
        progress_increment[caller_id] = "========>";
    } else {
        progress_increment[caller_id] = "  Done!  ";
    }


}

/**
 * Set the progress of a thread
 *
 * @param id                The id of the thread
 * @param thread_progress   The progress of the thread
 */
void Print_Progress::setProgress(int id, int thread_progress) {
    this->progress.at(id) = thread_progress;
    updateProgress(id);
    printProgress();
}
