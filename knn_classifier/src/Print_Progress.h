#ifndef KNN_CLASSIFIER_PRINT_PROGRESS_H
#define KNN_CLASSIFIER_PRINT_PROGRESS_H


#include <array>
#include <string>
#include <vector>

class Print_Progress {
public:
    Print_Progress();

    // Constructors
    Print_Progress(int threads, int max_progress);

    // Destructor
    ~Print_Progress() = default;

    // Getters

    // Setters
    void setProgress(int id, int thread_progress);

    // Functions
    void printProgress();
    void printTableHeader();

private:
    int n_threads {16};  /// The number of threads
    int progress_max {625};  /// The maximum progress value
    std::vector<int> progress {0};  /// The progress of each thread
    std::vector<std::string> progress_increment {};  /// The string increment for each thread

    void updateProgress(int caller_id);
};


#endif
