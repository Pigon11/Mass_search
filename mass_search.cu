#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <algorithm>

using namespace std;

struct SymbolEntry {
    int substr_id;
    int position;
};

// CPU

void searchCPU(const vector<unsigned char>& buffer,
               const vector<vector<unsigned char>>& substrings,
               vector<unsigned char>& results) {
    int h = buffer.size();
    int n_sub = substrings.size();
    results.assign(n_sub, 0);

    for (int i = 0; i < n_sub; i++) {
        const auto& pattern = substrings[i];
        int len = pattern.size();

        for (int j = 0; j <= h - len; j++) {
            bool match = true;
            for (int k = 0; k < len; k++) {
                if (buffer[j + k] != pattern[k]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                results[i] = 1;
                break;
            }
        }
    }
}

// Простой GPU

__global__ void simpleSearchKernel(const unsigned char* buffer, int buffer_size,
                                    const unsigned char* substrings, const int* substr_lengths,
                                    int num_substrings, unsigned char* found_flags) {
    int sub_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (sub_id >= num_substrings) return;

    int len = substr_lengths[sub_id];
    const unsigned char* pattern = &substrings[sub_id * 256];

    for (int start = 0; start <= buffer_size - len; start++) {
        bool match = true;
        for (int k = 0; k < len; k++) {
            if (buffer[start + k] != pattern[k]) {
                match = false;
                break;
            }
        }
        if (match) {
            found_flags[sub_id] = 1;
            break;
        }
    }
}

// Матричный GPU 

__global__ void matrixSearchKernel(const unsigned char* buffer, int buffer_size,
                                    const SymbolEntry* symbol_index, const int* index_offsets,
                                    int* matrix, int num_substrings) {

    int pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos >= buffer_size) return;

    unsigned char c = buffer[pos];
    int start = index_offsets[c];
    int end = index_offsets[c + 1];

    for (int idx = start; idx < end; idx++) {
        SymbolEntry entry = symbol_index[idx];
        int substr_id = entry.substr_id;
        int char_pos = entry.position;

        int matrix_pos = pos - char_pos;
        if (matrix_pos >= 0 && matrix_pos < buffer_size) {
            atomicSub((unsigned int*)&matrix[substr_id * buffer_size + matrix_pos], 1);
        }
    }
}

// Генерация данных

void generateData(vector<unsigned char>& buffer, int buffer_size,
                  vector<vector<unsigned char>>& substrings, int num_substrings,
                  int min_len, int max_len) {

    random_device rd;
    mt19937 gen(42);
    uniform_int_distribution<> byte_dist(0, 255);
    uniform_int_distribution<> len_dist(min_len, max_len);
    uniform_int_distribution<> pos_dist(0, buffer_size - min_len);

    buffer.resize(buffer_size);
    for (int i = 0; i < buffer_size; i++) {
        buffer[i] = byte_dist(gen);
    }

    substrings.resize(num_substrings);
    for (int i = 0; i < num_substrings; i++) {
        int len = len_dist(gen);
        if (i % 5 != 0) {
            int start = pos_dist(gen);
            substrings[i].assign(buffer.begin() + start, buffer.begin() + start + len);
        } else {
            substrings[i].resize(len);
            for (int j = 0; j < len; j++) {
                substrings[i][j] = byte_dist(gen);
            }
        }
    }
}

int main() {
    cout << "========================================\n";
    cout << "Mass Substring Search Laboratory Work\n";
    cout << "========================================\n\n";

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, 0);
        cout << "GPU Device: " << props.name << "\n";
        cout << "Compute Capability: " << props.major << "." << props.minor << "\n";
        cout << "Total memory: " << props.totalGlobalMem / (1024*1024) << " MB\n\n";
    }

    const int BUFFER_SIZE = 100000;
    const int NUM_SUBSTRINGS = 2000;
    const int MIN_LEN = 3;
    const int MAX_LEN = 15;

    cout << "Parameters:\n";
    cout << "  Buffer size: " << BUFFER_SIZE << " bytes\n";
    cout << "  Number of substrings: " << NUM_SUBSTRINGS << "\n";
    cout << "  Substring length: " << MIN_LEN << " - " << MAX_LEN << " bytes\n\n";

    cout << "Generating test data\n";
    vector<unsigned char> buffer;
    vector<vector<unsigned char>> substrings;
    generateData(buffer, BUFFER_SIZE, substrings, NUM_SUBSTRINGS, MIN_LEN, MAX_LEN);

    cout << "\nRunning CPU search\n";
    vector<unsigned char> cpu_results;
    auto cpu_start = chrono::high_resolution_clock::now();
    searchCPU(buffer, substrings, cpu_results);
    auto cpu_end = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double, milli>(cpu_end - cpu_start).count();
    int cpu_found = 0;
    for (unsigned char val : cpu_results) if (val) cpu_found++;
    cout << "  CPU time: " << fixed << setprecision(2) << cpu_time << " ms\n";
    cout << "  Found: " << cpu_found << "/" << NUM_SUBSTRINGS << "\n";

    cout << "\nRunning Simple GPU search\n";
    const int MAX_SUBSTR_LEN = 256;
    vector<unsigned char> flat_substrings(NUM_SUBSTRINGS * MAX_SUBSTR_LEN, 0);
    vector<int> substr_lengths(NUM_SUBSTRINGS);
    for (int i = 0; i < NUM_SUBSTRINGS; i++) {
        substr_lengths[i] = substrings[i].size();
        for (size_t j = 0; j < substrings[i].size(); j++) {
            flat_substrings[i * MAX_SUBSTR_LEN + j] = substrings[i][j];
        }
    }

    unsigned char* d_buffer;
    unsigned char* d_substrings;
    int* d_lengths;
    unsigned char* d_results;

    cudaMalloc(&d_buffer, BUFFER_SIZE * sizeof(unsigned char));
    cudaMalloc(&d_substrings, NUM_SUBSTRINGS * MAX_SUBSTR_LEN * sizeof(unsigned char));
    cudaMalloc(&d_lengths, NUM_SUBSTRINGS * sizeof(int));
    cudaMalloc(&d_results, NUM_SUBSTRINGS * sizeof(unsigned char));

    cudaMemcpy(d_buffer, buffer.data(), BUFFER_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_substrings, flat_substrings.data(), NUM_SUBSTRINGS * MAX_SUBSTR_LEN * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, substr_lengths.data(), NUM_SUBSTRINGS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, NUM_SUBSTRINGS * sizeof(unsigned char));

    int threads_per_block = 256;
    int blocks = (NUM_SUBSTRINGS + threads_per_block - 1) / threads_per_block;

    cudaEvent_t simple_start, simple_stop;
    cudaEventCreate(&simple_start);
    cudaEventCreate(&simple_stop);

    cudaEventRecord(simple_start);
    simpleSearchKernel<<<blocks, threads_per_block>>>(d_buffer, BUFFER_SIZE, d_substrings, d_lengths, NUM_SUBSTRINGS, d_results);
    cudaEventRecord(simple_stop);
    cudaEventSynchronize(simple_stop);

    float simple_time_ms;
    cudaEventElapsedTime(&simple_time_ms, simple_start, simple_stop);

    vector<unsigned char> gpu_simple_results(NUM_SUBSTRINGS);
    cudaMemcpy(gpu_simple_results.data(), d_results, NUM_SUBSTRINGS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    int gpu_simple_found = 0;
    for (unsigned char val : gpu_simple_results) if (val) gpu_simple_found++;
    cout << "  GPU time: " << fixed << setprecision(2) << simple_time_ms << " ms\n";
    cout << "  Found: " << gpu_simple_found << "/" << NUM_SUBSTRINGS << "\n";

    cout << "\nRunning Matrix Method GPU search\n";
    vector<SymbolEntry> symbol_index[256];
    for (int i = 0; i < NUM_SUBSTRINGS; i++) {
        for (size_t j = 0; j < substrings[i].size(); j++) {
            unsigned char ch = substrings[i][j];
            symbol_index[ch].push_back({i, (int)j});
        }
    }

    vector<SymbolEntry> flat_index;
    vector<int> offsets(257, 0);
    for (int c = 0; c < 256; c++) {
        offsets[c] = flat_index.size();
        flat_index.insert(flat_index.end(), symbol_index[c].begin(), symbol_index[c].end());
    }
    offsets[256] = flat_index.size();

    vector<int> matrix_cpu(NUM_SUBSTRINGS * BUFFER_SIZE);
    for (int i = 0; i < NUM_SUBSTRINGS; i++) {
        int len = substrings[i].size();
        for (int j = 0; j < BUFFER_SIZE; j++) {
            matrix_cpu[i * BUFFER_SIZE + j] = len;
        }
    }

    unsigned char* d_buffer_mat;
    SymbolEntry* d_index;
    int* d_offsets;
    int* d_matrix;

    cudaMalloc(&d_buffer_mat, BUFFER_SIZE * sizeof(unsigned char));
    cudaMalloc(&d_index, flat_index.size() * sizeof(SymbolEntry));
    cudaMalloc(&d_offsets, 257 * sizeof(int));
    cudaMalloc(&d_matrix, NUM_SUBSTRINGS * BUFFER_SIZE * sizeof(int));

    cudaMemcpy(d_buffer_mat, buffer.data(), BUFFER_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index, flat_index.data(), flat_index.size() * sizeof(SymbolEntry), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), 257 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix, matrix_cpu.data(), NUM_SUBSTRINGS * BUFFER_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t matrix_start, matrix_stop;
    cudaEventCreate(&matrix_start);
    cudaEventCreate(&matrix_stop);

    int matrix_blocks = (BUFFER_SIZE + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(matrix_start);
    matrixSearchKernel<<<matrix_blocks, threads_per_block>>>(d_buffer_mat, BUFFER_SIZE, d_index, d_offsets, d_matrix, NUM_SUBSTRINGS);
    cudaEventRecord(matrix_stop);
    cudaEventSynchronize(matrix_stop);

    float matrix_time_ms;
    cudaEventElapsedTime(&matrix_time_ms, matrix_start, matrix_stop);
    cudaMemcpy(matrix_cpu.data(), d_matrix, NUM_SUBSTRINGS * BUFFER_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    vector<unsigned char> matrix_results(NUM_SUBSTRINGS, 0);
    for (int i = 0; i < NUM_SUBSTRINGS; i++) {
        int len = substrings[i].size();
        for (int j = 0; j <= BUFFER_SIZE - len; j++) {
            if (matrix_cpu[i * BUFFER_SIZE + j] == 0) {
                matrix_results[i] = 1;
                break;
            }
        }
    }

    int matrix_found = 0;
    for (unsigned char val : matrix_results) if (val) matrix_found++;
    cout << "  Matrix GPU time: " << fixed << setprecision(2) << matrix_time_ms << " ms\n";
    cout << "  Found: " << matrix_found << "/" << NUM_SUBSTRINGS << "\n";

    cout << "\n========================================\n";
    cout << "Verification Results\n";
    cout << "========================================\n";

    bool simple_match = true;
    bool matrix_match = true;
    for (int i = 0; i < NUM_SUBSTRINGS; i++) {
        if (cpu_results[i] != gpu_simple_results[i]) simple_match = false;
        if (cpu_results[i] != matrix_results[i]) matrix_match = false;
    }

    cout << "Simple GPU method vs CPU: " << (simple_match ? "+ MATCH" : "- MISMATCH") << "\n";
    cout << "Matrix GPU method vs CPU: " << (matrix_match ? "+ MATCH" : "- MISMATCH") << "\n";

    cout << "\n========================================\n";
    cout << "Performance Summary\n";
    cout << "========================================\n";
    cout << "CPU time:        " << fixed << setprecision(2) << cpu_time << " ms\n";
    cout << "Simple GPU time: " << simple_time_ms << " ms\n";
    cout << "Matrix GPU time: " << matrix_time_ms << " ms\n";
    cout << "\nSpeedup (Simple GPU): " << (cpu_time / simple_time_ms) << "x\n";
    cout << "Speedup (Matrix GPU): " << (cpu_time / matrix_time_ms) << "x\n";

    cudaFree(d_buffer);
    cudaFree(d_substrings);
    cudaFree(d_lengths);
    cudaFree(d_results);
    cudaFree(d_buffer_mat);
    cudaFree(d_index);
    cudaFree(d_offsets);
    cudaFree(d_matrix);

    cudaEventDestroy(simple_start);
    cudaEventDestroy(simple_stop);
    cudaEventDestroy(matrix_start);
    cudaEventDestroy(matrix_stop);

    return 0;
}