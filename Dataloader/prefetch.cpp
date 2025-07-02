#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <xmmintrin.h> // For _mm_prefetch
#include <cstdlib>
#include <cstring>

namespace py = pybind11;

// Simple ReLU kernel with prefetching
void process_batch(py::array_t<float, py::array::c_style | py::array::forcecast> batch) {
    auto buf = batch.request();
    float* data = static_cast<float*>(buf.ptr);

    const ssize_t batch_size = buf.shape[0];
    const ssize_t feature_size = buf.shape[1];

    for (ssize_t i = 0; i < batch_size; ++i) {
        // Prefetch next sample into L1 cache if in range
        if (i + 1 < batch_size) {
            _mm_prefetch(reinterpret_cast<const char*>(&data[(i + 1) * feature_size]), _MM_HINT_T0);
        }

        // Apply ReLU
        for (ssize_t j = 0; j < feature_size; ++j) {
            float& x = data[i * feature_size + j];
            x = x > 0.0f ? x : 0.0f;
        }
    }
}

PYBIND11_MODULE(prefetch, m) {
    m.def("process_batch", &process_batch, "Batch processing with prefetching");
}
