#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <xmmintrin.h> // For _mm_prefetch
#include <cstdlib>
#include <cstring>

namespace py = pybind11;

// ReLU kernel with prefetching on X only
void process_batch(
    py::array_t<float, py::array::c_style | py::array::forcecast> X,
    py::array_t<int, py::array::c_style | py::array::forcecast> y  // or float if needed
) {
    auto X_buf = X.request();
    auto y_buf = y.request();

    if (X_buf.ndim != 2 || y_buf.ndim != 1)
        throw std::runtime_error("Invalid input shapes");

    const ssize_t batch_size = X_buf.shape[0];
    const ssize_t feature_size = X_buf.shape[1];

    if (y_buf.shape[0] != batch_size)
        throw std::runtime_error("Mismatched batch size between X and y");

    float* X_data = static_cast<float*>(X_buf.ptr);
    int* y_data = static_cast<int*>(y_buf.ptr);  // use float* if y is float

    for (ssize_t i = 0; i < batch_size; ++i) {
        // Prefetch next X sample into L1 cache
        if (i + 1 < batch_size) {
            _mm_prefetch(reinterpret_cast<const char*>(&X_data[(i + 1) * feature_size]), _MM_HINT_T0);
        }

        // Process current X sample
        for (ssize_t j = 0; j < feature_size; ++j) {
            float& x = X_data[i * feature_size + j];
            x = x > 0.0f ? x : 0.0f;
        }

        // Access y[i] if needed (e.g., print or pass along)
        int label = y_data[i];
        // No prefetching for y — it's small and sequential
    }
}

PYBIND11_MODULE(prefetch, m) {
    m.def("process_batch", &process_batch, "Process (X, y) batch with prefetching on X only");
}
