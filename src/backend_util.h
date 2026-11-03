#pragma once

#include <ggml-backend.h>
#include <mutex>

namespace qwen3_asr {

// The portable release builds load the ggml backends as modules (GGML_BACKEND_DL), with one CPU
// backend per instruction set, and ggml picks the best one for this CPU at run time. Modules only
// register once loaded, and the CPU backend is then not linked in, so CPU-specific calls go through
// the backend registry. With the backends linked in (the default build) loading finds nothing new.
inline void load_backends() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}

inline void set_backend_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto fn = reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
    if (fn) {
        fn(backend, n_threads);
    }
}

inline ggml_backend_buffer_t cpu_buffer_from_ptr(void * ptr, size_t size, size_t max_tensor_size) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return cpu_dev ? ggml_backend_dev_buffer_from_host_ptr(cpu_dev, ptr, size, max_tensor_size) : nullptr;
}

} // namespace qwen3_asr
