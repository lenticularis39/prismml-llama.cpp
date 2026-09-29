#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_sycl_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// input state read straight from the recurrent cache: seq s uses row rows[s] of states
// (see ggml_sycl_try_gdn_state_gather_fusion)
struct ggml_sycl_gated_delta_net_state_rows {
    const float *   states;
    const int32_t * rows;
    int64_t         row_size; // in elements
};

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                  const ggml_sycl_gated_delta_net_state_rows * rows = nullptr);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache,
                                              const ggml_sycl_gated_delta_net_state_rows * rows = nullptr);
