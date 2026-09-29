#include "fwht.hpp"

#include <cmath>

template <int N, typename T>
static void fwht_kernel(const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = (float) src[i * WARP_SIZE + lane] * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N, typename T>
static void launch_fwht(const T * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N, T>(src, dst, n_rows, scale, item);
                         });
}

// Wide rows: one work-group per row. The sub-group kernel above would need N/WARP_SIZE
// registers per lane and puts a whole row on one sub-group, so rows from 1024 up used to
// fall back to a oneMKL GEMM with the dense N x N matrix.
#define FWHT_BLOCK_THREADS 256

template <int N, int NT, typename T>
static void fwht_block_kernel(const T * __restrict__ src, float * __restrict__ dst, const float scale,
                              float * __restrict__ s, const sycl::nd_item<1> & item) {
    constexpr int NE = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % WARP_SIZE == 0, "bad FWHT block shape");

    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r   = item.get_group(0);
    const int     tid = item.get_local_id(0);

    src += r * N;
    dst += r * N;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = (float) src[i * NT + tid] * scale;
    }

    // stages inside a sub-group: the partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across sub-groups: the partner differs in the work-item bits above the lane
#pragma unroll
    for (int h = WARP_SIZE; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            s[j * NT + tid] = reg[j];
        }
        item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];

            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

    // stages above the work-group width: the partner is another register of the same work-item
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <int N, typename T>
static void launch_fwht_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                              dpct::queue_ptr stream) {
    constexpr int NT = FWHT_BLOCK_THREADS;

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> s(sycl::range<1>(N), cgh);

        cgh.parallel_for(sycl::nd_range<1>(n_rows * NT, NT),
                         [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_block_kernel<N, NT, T>(src, dst, scale,
                                                      s.get_multi_ptr<sycl::access::decorated::no>().get(), item);
                         });
    });
}

template <typename T>
static bool fwht_dispatch(const T * src_d, float * dst_d, const int n, const int64_t rows, dpct::queue_ptr stream) {
    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1024:
            launch_fwht_block<1024>(src_d, dst_d, rows, scale, stream);
            return true;
        case 2048:
            launch_fwht_block<2048>(src_d, dst_d, rows, scale, stream);
            return true;
        case 4096:
            launch_fwht_block<4096>(src_d, dst_d, rows, scale, stream);
            return true;
        case 8192:
            launch_fwht_block<8192>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(src, dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int       n      = (int) src->ne[0];
    const int64_t   rows   = ggml_nrows(src);
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    // F16 input is widened on load; the transform itself always runs in F32
    if (src->type == GGML_TYPE_F16) {
        return fwht_dispatch((const sycl::half *) src->data, dst_d, n, rows, stream);
    }
    return fwht_dispatch((const float *) src->data, dst_d, n, rows, stream);
}
