/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  quantize.hpp
 *
 *  Description:
 *     Sycl backend specific quantization functions
 **************************************************************************/

#pragma once

#include <sycl/nd_item.hpp>

#include "ggml-sycl/dpct/helper.hpp"

template <int ElementsPerWI>
__dpct_inline__ static void quantize_q8_1_impl(const float * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_values, float & d,
                                               float & sum, const sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<float, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(x + float_ptr_offset);

    float amax = 0.0f;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_values[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<float>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<float>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    for (int i = 0; i < ElementsPerWI; i++) {
        quantized_values[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <int ElementsPerWI> struct no_quantize_q8_1 {
    void operator()(const float *, void *, int, int, const sycl::nd_item<1> &) const {}
};

template <int ElementsPerWI> struct quantize_and_reorder_q8_1_soa {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * reordered_q8_tensor, const int kx,
                                    const int kx_padded, const sycl::nd_item<1> & it) const {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        auto      col                = subgroup_id % num_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * sizeof(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((char *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_values;

        auto ds_ptr = (sycl::half2 *) ((char *) reordered_q8_tensor + row_offset + kx + col * sizeof(sycl::half2));
        if (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <int ElementsPerWI> struct quantize_q8_1 {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * q8_tensor, const int kx, const int kx_padded,
                                    const sycl::nd_item<1> & it) const {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        const int num_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / num_blocks_per_row;
        const int pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_values;
        float                            d   = 0.0f;
        float                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_values, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % num_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_values;
        if (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

// Slot m and half h of int iu of q8_1 block ib within a PTQ1_0 block, in the order vec_dot_ptq1_0_q8_1 uses
// them: m = 5*k + t for qs word 2h+k (k < 2) or 4+h (k = 2) at trit t, and m = 15 for qh.
static __dpct_inline__ void ptq1_0_act_slot(const int ib, const int iu, int & m, int & h) {
    if (ib < 2 || (ib == 2 && iu < 4)) {
        h = (iu >> 1) & 1;
        m = 5 * (iu & 1) + 2 * ib + (iu >> 2);
    } else if (ib == 3 && iu >= 6) {
        h = iu & 1;
        m = 15;
    } else {
        h = iu & 1;
        m = 10 + ((8 * ib + iu - 20) >> 1);
    }
}

// q8_1 laid out for mul_mat_vec_ptq1_0_q8_1_act: per 1024 elements (8 PTQ1_0 blocks), 16 slots of 16 ints
// (int of lane 2*b + h at slot*16 + lane) and then one uint per lane: the q8 scale of block b as half in the
// low 16 bits and the sum of the lane's 64 values as int16 in the high 16 bits. One scale per 128 values
// (finer than the 256 of q8_K), so a sub-group quantizes a whole PTQ1_0 block. 1152 B with the last 64
// unused, the same as the 32 plain q8_1 blocks it replaces. Needs kx % 1024 == 0.
template <int ElementsPerWI> struct quantize_q8_1_ptq1_0 {
    __dpct_inline__ void operator()(const float * __restrict__ x, void * q8_tensor, const int kx, const int kx_padded,
                                    const sycl::nd_item<1> & it) const {
        static_assert(ElementsPerWI * WARP_SIZE == QK_PTQ1_0 && ElementsPerWI % 4 == 0,
                      "a sub-group takes one PTQ1_0 block and a work-item whole ints");

        const auto sg   = it.get_sub_group();
        const int  lane = it.get_local_id(0);

        const int blocks_per_row = kx / QK_PTQ1_0;
        const int row            = it.get_group(0) / blocks_per_row;
        const int col            = it.get_group(0) % blocks_per_row;
        const int b              = col % 8;

        const sycl::vec<float, ElementsPerWI> xv = *reinterpret_cast<const sycl::vec<float, ElementsPerWI> *>(
            x + (size_t) it.get_group(0) * QK_PTQ1_0 + ElementsPerWI * lane);

        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < ElementsPerWI; ++i) {
            amax = sycl::fmax(amax, sycl::fabs(xv[i]));
        }
        amax          = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
        const float d = amax == 0 ? 1 : amax / 127;

        char * group = (char *) q8_tensor + (size_t) row * (kx_padded / QK8_1) * sizeof(block_q8_1) +
                       (col / 8) * 32 * sizeof(block_q8_1);

        int sum0 = 0;
        int sum1 = 0;
#pragma unroll
        for (int j = 0; j < ElementsPerWI / 4; ++j) {
            const int e = ElementsPerWI * lane + 4 * j;
            int       m, h;
            ptq1_0_act_slot(e / QK8_1, (e % QK8_1) / 4, m, h);

            sycl::vec<int8_t, 4> q;
            int                  s = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q[i] = sycl::round(xv[4 * j + i] / d);
                s += q[i];
            }
            sum0 += h == 0 ? s : 0;
            sum1 += h == 0 ? 0 : s;
            *reinterpret_cast<sycl::vec<int8_t, 4> *>(group + (m * 16 + 2 * b + h) * 4) = q;
        }
        sum0 = sycl::reduce_over_group(sg, sum0, sycl::plus<int>());
        sum1 = sycl::reduce_over_group(sg, sum1, sycl::plus<int>());

        if (lane < 2) {
            const uint32_t dh = sycl::bit_cast<uint16_t>(sycl::half(amax == 0 ? 0 : d));
            ((uint32_t *) (group + 1024))[2 * b + lane] = dh | ((uint32_t) (lane == 0 ? sum0 : sum1) << 16);
        }
    }
};

template <template <int> typename quantize_f>
void quantize_row_q8_1_sycl(const float * x, void * vy, const int kx, const int ky, const int kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto num_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = num_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    if constexpr (std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>, quantize_q8_1_ptq1_0<QK8_1 / WARP_SIZE>>) {
        // a sub-group per PTQ1_0 block
        stream->parallel_for(sycl::nd_range<1>({ global_range / 4 }, { local_range }),
                             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 quantize_f<QK_PTQ1_0 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                             });
    } else {
        stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                             [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                             });
    }
}
