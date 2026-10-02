#ifndef GGML_SYCL_FATTN_VEC_HPP
#define GGML_SYCL_FATTN_VEC_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <iostream>
#include <iomanip>

#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "fattn-common.hpp"
#include <cmath>
#include <float.h>

namespace syclex = sycl::ext::oneapi::experimental;

static int ggml_sycl_fattn_vec_get_nthreads_device(gpu_arch arch) {
    // Xe2 (Battlemage, Lunar Lake) runs the flash-attention vec kernel best with a 256-thread work group.
    return (arch == gpu_arch::intel_gpu_bmg_g21 ||
            arch == gpu_arch::intel_gpu_bmg_g31 ||
            arch == gpu_arch::intel_gpu_lnl_m) ? 256 : 128;
}

// Currenlty llvm with the amdgcn target dose not support unrolling loops
// that contain a break that can not be resolved at compile time.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__

template <int D,
          int ncols,
          int type_K,
          int type_V,
          bool use_logit_softcap,
          int warp_size,
          int nthreads,
          bool gqa_order = false>  // D == head size
static void flash_attn_ext_vec(const char* __restrict__ Q,
                        const char* __restrict__ K,
                        const char* __restrict__ V,
                        const char* __restrict__ mask,
                        const char* __restrict__ sinks,
                        const int* __restrict__ KV_max,
                        float* __restrict__ dst,
                        sycl::float2* __restrict__ dst_meta,
                        const float scale,
                        const float max_bias,
                        const float m0,
                        const float m1,
                        const uint32_t n_head_log2,
                        const float logit_softcap,
                        const int32_t ne00,
                        const sycl::uint3 ne01,
                        const int32_t ne02,
                        const int32_t ne03,
                        const int32_t nb01,
                        const int32_t nb02,
                        const int32_t nb03,
                        const int32_t ne10,
                        const int32_t ne11,
                        const int32_t ne12,
                        const int32_t ne13,
                        const int32_t nb11,
                        const int32_t nb12,
                        const int64_t nb13,
                        const int32_t nb21,
                        const int32_t nb22,
                        const int64_t nb23,
                        const int32_t ne31,
                        const int32_t ne32,
                        const int32_t ne33,
                        const int32_t nb31,
                        const int32_t nb32,
                        const int64_t nb33) {

#ifdef SYCL_FLASH_ATTN
    // Skip unused kernel variants for faster compilation:

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int nthreads_KQ_q = (D/4 < warp_size ? D/4 : warp_size);
    constexpr int nthreads_V_q  = (D/4 < warp_size ? D/4 : warp_size);

    constexpr int nthreads_KQ = type_K == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_KQ_q;
    constexpr int nthreads_V  = type_V == GGML_TYPE_F16 ? 128 / cpy_nb : nthreads_V_q;

    static_assert(warp_size % nthreads_KQ == 0, "bad nthreads_K");
    static_assert(warp_size % nthreads_V  == 0, "bad nthreads_V");

    constexpr int V_rows_per_thread = type_V == GGML_TYPE_F16 ? 2*cpy_ne : 4;
    constexpr int V_cols_per_iter   = warp_size / nthreads_V;

    constexpr vec_dot_KQ_t vec_dot_KQ = get_vec_dot_KQ<type_K, D, nthreads_KQ, warp_size>();
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
#ifdef GGML_SYCL_F16
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, sycl::half, V_rows_per_thread>();
#else
    constexpr dequantize_V_t dequantize_V = get_dequantize_V<type_V, float, V_rows_per_thread>();
#endif // GGML_SYCL_F16

    const int ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    // Work-groups run roughly in launch order, parallel block (dim 1) fastest. With gqa_order the Q heads of
    // one K/V head come fastest instead, so the gqa_ratio work-groups reading the same K/V block run together
    // and all but the first find it in cache: decode then reads K/V from memory about once instead of once
    // per Q head.
    const int nblk = item_ct1.get_group_range(1);
    int       sequence, head, blk;
    if constexpr (gqa_order) {
        const int l   = item_ct1.get_group(0) * nblk + item_ct1.get_group(1);
        const int r   = l / gqa_ratio;
        const int kvh = r / nblk;
        blk      = r - kvh * nblk;
        sequence = kvh / ne12;
        head     = (kvh - sequence * ne12) * gqa_ratio + (l - r * gqa_ratio);
    } else {
        sequence = item_ct1.get_group(0) / ne02;
        head     = item_ct1.get_group(0) - sequence * ne02;
        blk      = item_ct1.get_group(1);
    }
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const sycl::half * maskh = (const sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    static_assert(D % (2*warp_size) == 0, "D not divisible by 2*warp_size == 64.");
    constexpr int nwarps = nthreads / warp_size;
    const int     tid    = warp_size * item_ct1.get_local_id(1) + item_ct1.get_local_id(2);
    __builtin_assume(tid < nthreads);

    constexpr int ne_KQ      = ncols*D;
    constexpr int ne_combine = nwarps*V_cols_per_iter*D;

    constexpr size_t lsm_size1 = ncols * warp_size;
    constexpr size_t lsm_size2 = ncols * warp_size;
#ifdef GGML_SYCL_F16
    sycl::half2 VKQ[ncols][(D / 2) / nthreads_V] = { { { 0.0f, 0.0f } } };
    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2)*sizeof(float) + lsm_size3*sizeof(sycl::half);

    syclex::work_group_static<char[local_share_mem_size]> lsm;

    float *KQ_max_shared = (float *)&lsm;
    float *KQ_sum_shared = KQ_max_shared+lsm_size1;
    sycl::half* KQ = (sycl::half*)(KQ_sum_shared + lsm_size2);


#else
    sycl::float2 VKQ[ncols][(D/2)/nthreads_V] = {{{0.0f, 0.0f}}};

    constexpr size_t lsm_size3 = (ne_KQ > ne_combine ? ne_KQ : ne_combine);
    constexpr size_t local_share_mem_size = (lsm_size1 + lsm_size2 + lsm_size3)*sizeof(float);


    syclex::work_group_static<char[local_share_mem_size]> lsm;
    float *KQ_max_shared = (float *)&lsm;
    float *KQ_sum_shared = KQ_max_shared+lsm_size1;
    float* KQ = KQ_sum_shared + lsm_size2;

#endif // GGML_SYCL_F16

    float KQ_max[ncols];
    float KQ_sum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX/2.0f;
        KQ_sum[j] = 0.0f;
    }

    // Convert Q to float2 (f16 K) or q8_1 (quantized K) and store in registers:
#ifdef GGML_SYCL_F16
    sycl::half2 Q_reg[ncols][(D / 2) / nthreads_KQ] = {{{0.0f, 0.0f}}};  // Will be initialized completely.
#else
    sycl::float2 Q_reg[ncols][(D/2)/nthreads_KQ] = {{{0.0f, 0.0f}}}; // May be only partially initialized.
#endif // GGML_SYCL_F16
    int    Q_i32[ncols][1 > D/(sizeof(int)*nthreads_KQ) ? 1 : D/(sizeof(int)*nthreads_KQ)];
    sycl::float2 Q_ds[ncols][1 > D / (sizeof(int) * nthreads_KQ) ? 1 : D / (sizeof(int) * nthreads_KQ)];
    if constexpr (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + item_ct1.get_local_id(1);

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 1 && ic0 + j >= int(ne01.z())) {
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += warp_size) {
                    const int i = i0 + item_ct1.get_local_id(2);

                    if (i0 + warp_size <= int(D/sizeof(int)) || i < int(D/sizeof(int))) {
                        tmp_q_i32[i] = 0;
                    }
                }
                if (item_ct1.get_local_id(2) < D/QK8_1) {
                    tmp_q_ds[item_ct1.get_local_id(2)] = sycl::float2(0.0f, 0.0f);
                }
            } else {
                const float * Q_f = (const float *) (Q + j*nb01);
                constexpr int nthreads_quantize = D/sizeof(int) < warp_size ? D/sizeof(int) : warp_size;
#pragma unroll
                for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_quantize) {
                    quantize_q8_1_to_shared<sycl::float2, nthreads_quantize, warp_size>
                        (Q_f + i0*sizeof(int), scale, tmp_q_i32 + i0, tmp_q_ds + i0/QI8_1);
                }
            }
        }


        item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i32 = (int    *) &KQ[j*D];
            sycl::float2 * tmp_q_ds  = (sycl::float2 *) (tmp_q_i32 + D / sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < int(D/sizeof(int)); i0 += nthreads_KQ) {
                const int i =
                    i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ);

                Q_i32[j][i0/nthreads_KQ] = tmp_q_i32[i];
                Q_ds[j][i0/nthreads_KQ]  = tmp_q_ds[i/QI8_1];
            }
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

    } else {
#ifdef GGML_SYCL_F16
        const sycl::half2 scale_h2 = sycl::half2(scale, scale);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const sycl::float2 * Q_j = (const sycl::float2 *) (Q + j * nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) :
                                                               item_ct1.get_local_id(2) % nthreads_KQ) *
                                       cpy_ne;

                sycl::float2 tmp[cpy_ne] = {
                    { 0.0f, 0.0f }
                };
                if (ncols == 1 || ic0 + j < int(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(tmp,            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(tmp + cpy_ne/2, &Q_j[i + cpy_ne/2]);
                }
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne; ++i1) {
                    Q_reg[j][i0 / nthreads_KQ + i1] = sycl::half2(tmp[i1].x(), tmp[i1].y());
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k] *= scale_h2;
            }
        }
#else
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const sycl::float2 * Q_j = (const sycl::float2 *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += nthreads_KQ*cpy_ne) {
                const int i = i0 + (nthreads_KQ == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_KQ)*cpy_ne;
                if (ncols == 1 || ic0 + j < int(ne01.z())) {
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ],            &Q_j[i]);
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_reg[j][i0/nthreads_KQ + cpy_ne/2], &Q_j[i + cpy_ne/2]);
                }
            }
#pragma unroll
            for (int k = 0; k < (D/2)/nthreads_KQ; ++k) {
                Q_reg[j][k].x() *= scale;
                Q_reg[j][k].y() *= scale;
            }
        }
#endif // GGML_SYCL_F16
    }

    const int k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    K += blk * nthreads * nb11;
    V += blk * nthreads * nb21;
    maskh += blk * nthreads;
    for (int k_VKQ_0 = blk * nthreads; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += nblk * nthreads,
             // Increment pointers after each loop:
         K += nblk * nthreads * nb11, V += nblk * nthreads * nb21,
             maskh += nblk * nthreads) {
        // Calculate KQ tile and keep track of new maximum KQ values:
        float KQ_reg[ncols]={}; // KQ in registers.
        float KQ_max_new[ncols]={};


#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nthreads_KQ; ++i_KQ_0) {
            const int i_KQ = item_ct1.get_local_id(1) * warp_size +
                             (nthreads_KQ == warp_size ? 0 : (item_ct1.get_local_id(2) & ~(nthreads_KQ - 1))) + i_KQ_0;

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K + i_KQ*nb11, Q_reg[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum<nthreads_KQ>(sum);

                if (use_logit_softcap) {
                    sum = logit_softcap * sycl::tanh(sum);
                }
                if (mask) {
                    sum += slope * sycl::vec<sycl::half, 1>(maskh[j * ne11 + i_KQ])
                                       .convert<float, sycl::rounding_mode::automatic>()[0];
                }

                KQ_max_new[j] = sycl::fmax((float) KQ_max_new[j], sum);

                if (int(nthreads_KQ == warp_size ? item_ct1.get_local_id(2)
                                                 : item_ct1.get_local_id(2) %
                                                       nthreads_KQ) == i_KQ_0) {
                  KQ_reg[j] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int offset = nthreads_KQ; offset < warp_size; offset <<= 1) {
               KQ_max_new[j] = sycl::fmax(
                  (float)KQ_max_new[j],
                  (float)dpct::permute_sub_group_by_xor(
                      sycl::ext::oneapi::this_work_item::get_sub_group(),
                      KQ_max_new[j],
                      offset,
                      warp_size));
            }
            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[j] - KQ_max_new[j]));
            KQ_max[j] = KQ_max_new[j];

            KQ_reg[j]            = sycl::native::exp((float) (KQ_reg[j] - KQ_max[j]));
            KQ_sum[j] = KQ_sum[j]*KQ_max_scale + KQ_reg[j];
            KQ[j*nthreads + tid] = KQ_reg[j];

#ifdef GGML_SYCL_F16
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }

        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());

#pragma unroll
        for (int k0 = 0; k0 < warp_size; k0 += V_cols_per_iter) {
            const int k = item_ct1.get_local_id(1) * warp_size + k0 +
                          (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V);

#ifdef GGML_SYCL_F16
            sycl::half2 KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = sycl::half2(KQ[j * nthreads + k]);
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::half2 tmp[V_rows_per_thread / 2];
                dequantize_V(V + k * nb21, tmp,
                             2 * i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) :
                                                                      item_ct1.get_local_id(2) % nthreads_V) *
                                               V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1] += tmp[i_VKQ_1]*KQ_k[j];
                    }
                }
            }
#else
            float KQ_k[ncols];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                KQ_k[j] = KQ[j*nthreads + k];
            }
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
                sycl::float2 tmp[V_rows_per_thread/2];
                dequantize_V(V + k*nb21, tmp,
                    2*i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*V_rows_per_thread);
#pragma unroll
                for (int i_VKQ_1 = 0; i_VKQ_1 < V_rows_per_thread/2; ++i_VKQ_1) {
#pragma unroll
                    for (int j = 0; j < ncols; ++j) {
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].x() += tmp[i_VKQ_1].x()*KQ_k[j];
                        VKQ[j][i_VKQ_0/nthreads_V + i_VKQ_1].y() += tmp[i_VKQ_1].y()*KQ_k[j];
                    }
                }
            }
#endif // GGML_SYCL_F16
        }
    }

    if (sinks && blk == 0) {
        const float sink = ((const float *) sinks)[head];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + item_ct1.get_local_id(1);

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }
            const float kqmax_new_j  = sycl::fmax(sink, (float) KQ_max[j]);
            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[j] - kqmax_new_j));
            KQ_max[j] = kqmax_new_j;

            KQ_sum[j] = KQ_sum[j] * KQ_max_scale +
                        (item_ct1.get_local_id(2) == 0 ? sycl::native::exp((float) (sink - KQ_max[j])) : 0.0f);
#ifdef GGML_SYCL_F16
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
                VKQ[j][i_VKQ_0/nthreads_V].x() *= KQ_max_scale;
                VKQ[j][i_VKQ_0/nthreads_V].y() *= KQ_max_scale;
            }
#endif // GGML_SYCL_F16
        }
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (item_ct1.get_local_id(1) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(2)] = -FLT_MAX / 2.0f;
            KQ_sum_shared[j*warp_size+item_ct1.get_local_id(2)] = 0.0f;
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (item_ct1.get_local_id(2) == 0) {
            KQ_max_shared[j*warp_size+item_ct1.get_local_id(1)] = KQ_max[j];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 1 && ic0 + j_VKQ >= int(ne01.z())) {
            break;
        }

        float kqmax_new         = KQ_max_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
        kqmax_new = warp_reduce_max<warp_size>(kqmax_new);
        const float kqmax_scale = sycl::native::exp((float) (KQ_max[j_VKQ] - kqmax_new));
        KQ_max[j_VKQ] = kqmax_new;

#ifdef GGML_SYCL_F16
        sycl::half2 * VKQ_tmp = (sycl::half2 *) KQ + item_ct1.get_local_id(1) * (V_cols_per_iter * D / 2) +
                                (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V) * (D / 2);

        const sycl::half2 kqmax_scale_h2 = sycl::half2(kqmax_scale, kqmax_scale);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V] *= kqmax_scale_h2;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ =
                i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V) *
                              (V_rows_per_thread / 2);

            ggml_sycl_memcpy_1<V_rows_per_thread * sizeof(sycl::half)>(VKQ_tmp + i_VKQ,
                                                                       &VKQ[j_VKQ][i_VKQ_0 / nthreads_V]);
        }
#else
        sycl::float2 * VKQ_tmp = (sycl::float2 *) KQ + item_ct1.get_local_id(1)*(V_cols_per_iter*D/2)
            + (nthreads_V == warp_size ? 0 : item_ct1.get_local_id(2) / nthreads_V)*(D/2);
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V) {
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].x() *= kqmax_scale;
            VKQ[j_VKQ][i_VKQ_0/nthreads_V].y() *= kqmax_scale;
        }
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D/2; i_VKQ_0 += nthreads_V*V_rows_per_thread/2) {
            const int i_VKQ = i_VKQ_0 + (nthreads_V == warp_size ? item_ct1.get_local_id(2) : item_ct1.get_local_id(2) % nthreads_V)*(V_rows_per_thread/2);

            ggml_sycl_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ,                       &VKQ[j_VKQ][i_VKQ_0/nthreads_V]);
            ggml_sycl_memcpy_1<V_rows_per_thread/2*sizeof(float)>(VKQ_tmp + i_VKQ + V_rows_per_thread/4, &VKQ[j_VKQ][i_VKQ_0/nthreads_V + V_rows_per_thread/4]);
        }
#endif // GGML_SYCL_F16

        KQ_sum[j_VKQ] *= kqmax_scale;
        KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);
        if (item_ct1.get_local_id(2) == 0) {
            KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(1)] = KQ_sum[j_VKQ];
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);


        if (nthreads <= D || tid < D) {
            KQ_sum[j_VKQ] = KQ_sum_shared[j_VKQ*warp_size+item_ct1.get_local_id(2)];
            KQ_sum[j_VKQ] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ]);

#pragma unroll
            for (int i0 = 0; i0 < D; i0 += nthreads) {
                float dst_val = 0;
#pragma unroll
                for (int w = 0; w < nwarps; ++w) {
#pragma unroll
                    for (int v = 0; v < V_cols_per_iter; ++v) {
                        dst_val += float(KQ[w*V_cols_per_iter*D + v*D + i0 + tid]);
                    }
                }
                if (nblk == 1) {
                    dst_val /= KQ_sum[j_VKQ];
                }
                dst[(((sequence * int(ne01.z()) + ic0 + j_VKQ) * ne02 + head) * nblk +
                     blk) *
                        D +
                    i0 + tid] = dst_val;
            }
        }

        if (j_VKQ < ncols-1) {
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

    }

    if (nblk != 1 && tid < ncols && (ncols == 1 || ic0 + tid < int(ne01.z()))) {
        dst_meta[((sequence * int(ne01.z()) + ic0 + tid) * ne02 + head) * nblk +
                 blk] = make_float2(KQ_max[tid], KQ_sum[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);

#endif // SYCL_FLASH_ATTN
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__



// Values 64*p .. 64*p + 63 of a Q8_0 or Q4_0 K or V row (the blocks 2p and 2p+1) into dst as half. Two Q8_0
// (68 B) or Q4_0 (36 B) blocks are 4-byte aligned where one is only 2-byte aligned, so they are read as 32-bit
// words, 17 or 9 loads instead of a byte load per value. Rows must be 4-byte aligned, as with head sizes 128 and
// 256. Each 8 values are stored as soon as they are converted: holding all 64 spilled registers on Meteor Lake.
template <int type>
static __dpct_inline__ void fattn_gqa_dequant64(const char * __restrict__ row, const int p,
                                                sycl::vec<sycl::half, 8> * __restrict__ dst) {
    static_assert(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0, "unsupported K/V type");
    constexpr int bs = type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_0);
    constexpr int nw = 2 * bs / 4;
    static_assert(2 * bs % 4 == 0, "two blocks are whole 32-bit words");

    const uint32_t * w = (const uint32_t *) (row + 2 * bs * p);
    uint32_t         u[nw];
#pragma unroll
    for (int i = 0; i < nw; ++i) {
        u[i] = w[i];
    }
    // byte j of the two blocks; j is a constant after unrolling
    auto byte = [&](const int j) -> uint32_t { return (u[j / 4] >> (8 * (j % 4))) & 0xFF; };

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int   o = h * bs;
        const float d = sycl::bit_cast<sycl::half>((uint16_t) (byte(o) | (byte(o + 1) << 8)));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            sycl::vec<sycl::half, 8> v;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int i = 8 * c + e;
                int       q;
                if constexpr (type == GGML_TYPE_Q8_0) {
                    q = (int8_t) byte(o + 2 + i);
                } else {
                    q = (i < 16 ? byte(o + 2 + i) & 0xF : byte(o + 2 + i - 16) >> 4) - 8;
                }
                v[e] = sycl::half(d * q);
            }
            dst[4 * h + c] = v;
        }
    }
}

// Rows k0 .. k0 + T - 1 of K or V (row stride nb) into local memory as half, zero past k_max
template <int type, int D, int T>
static __dpct_inline__ void fattn_gqa_fill(const char * __restrict__ src, const int nb, const int k0, const int k_max,
                                           const int tid, const int nthreads, sycl::vec<sycl::half, 8> * dst) {
    constexpr int C8 = D / 8;
    if constexpr (type == GGML_TYPE_F16) {
        // a task per 8 values, consecutive lanes on consecutive 16 B of a row
        for (int idx = tid; idx < T * C8; idx += nthreads) {
            const int t = idx / C8;
            const int c = idx - t * C8;
            // rows past the end get weight 0; keep them finite
            dst[idx] = k0 + t < k_max ? ((const sycl::vec<sycl::half, 8> *) (src + (int64_t) (k0 + t) * nb))[c] :
                                        sycl::vec<sycl::half, 8>(sycl::half(0.0f));
        }
    } else {
        // a task per 64 values (two blocks) of a row
        constexpr int P = D / 64;
        for (int idx = tid; idx < T * P; idx += nthreads) {
            const int t = idx / P;
            const int p = idx - t * P;
            if (k0 + t < k_max) {
                fattn_gqa_dequant64<type>(src + (int64_t) (k0 + t) * nb, p, dst + t * C8 + 8 * p);
            } else {
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    dst[t * C8 + 8 * p + c] = sycl::vec<sycl::half, 8>(sycl::half(0.0f));
                }
            }
        }
    }
}

// Tokens per work-group of flash_attn_ext_vec_gqa: up to 32 sub-groups (512 work-items)
static constexpr int fattn_vec_gqa_ncols(const int64_t ne01, const int64_t gqa_ratio) {
    int ncols = ne01 == 1 ? 1 : ne01 == 2 ? 2 : ne01 <= 4 ? 4 : 8;
    while (ncols > 1 && gqa_ratio * ncols > 32) {
        ncols /= 2;
    }
    return ncols;
}

// Whether flash_attn_ext_vec_gqa takes this op: head size 128 or 256, F16, Q8_0 or Q4_0 K and V, GQA, up to
// 8 query tokens (decode and speculative verification), no alibi, sinks or logit softcap.
inline bool ggml_sycl_fattn_vec_gqa_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    float max_bias;
    float logit_softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    auto type_ok = [](const ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
    };
    if ((K->ne[0] != 128 && K->ne[0] != 256) || V->ne[0] != K->ne[0] || !type_ok(K->type) || !type_ok(V->type)) {
        return false;
    }
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio < 2 || gqa_ratio > 16 || Q->ne[1] > 8 || max_bias != 0.0f || logit_softcap != 0.0f ||
        dst->src[4]) {
        return false;
    }
    const int64_t nsg = gqa_ratio * fattn_vec_gqa_ncols(Q->ne[1], gqa_ratio);
    return nsg * WARP_16_SIZE <= ggml_sycl_info().max_work_group_sizes[ggml_sycl_get_device()];
}

// Flash attention for GQA and up to 8 query tokens: a work-group per K/V head, parallel block and ncols tokens,
// with a sub-group for each of its gqa_ratio Q heads and tokens. Each tile of K/V is loaded (and dequantized)
// into local memory once for all of them, where a work-group per Q head and token would read it from memory
// gqa_ratio * ncols times. A lane holds D/16 consecutive values of its Q row and output: more tokens per
// sub-group spill registers on Meteor Lake. No alibi, sinks or logit softcap.
template <int D, int type_K, int type_V, int ncols>
static void flash_attn_ext_vec_gqa(const char * __restrict__ Q, const char * __restrict__ K,
                                   const char * __restrict__ V, const char * __restrict__ mask,
                                   const char * __restrict__ sinks, const int * __restrict__ KV_max,
                                   float * __restrict__ dst, sycl::float2 * __restrict__ dst_meta,
                                   const float scale, const float max_bias, const float m0, const float m1,
                                   const uint32_t n_head_log2, const float logit_softcap, const int32_t ne00,
                                   const sycl::uint3 ne01, const int32_t ne02, const int32_t ne03,
                                   const int32_t nb01, const int32_t nb02, const int32_t nb03, const int32_t ne10,
                                   const int32_t ne11, const int32_t ne12, const int32_t ne13, const int32_t nb11,
                                   const int32_t nb12, const int64_t nb13, const int32_t nb21, const int32_t nb22,
                                   const int64_t nb23, const int32_t ne31, const int32_t ne32, const int32_t ne33,
                                   const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#ifdef SYCL_FLASH_ATTN
    constexpr int warp_size = WARP_16_SIZE;
    constexpr int T         = 8;              // K/V rows per tile
    constexpr int EPL       = D / warp_size;  // values of a row per lane
    constexpr int C8        = D / 8;          // 8-value chunks per row
    static_assert(EPL % 8 == 0 && D % 64 == 0, "a lane takes whole 8-value chunks");

    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto      sg       = item_ct1.get_sub_group();
    const int lane     = item_ct1.get_local_id(2);
    const int g        = item_ct1.get_local_id(1);
    const int nthreads = item_ct1.get_local_range(1) * warp_size;
    const int tid      = g * warp_size + lane;

    const int blk       = item_ct1.get_group(1);
    const int nblk      = item_ct1.get_group_range(1);
    const int gqa_ratio = ne02 / ne12;
    const int sequence  = item_ct1.get_group(0) / ne12;
    const int kvh       = item_ct1.get_group(0) - sequence * ne12;
    const int head      = kvh * gqa_ratio + g % gqa_ratio;
    const int n_tokens  = ne01.z();
    const int token     = item_ct1.get_group(2) * ncols + g / gqa_ratio;
    // a token past the end computes on the last one and is not written
    const int token_in  = sycl::min(token, n_tokens - 1);

    Q += nb03 * sequence + nb02 * head + nb01 * token_in;
    K += nb13 * sequence + nb12 * kvh;
    V += nb23 * sequence + nb22 * kvh;
    const sycl::half * maskh =
        mask ? (const sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * token_in) : nullptr;

    syclex::work_group_static<sycl::vec<sycl::half, 8>[2 * T * C8]> lsm;
    sycl::vec<sycl::half, 8> * Ks = (sycl::vec<sycl::half, 8> *) &lsm;
    sycl::vec<sycl::half, 8> * Vs = Ks + T * C8;

    float q[EPL];
    float acc[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        q[i]   = ((const float *) Q)[EPL * lane + i] * scale;
        acc[i] = 0.0f;
    }
    float kq_max = -FLT_MAX / 2.0f;
    float kq_sum = 0.0f;

    const int k_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    for (int k0 = blk * T; k0 < k_max; k0 += nblk * T) {
        fattn_gqa_fill<type_K, D, T>(K, nb11, k0, k_max, tid, nthreads, Ks);
        fattn_gqa_fill<type_V, D, T>(V, nb21, k0, k_max, tid, nthreads, Vs);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        float sc[T];
        float m_new = kq_max;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float dot = 0.0f;
#pragma unroll
            for (int c = 0; c < EPL / 8; ++c) {
                const sycl::vec<sycl::half, 8> kv = Ks[t * C8 + lane * (EPL / 8) + c];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dot += q[8 * c + i] * float(kv[i]);
                }
            }
            dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
            if (maskh) {
                dot += float(maskh[k0 + t]);
            }
            sc[t] = k0 + t < k_max ? dot : -INFINITY;
            m_new = sycl::fmax(m_new, sc[t]);
        }

        const float corr = sycl::native::exp(kq_max - m_new);
        kq_max = m_new;
        kq_sum *= corr;
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            acc[i] *= corr;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float p = sycl::native::exp(sc[t] - kq_max);
            kq_sum += p;
#pragma unroll
            for (int c = 0; c < EPL / 8; ++c) {
                const sycl::vec<sycl::half, 8> vv = Vs[t * C8 + lane * (EPL / 8) + c];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[8 * c + i] += p * float(vv[i]);
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if (token < n_tokens) {
        const int j_dst = (sequence * n_tokens + token) * ne02 + head;
        if (nblk == 1) {
#pragma unroll
            for (int i = 0; i < EPL; ++i) {
                dst[j_dst * D + EPL * lane + i] = acc[i] / kq_sum;
            }
        } else {
#pragma unroll
            for (int i = 0; i < EPL; ++i) {
                dst[(j_dst * nblk + blk) * D + EPL * lane + i] = acc[i];
            }
            if (lane == 0) {
                dst_meta[j_dst * nblk + blk] = make_float2(kq_max, kq_sum);
            }
        }
    }
    GGML_UNUSED_VARS(sinks, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne13, ne31, ne32,
                     nb32);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2,
                     logit_softcap, ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12,
                     nb13, nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
#endif  // SYCL_FLASH_ATTN
}

template <int D, int cols_per_block, int type_K, int type_V, bool use_logit_softcap, bool gqa_order = false>
void ggml_sycl_flash_attn_ext_vec_case_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    constexpr int warp_size = WARP_16_SIZE; //better performance than WARP_32_SIZE

    const bool need_f16_K = type_K == GGML_TYPE_F16;
    const bool need_f16_V = type_V == GGML_TYPE_F16;
    constexpr size_t nbytes_shared = 0;

    const auto arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    const int nthreads = ggml_sycl_fattn_vec_get_nthreads_device(arch);
    if constexpr (D <= 256) {
        if (nthreads == 256) {
            constexpr int nthreads_hw = 256;
            constexpr int nwarps = nthreads_hw / warp_size;
            launch_fattn<D, cols_per_block, 1,
                         flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                            use_logit_softcap, warp_size, nthreads_hw, gqa_order>, warp_size>(
                ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
            return;
        }
    }

    constexpr int nthreads_hw = 128;
    constexpr int nwarps = nthreads_hw / warp_size;
    launch_fattn<D, cols_per_block, 1,
                 flash_attn_ext_vec<D, cols_per_block, type_K, type_V,
                                    use_logit_softcap, warp_size, nthreads_hw, gqa_order>, warp_size>(
        ctx, dst, nwarps, nbytes_shared, D, need_f16_K, need_f16_V, false);
}

template <int D, int type_K, int type_V>
void ggml_sycl_flash_attn_ext_vec_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // GQA with F16, Q8_0 or Q4_0 K/V and up to 8 tokens: a work-group per K/V head, see flash_attn_ext_vec_gqa
    if constexpr ((D == 128 || D == 256) &&
                  (type_K == GGML_TYPE_F16 || type_K == GGML_TYPE_Q8_0 || type_K == GGML_TYPE_Q4_0) &&
                  (type_V == GGML_TYPE_F16 || type_V == GGML_TYPE_Q8_0 || type_V == GGML_TYPE_Q4_0)) {
        if (ggml_sycl_fattn_vec_gqa_supported(dst)) {
            const int gqa_ratio = Q->ne[2] / dst->src[1]->ne[2];
            // At most ne11/nbatch parallel blocks, the work-groups beyond one per K/V head: more for short KV,
            // fewer partial results to combine for long KV. On Meteor Lake 32 took 54 instead of 140 us at 256 KV
            // and 128 153 instead of 167 us at 1024 KV.
            const int nbatch = dst->src[1]->ne[1] <= 512 ? 32 : 128;
            // ncols2 16 so that launch_fattn makes one work-group per K/V head (gqa_ratio <= 16)
            switch (fattn_vec_gqa_ncols(Q->ne[1], gqa_ratio)) {
                case 1:
                    launch_fattn<D, 1, 16, flash_attn_ext_vec_gqa<D, type_K, type_V, 1>, WARP_16_SIZE>(
                        ctx, dst, gqa_ratio, 0, nbatch, false, false, false);
                    break;
                case 2:
                    launch_fattn<D, 2, 16, flash_attn_ext_vec_gqa<D, type_K, type_V, 2>, WARP_16_SIZE>(
                        ctx, dst, gqa_ratio * 2, 0, nbatch, false, false, false);
                    break;
                case 4:
                    launch_fattn<D, 4, 16, flash_attn_ext_vec_gqa<D, type_K, type_V, 4>, WARP_16_SIZE>(
                        ctx, dst, gqa_ratio * 4, 0, nbatch, false, false, false);
                    break;
                default:
                    launch_fattn<D, 8, 16, flash_attn_ext_vec_gqa<D, type_K, type_V, 8>, WARP_16_SIZE>(
                        ctx, dst, gqa_ratio * 8, 0, nbatch, false, false, false);
                    break;
            }
            return;
        }
    }

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block = 1;
        const ggml_tensor * K         = dst->src[1];
        const int           gqa_ratio = Q->ne[2] / K->ne[2];

        // other GQA decode with F16 or Q8_0 K/V (other types keep one kernel variant each)
        if constexpr ((type_K == GGML_TYPE_F16 && type_V == GGML_TYPE_F16) ||
                      (type_K == GGML_TYPE_Q8_0 && type_V == GGML_TYPE_Q8_0)) {
            if (gqa_ratio > 1 && logit_softcap == 0.0f) {
                ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, false, true>(ctx, dst);
                return;
            }
        }
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 2;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_sycl_flash_attn_ext_vec_case_impl<D, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_CASE(D, type_K, type_V)                              \
    template void ggml_sycl_flash_attn_ext_vec_case                         \
    <D, type_K, type_V>(ggml_backend_sycl_context & ctx, ggml_tensor * dst) \

#define EXTERN_DECL_FATTN_VEC_CASES(D, type_K)             \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_F16);  \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q4_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_0); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q5_1); \
    extern DECL_FATTN_VEC_CASE(D, type_K, GGML_TYPE_Q8_0); \

EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES( 64, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(128, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(256, GGML_TYPE_Q8_0)

EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_F16)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q4_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_0)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q5_1)
EXTERN_DECL_FATTN_VEC_CASES(512, GGML_TYPE_Q8_0)

#endif // GGML_SYCL_FATTN_VEC_HPP
