/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int Stages,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopIntelPVC<Stages>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopIntelPVC<Stages>;
  using WorkgroupTileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(WorkgroupTileShape{});
  static constexpr auto BLK_N = get<1>(WorkgroupTileShape{});
  static constexpr auto BLK_K = get<2>(WorkgroupTileShape{});
  
  static constexpr auto ATOM_M = get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N = get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K = get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);
  using SubgroupTileShape = Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;

  static constexpr size_t cacheline_bytes = 64;
  static constexpr auto block_size_w_a = cute::min(SG_K, cacheline_bytes / sizeof(ElementA));
  static constexpr auto block_size_w_b = cute::min(SG_N, cacheline_bytes / sizeof(ElementB));
  static constexpr auto nums_block_w_a = ceil_div(SG_K, block_size_w_a);
  static constexpr auto nums_block_w_b = ceil_div(SG_N, block_size_w_b);
  using PrefetchAThrShape = Shape<Int<ATOM_N /cute::gcd(ATOM_N, nums_block_w_a)>, Int<cute::gcd(ATOM_N, nums_block_w_a)>>;
  using PrefetchBThrShape = Shape<Int<ATOM_M /cute::gcd(ATOM_M, nums_block_w_b)>, Int<cute::gcd(ATOM_M, nums_block_w_b)>>;
  using PrefetchATileSize = decltype(ceil_div(Shape<Int<SG_M>, Int<SG_K>>{},PrefetchAThrShape{}));
  using PrefetchBTileSize = decltype(ceil_div(Shape<Int<SG_K>, Int<SG_N>>{},PrefetchBThrShape{}));
  
  static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});

  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;

  using XE_Prefetch_A = decltype(cute::detail::prefetch_selector<PrefetchATileSize, ElementA>());
  using XE_Prefetch_B = decltype(cute::detail::prefetch_selector<PrefetchBTileSize, ElementB>());

  using  TensorMKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), make_shape(0,0,0), StrideA{}));   //(m, k)
  using  TensorNKL = decltype(make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), make_shape(0,0,0), StrideB{}));   //(n, k)
 
  using CopyA2 = decltype(make_tiled_copy(atom_load_A{}.with(
                                   nullptr, 0, 0),
                                   Layout<Shape<_1, Int<SubgroupSize>>>{},
                                   make_layout(make_shape(get<0>(typename traits_load_A::BlockShape{}),
                                                          get<1>(typename traits_load_A::BlockShape{}) / Int<SubgroupSize>{}))));
          
  using CopyB2 = decltype(make_tiled_copy(atom_load_B{}.with(
                                   nullptr, 0, 0),
                                   Layout<Shape<_1, Int<SubgroupSize>>>{},
                                   make_layout(make_shape(get<0>(typename traits_load_B::BlockShape{}),
                                                          get<1>(typename traits_load_B::BlockShape{}) / Int<SubgroupSize>{}))));
  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
  };

  struct Params {
    TensorMKL mA;
    TensorNKL mB;
    CopyA2 copy_A2;
    CopyB2 copy_B2;
  };

  //
  // Methods
  //

  CollectiveMma() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;

    auto [M,N,K,L] = problem_shape;

    auto mA_mkl = make_tensor(make_gmem_ptr(static_cast<ElementA const*>(args.ptr_A)),
                              make_layout(make_shape(M, K, L), args.dA));

    auto mB_nkl = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(args.ptr_B)),
                              make_layout(make_shape(N, K, L), args.dB));

    
    auto tiled_copy_a2 = make_tiled_copy(atom_load_A{}.with(
                                   static_cast<ElementA const*>(args.ptr_A), M, K),
                                   Layout<Shape<_1, Int<SubgroupSize>>>{},
                                   make_layout(make_shape(get<0>(typename traits_load_A::BlockShape{}),
                                                          get<1>(typename traits_load_A::BlockShape{}) / Int<SubgroupSize>{})));
    auto tiled_copy_b2 = make_tiled_copy(atom_load_B{}.with(
                                   static_cast<ElementB const*>(args.ptr_B), N, K),
                                   Layout<Shape<_1, Int<SubgroupSize>>>{},
                                   make_layout(make_shape(get<0>(typename traits_load_B::BlockShape{}),
                                                          get<1>(typename traits_load_B::BlockShape{}) / Int<SubgroupSize>{})));

    return Params{mA_mkl, mB_nkl, tiled_copy_a2, tiled_copy_b2};
  }

  /// Perform a subgroup-scoped matrix multiply-accumulate
  template <
    int PrefetchStrideA,
    int PrefetchStrideB,
    class FrgTensorD,
    class TensorA,
    class TensorB,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK,
    class BlkCoord
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      BlkCoord const &blk_coord,
      int const &K_start,
      int thread_idx,
      char *smem_buf,
      Params const& mainloop) 
  {
    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    (void)residue_mnk;
    (void)thread_idx;
    (void)smem_buf;
    
    auto [m_idx, n_idx, k_idx, l_idx] = blk_coord;

    auto linearize_stride_3d = [&](auto coord_stride, auto tensor_stride){
    auto [coord_stride_0, coord_stride_1, coord_stride_2] = coord_stride;
      return make_stride(basis_value(coord_stride_0) * basis_get(coord_stride_0, tensor_stride),
                         basis_value(coord_stride_1) * basis_get(coord_stride_1, tensor_stride),
                         basis_value(coord_stride_2) * basis_get(coord_stride_2, tensor_stride));
    };

    Tensor gA_nullptr = make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)), 
                               gA.shape(), 
                               linearize_stride_3d(gA.stride(), StrideA{}));
    Tensor gB_nullptr = make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)), 
                               gB.shape(), 
                               linearize_stride_3d(gB.stride(), StrideB{}));
    
    auto tiled_copy_a = make_xe_2d_copy(atom_load_A{}.with(mainloop.mA),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});
    auto tiled_copy_b = make_xe_2d_copy(atom_load_B{}.with(mainloop.mB),
                                             Layout<Shape<_1, Int<SubgroupSize>>>{});

    // Partition the copying of A and B tiles across the threads
    auto thr_copy_A = tiled_copy_a.get_slice(thread_idx);
    auto thr_copy_B = tiled_copy_b.get_slice(thread_idx);
    
    auto thr_copy_A2 = mainloop.copy_A2.get_slice(thread_idx);
    auto thr_copy_B2 = mainloop.copy_B2.get_slice(thread_idx);

    // Instantiate the MMA object and get thread slice
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx & ~15);

    // Partition fragment
    Tensor fragment_A = thr_mma.partition_fragment_A(gA_nullptr(_, _, 0));
    Tensor fragment_B = thr_mma.partition_fragment_B(gB_nullptr(_, _, 0));

    // Retile for copy
    Tensor copy_tCrA = thr_copy_A.retile_D(fragment_A);
    Tensor copy_tCrB = thr_copy_B.retile_D(fragment_B);

    // Retile for cute::gemm
    Tensor mma_tCrA = thr_copy_A.retile_MMA(thr_mma, fragment_A);
    Tensor mma_tCrB = thr_copy_B.retile_MMA(thr_mma, fragment_B);
    
    // Partition
    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    
    Tensor tAgA = thr_copy_A2.retile_S(tCgA);
    Tensor tBgB = thr_copy_B2.retile_S(tCgB);
    Tensor tAgA2 = thr_copy_A2.partition_S(gA);
    Tensor tBgB2 = thr_copy_B2.partition_S(gB);

    // Register fragments
    //Tensor tCrA = thr_mma.make_fragment_A(gA_nullptr);
    //Tensor tCrB = thr_mma.make_fragment_B(gB_nullptr);
    
    //Tensor tCrA = make_tensor<ElementA>(tCgA(_,_,_,_0{}).shape(), linearize_stride_3d(tCgA(_,_,_,_0{}).stride(), StrideA{}));
    //Tensor tCrB = make_tensor<ElementB>(tCgB(_,_,_,_0{}).shape(), linearize_stride_3d(tCgB(_,_,_,_0{}).stride(), StrideB{}));

    //TODO private ???
    //Tensor tAgA = thr_copy_A.partition_D(gA);
    //Tensor tBgB = thr_copy_B.partition_D(gB);


  #if CUTLASS_ENABLE_DEBUG_PRINTS
    if (thread(LOG_THREAD, LOG_GROUP)) {
        print("======================= A: \n");
        print("  gA : "); print(gA); print("\n");
        print("copy_tCrA : "); print(copy_tCrA); print("\n");
        print("  mma_tCrA : "); print(mma_tCrA); print("\n");

        print("=====================  B :\n");
        print("  gB : "); print(gB); print("\n");
        print("copy_tCrB : "); print(copy_tCrB); print("\n");
        print("  mma_tCrB : "); print(mma_tCrB); print("\n");

        print("=====================  Config: \n");
        print("  threads per workgroup : "); print(MaxThreadsPerBlock); print("\n");
        print("  SubgroupTileShape : "); print(SubgroupTileShape{}); print("\n");

        print(" PrefetchAThrShape :    ");print(PrefetchAThrShape{});print("\n");
        print(" PrefetchBThrShape :    ");print(PrefetchBThrShape{});print("\n");
        print(" PrefetchATileSize :    ");print(PrefetchATileSize{});print("\n");
        print(" PrefetchBTileSize :    ");print(PrefetchBTileSize{});print("\n");
      }
 #endif

    //
    // Mainloop
    //
  #ifdef CUTLASS_SYCL_SWITCH_WG
    const int m_coord = n_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = m_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
  #else
    const int m_coord = m_idx * BLK_M + (get_sub_group_id() / ATOM_N) * SG_M;
    const int n_coord = n_idx * BLK_N + (get_sub_group_id() % ATOM_N) * SG_N;
  #endif
    const int l_coord = l_idx;

    Tensor block2d_copy_iter_a = tiled_copy_a.get_pvc_tensor(make_coord(m_coord, 0, l_coord), copy_tCrA.shape());
    auto copy_iter_a = append_pvc_tensor<1>(block2d_copy_iter_a, k_tile_count, BLK_K);

    Tensor block2d_copy_iter_b = tiled_copy_b.get_pvc_tensor(make_coord(n_coord, 0, l_coord), copy_tCrB.shape());
    auto copy_iter_b = append_pvc_tensor<1>(block2d_copy_iter_b, k_tile_count, BLK_K);

    const int k_start_idx = crd2idx((*k_tile_iter), make_shape(K_start));
    int prefetch_k = 0;

    Tensor block2d_prefetch_iter_a = XE_Prefetch_A{}.get_pvc_tensor(
                               make_coord(m_coord + (get_sub_group_id() % ATOM_N) / get<1>(PrefetchAThrShape{}) * get<0>(PrefetchATileSize{}),
                                          (k_start_idx + (get_sub_group_id() % ATOM_N) % get<1>(PrefetchAThrShape{})) * PrefetchStrideA,
                                          l_coord),
                               make_shape(_1{}, _1{}, _1{}));
    auto prefetch_iter_a = append_pvc_tensor<1>(block2d_prefetch_iter_a, k_tile_count, BLK_K);

    Tensor block2d_prefetch_iter_b = XE_Prefetch_B{}.get_pvc_tensor(
                               make_coord((get_sub_group_id() / ATOM_N / get<1>(PrefetchBThrShape{}) + k_start_idx) * PrefetchStrideB,
                                           n_coord + (get_sub_group_id() / ATOM_N) % get<1>(PrefetchBThrShape{}) * get<1>(PrefetchBTileSize{}),
                                           l_coord),
                               make_shape(_1{}, _1{}, _1{}));
    auto prefetch_iter_b = append_pvc_tensor<0>(block2d_prefetch_iter_b, k_tile_count, BLK_K);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < DispatchPolicy::Stages; i++, prefetch_k++) {
      if constexpr(cute::detail::has_prefetch<GmemTiledCopyA>) {
        prefetch(tiled_copy_a, prefetch_iter_a(_,_,_,prefetch_k));
      }
      if constexpr(cute::detail::has_prefetch<GmemTiledCopyB>) {
        prefetch(tiled_copy_b, prefetch_iter_b(_,_,_,prefetch_k));
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_tile = 0, k = k_start_idx; k_tile < k_tile_count; ++k_tile, ++k, ++prefetch_k) {
      // Copy gmem to rmem for the first k_tile
      /*if(cute::thread(99,3) && k_tile == 0){
        print("mma\n");
        print("gA: "); print(gA); print("\n");
        print("blk_coord: "); print(blk_coord); print("\n");
        print("m_coord: "); print(m_coord); print("\n");
        print("n_coord: "); print(n_coord); print("\n");

        print("tCgA: "); print(tCgA); print("\n");
        print("tAgA: "); print(tAgA); print("\n");
        print("tAgA2: "); print(tAgA2); print("\n");
        print("copy_iter_a: "); print(copy_iter_a); print("\n");
        //print("tCgA(_,_,_,k).data().coord_: "); print(tCgA(_,_,_,k).data().coord_); print("\n");
        //print("copy_iter_a(_,_,_,k).data().coord_: "); print(copy_iter_a(_,_,_,k).data().coord_); print("\n");
        print("tCgB: "); print(tCgB); print("\n");
        print("tBgB: "); print(tBgB); print("\n");
        print("tBgB2: "); print(tBgB2); print("\n");
        print("copy_iter_b: "); print(copy_iter_b); print("\n");
        //print("tCgB(_,_,_,k).data().coord_: "); print(tCgB(_,_,_,k).data().coord_); print("\n");
        //print("copy_iter_b(_,_,_,k).data().coord_: "); print(copy_iter_b(_,_,_,k).data().coord_); print("\n");

        //print("StrideA{}: "); print(StrideA{}); print("\n");
        //print("tCgA(_,_,_,_0{}).shape(): "); print(tCgA(_,_,_,_0{}).shape()); print("\n");
        //print("linearize_stride_3d(tCgA(_,_,_,_0{}).stride(), StrideA{}): "); print(linearize_stride_3d(tCgA(_,_,_,_0{}).stride(), StrideA{})); print("\n");
        //print("tCrA: "); print(tCrA); print("\n");
        print("tiled_mma.get_thr_layout_vmnk(): "); print(tiled_mma.get_thr_layout_vmnk()); print("\n");
        print("gA_nullptr: "); print(gA_nullptr); print("\n");
        print("fragment_A: "); print(fragment_A); print("\n");
        print("fragment_B: "); print(fragment_B); print("\n");
        print("copy_tCrA: "); print(copy_tCrA); print("\n");
        print("copy_tCrB: "); print(copy_tCrB); print("\n");
        print("mma_tCrA: "); print(mma_tCrA); print("\n");
        //print("block2d_prefetch_iter_a: "); print(block2d_prefetch_iter_a); print("\n");
        print("\n");
      }*/
      //if(cute::thread(99,3) && k_tile == 0){
        //print("copy A: "); print("\n");
        copy(tiled_copy_a, tAgA(_,_,_,k), copy_tCrA);
        //copy(tiled_copy_a, copy_iter_a(_,_,_,k), copy_tCrA);
        //print("copy B: "); print("\n");
        copy(tiled_copy_b, tBgB(_,_,_,k), copy_tCrB);
        //copy(tiled_copy_b, copy_iter_b(_,_,_,k), copy_tCrB);
        //print("done copying: "); print("\n\n");
      //}

      if(prefetch_k < k_tile_count) {
        if constexpr(cute::detail::has_prefetch<GmemTiledCopyA>) {
          prefetch(tiled_copy_a, prefetch_iter_a(_,_,_,prefetch_k));
        }
        if constexpr(cute::detail::has_prefetch<GmemTiledCopyB>) {
          prefetch(tiled_copy_b, prefetch_iter_b(_,_,_,prefetch_k));
        } 
      }

      cute::gemm(tiled_mma, mma_tCrA, mma_tCrB, accum);
    }
  }
};

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
