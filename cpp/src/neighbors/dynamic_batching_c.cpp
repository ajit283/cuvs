
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/core/exceptions.hpp>
#include <cuvs/core/interop.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.h>
#include <cuvs/neighbors/dynamic_batching.h>
#include <cuvs/neighbors/dynamic_batching.hpp>

#include <fstream>

namespace {

template <typename T>
void* _create(cuvsResources_t res,
              cuvsDynamicBatchingIndex index,
              cuvsDynamicBatchingIndexParams params,
              cuvsDynamicBatchingUpstream upstream,
              cuvsFilter filter)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);

  switch (upstream.type) {
    case CAGRA: {
      auto upstream_index_ptr =
        reinterpret_cast<cuvs::neighbors::cagra::index<T, uint32_t>*>(upstream.cagra.index->addr);
      // auto index_ptr =
      //   reinterpret_cast<cuvs::neighbors::dynamic_batching::index<T, uint32_t>>(index.addr);
      auto index_params   = cuvs::neighbors::dynamic_batching::index_params();
      index_params.metric = upstream_index_ptr->metric(),

      index_params.conservative_dispatch = params.convervative_dispatch;
      index_params.n_queues              = params.n_queues;
      index_params.k                     = params.k;
      index_params.max_batch_size        = params.max_batch_size;

      auto upstream_search_params           = cuvs::neighbors::cagra::search_params();
      upstream_search_params.max_queries    = upstream.cagra.search_params->max_queries;
      upstream_search_params.itopk_size     = upstream.cagra.search_params->itopk_size;
      upstream_search_params.max_iterations = upstream.cagra.search_params->max_iterations;
      upstream_search_params.algo =
        static_cast<cuvs::neighbors::cagra::search_algo>(upstream.cagra.search_params->algo);
      upstream_search_params.team_size         = upstream.cagra.search_params->team_size;
      upstream_search_params.search_width      = upstream.cagra.search_params->search_width;
      upstream_search_params.min_iterations    = upstream.cagra.search_params->min_iterations;
      upstream_search_params.thread_block_size = upstream.cagra.search_params->thread_block_size;
      upstream_search_params.hashmap_mode =
        static_cast<cuvs::neighbors::cagra::hash_mode>(upstream.cagra.search_params->hashmap_mode);
      upstream_search_params.hashmap_min_bitlen = upstream.cagra.search_params->hashmap_min_bitlen;
      upstream_search_params.hashmap_max_fill_rate =
        upstream.cagra.search_params->hashmap_max_fill_rate;
      upstream_search_params.num_random_samplings =
        upstream.cagra.search_params->num_random_samplings;
      upstream_search_params.rand_xor_mask = upstream.cagra.search_params->rand_xor_mask;

      // auto index = new cuvs::neighbors::dynamic_batching::index<T, uint32_t>(
      //   res_ptr, index_params, upstream_index_ptr, upstream_search_params);

      auto index = new cuvs::neighbors::dynamic_batching::index<T, uint32_t>(
        *res_ptr, index_params, *upstream_index_ptr, upstream_search_params);

      return index;
    };
    case BRUTE_FORCE: return nullptr; ;
  };
  return nullptr;
}

template <typename T>
void _search(cuvsResources_t res,
             cuvsDynamicBatchingSearchParams params,
             cuvsDynamicBatchingIndex index,
             DLManagedTensor* queries_tensor,
             DLManagedTensor* neighbors_tensor,
             DLManagedTensor* distances_tensor)
{
  auto res_ptr = reinterpret_cast<raft::resources*>(res);
  auto index_ptr =
    reinterpret_cast<cuvs::neighbors::dynamic_batching::index<T, uint32_t>*>(index.addr);

  auto search_params                = cuvs::neighbors::dynamic_batching::search_params();
  search_params.dispatch_timeout_ms = params.dispatch_timeout_ms;

  using queries_mdspan_type   = raft::device_matrix_view<T const, int64_t, raft::row_major>;
  using neighbors_mdspan_type = raft::device_matrix_view<uint32_t, int64_t, raft::row_major>;
  using distances_mdspan_type = raft::device_matrix_view<float, int64_t, raft::row_major>;
  auto queries_mds            = cuvs::core::from_dlpack<queries_mdspan_type>(queries_tensor);
  auto neighbors_mds          = cuvs::core::from_dlpack<neighbors_mdspan_type>(neighbors_tensor);
  auto distances_mds          = cuvs::core::from_dlpack<distances_mdspan_type>(distances_tensor);

  cuvs::neighbors::dynamic_batching::search(
    *res_ptr, search_params, *index_ptr, queries_mds, neighbors_mds, distances_mds);
}

}  // namespace

extern "C" cuvsError_t cuvsDynamicBatchingIndexCreate(cuvsResources_t res,
                                                      cuvsDynamicBatchingIndex_t* index,
                                                      cuvsDynamicBatchingIndexParams_t params,
                                                      cuvsDynamicBatchingUpstream_t upstream,
                                                      cuvsFilter filter

)
{
  return cuvs::core::translate_exceptions([=] {
    DLDataType dtype;
    switch (upstream->type) {
      case CAGRA: {
        dtype = upstream->cagra.index->dtype;
        break;
      }
      case BRUTE_FORCE: {
        dtype = upstream->brute_force.index->dtype;
        break;
      }
    }

    *index = new cuvsDynamicBatchingIndex{};

    (*index)->dtype = dtype;

    if (dtype.code == kDLFloat && dtype.bits == 32) {
      (*index)->addr =
        reinterpret_cast<uintptr_t>(_create<float>(res, **index, *params, *upstream, filter));
    } else if (dtype.code == kDLFloat && dtype.bits == 16) {
      (*index)->addr =
        reinterpret_cast<uintptr_t>(_create<half>(res, **index, *params, *upstream, filter));
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      (*index)->addr =
        reinterpret_cast<uintptr_t>(_create<int8_t>(res, **index, *params, *upstream, filter));
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      (*index)->addr =
        reinterpret_cast<uintptr_t>(_create<uint8_t>(res, **index, *params, *upstream, filter));
    } else {
      RAFT_FAIL("Unsupported dataset DLtensor dtype: %d and bits: %d", dtype.code, dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDynamicBatchingIndexDestroy(cuvsDynamicBatchingIndex_t index_c_ptr)
{
  return cuvs::core::translate_exceptions([=] {
    if (index_c_ptr->dtype.code == kDLFloat) {
      auto index_ptr = reinterpret_cast<cuvs::neighbors::dynamic_batching::index<float, uint32_t>*>(
        index_c_ptr->addr);
      delete index_ptr;
    } else if (index_c_ptr->dtype.code == kDLInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::dynamic_batching::index<int8_t, uint32_t>*>(
          index_c_ptr->addr);
      delete index_ptr;
    } else if (index_c_ptr->dtype.code == kDLUInt) {
      auto index_ptr =
        reinterpret_cast<cuvs::neighbors::dynamic_batching::index<uint8_t, uint32_t>*>(
          index_c_ptr->addr);
      delete index_ptr;
    }
    delete index_c_ptr;
  });
}

extern "C" cuvsError_t cuvsDynamicBatchingSearch(cuvsResources_t res,
                                                 cuvsDynamicBatchingSearchParams_t params,
                                                 cuvsDynamicBatchingIndex_t index_c_ptr,
                                                 DLManagedTensor* queries_tensor,
                                                 DLManagedTensor* neighbors_tensor,
                                                 DLManagedTensor* distances_tensor)
{
  return cuvs::core::translate_exceptions([=] {
    auto queries   = queries_tensor->dl_tensor;
    auto neighbors = neighbors_tensor->dl_tensor;
    auto distances = distances_tensor->dl_tensor;

    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(neighbors),
                 "neighbors should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(distances),
                 "distances should have device compatible memory");

    RAFT_EXPECTS(neighbors.dtype.code == kDLUInt && neighbors.dtype.bits == 32,
                 "neighbors should be of type uint32_t");
    RAFT_EXPECTS(distances.dtype.code == kDLFloat && neighbors.dtype.bits == 32,
                 "distances should be of type float32");

    auto index = *index_c_ptr;
    RAFT_EXPECTS(queries.dtype.code == index.dtype.code, "type mismatch between index and queries");

    if (queries.dtype.code == kDLFloat && queries.dtype.bits == 32) {
      _search<float>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLFloat && queries.dtype.bits == 16) {
      _search<half>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLInt && queries.dtype.bits == 8) {
      _search<int8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else if (queries.dtype.code == kDLUInt && queries.dtype.bits == 8) {
      _search<uint8_t>(res, *params, index, queries_tensor, neighbors_tensor, distances_tensor);
    } else {
      RAFT_FAIL("Unsupported queries DLtensor dtype: %d and bits: %d",
                queries.dtype.code,
                queries.dtype.bits);
    }
  });
}

extern "C" cuvsError_t cuvsDynamicBatchingIndexParamsCreate(
  cuvsDynamicBatchingIndexParams_t* params)
{
  return cuvs::core::translate_exceptions([=] {
    *params = new cuvsDynamicBatchingIndexParams{
      .k = 64, .max_batch_size = 100, .n_queues = 3, .convervative_dispatch = false};
  });
}

extern "C" cuvsError_t cuvsDynamicBatchingIndexParamsDestroy(
  cuvsDynamicBatchingIndexParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}

extern "C" cuvsError_t cuvsDynamicBatchingSearchParamsCreate(
  cuvsDynamicBatchingSearchParams_t* params)
{
  return cuvs::core::translate_exceptions(
    [=] { *params = new cuvsDynamicBatchingSearchParams{.dispatch_timeout_ms = 1.0}; });
}

extern "C" cuvsError_t cuvsDynamicBatchingSearchParamsDestroy(
  cuvsDynamicBatchingSearchParams_t params)
{
  return cuvs::core::translate_exceptions([=] { delete params; });
}
