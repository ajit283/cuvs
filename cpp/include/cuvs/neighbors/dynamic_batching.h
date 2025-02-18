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

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/common.h>
#include <dlpack/dlpack.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup cagra_c_index_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

typedef struct {
  cuvsCagraIndex_t index;
  cuvsCagraSearchParams_t search_params;
} cuvsDynamicBatchingCagraUpstream;

typedef struct {
  cuvsBruteForceIndex_t index;
} cuvsDynamicBatchingBruteForceUpstream;

typedef enum { CAGRA, BRUTE_FORCE } IndexType;

struct cuvsDynamicBatchingUpstream {
  IndexType type;
  union {
    cuvsDynamicBatchingCagraUpstream cagra;
    cuvsDynamicBatchingBruteForceUpstream brute_force;
  };
};

typedef struct cuvsDynamicBatchingUpstream* cuvsDynamicBatchingUpstream_t;

/**
 * @brief Supplemental parameters to build CAGRA Index
 *
 */
struct cuvsDynamicBatchingIndexParams {
  int64_t k;
  int64_t max_batch_size;
  size_t n_queues;
  bool convervative_dispatch;
};

typedef struct cuvsDynamicBatchingIndexParams* cuvsDynamicBatchingIndexParams_t;

/**
 * @brief Allocate CAGRA Index params, and populate with default values
 *
 * @param[in] params cuvsCagraIndexParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsDynamicBatchingIndexParamsCreate(cuvsDynamicBatchingIndexParams_t* params);

/**
 * @brief De-allocate CAGRA Index params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsDynamicBatchingIndexParamsDestroy(cuvsDynamicBatchingIndexParams_t params);

/**
 * @}
 */

/**
 * @defgroup cagra_c_search_params C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Supplemental parameters to search CAGRA index
 *
 */
struct cuvsDynamicBatchingSearchParams {
  double dispatch_timeout_ms;
};

typedef struct cuvsDynamicBatchingSearchParams* cuvsDynamicBatchingSearchParams_t;

/**
 * @brief Allocate CAGRA search params, and populate with default values
 *
 * @param[in] params cuvsCagraSearchParams_t to allocate
 * @return cuvsError_t
 */
cuvsError_t cuvsDynamicBatchingSearchParamsCreate(cuvsDynamicBatchingSearchParams_t* params);

/**
 * @brief De-allocate CAGRA search params
 *
 * @param[in] params
 * @return cuvsError_t
 */
cuvsError_t cuvsDynamicBatchingSearchParamsDestroy(cuvsDynamicBatchingSearchParams_t params);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Struct to hold address of cuvs::neighbors::cagra::index and its active trained dtype
 *
 */
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsDynamicBatchingIndex;

typedef cuvsDynamicBatchingIndex* cuvsDynamicBatchingIndex_t;

/**
 * @brief Allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to allocate
 * @return cagraError_t
 */
cuvsError_t cuvsDynamicBatchingIndexCreate(cuvsResources_t res,
                                           cuvsDynamicBatchingIndex_t* index,
                                           cuvsDynamicBatchingIndexParams_t params,
                                           cuvsDynamicBatchingUpstream_t upstream,
                                           cuvsFilter filter);

/**
 * @brief De-allocate CAGRA index
 *
 * @param[in] index cuvsCagraIndex_t to de-allocate
 */
cuvsError_t cuvsDynamicBatchingIndexDestroy(cuvsDynamicBatchingIndex_t index);

/**
 * @}
 */

/**
 * @brief Build a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`,
 *        or `kDLCPU`. Also, acceptable underlying types are:
 *        1. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *        2. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *        3. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *        4. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 *
 * // Create default index params
 * cuvsCagraIndexParams_t params;
 * cuvsError_t params_create_status = cuvsCagraIndexParamsCreate(&params);
 *
 * // Create CAGRA index
 * cuvsCagraIndex_t index;
 * cuvsError_t index_create_status = cuvsCagraIndexCreate(&index);
 *
 * // Build the CAGRA Index
 * cuvsError_t build_status = cuvsCagraBuild(res, params, &dataset, index);
 *
 * // de-allocate `params`, `index` and `res`
 * cuvsError_t params_destroy_status = cuvsCagraIndexParamsDestroy(params);
 * cuvsError_t index_destroy_status = cuvsCagraIndexDestroy(index);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraIndexParams_t used to build CAGRA index
 * @param[in] dataset DLManagedTensor* training dataset
 * @param[out] index cuvsCagraIndex_t Newly built CAGRA index
 * @return cuvsError_t
 */
cuvsError_t cuvsCagraBuild(cuvsResources_t res,
                           cuvsCagraIndexParams_t params,
                           DLManagedTensor* dataset,
                           cuvsCagraIndex_t index);

/**
 * @}
 */

/**
 * @defgroup cagra_c_index_search C API for CUDA ANN Graph-based nearest neighbor search
 * @{
 */
/**
 * @brief Search a CAGRA index with a `DLManagedTensor` which has underlying
 *        `DLDeviceType` equal to `kDLCUDA`, `kDLCUDAHost`, `kDLCUDAManaged`.
 *        It is also important to note that the CAGRA Index must have been built
 *        with the same type of `queries`, such that `index.dtype.code ==
 * queries.dl_tensor.dtype.code` Types for input are:
 *        1. `queries`:
 *          a. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *          b. `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 16`
 *          c. `kDLDataType.code == kDLInt` and `kDLDataType.bits = 8`
 *          d. `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 8`
 *        2. `neighbors`: `kDLDataType.code == kDLUInt` and `kDLDataType.bits = 32`
 *        3. `distances`: `kDLDataType.code == kDLFloat` and `kDLDataType.bits = 32`
 *
 * @code {.c}
 * #include <cuvs/core/c_api.h>
 * #include <cuvs/neighbors/cagra.h>
 *
 * // Create cuvsResources_t
 * cuvsResources_t res;
 * cuvsError_t res_create_status = cuvsResourcesCreate(&res);
 *
 * // Assume a populated `DLManagedTensor` type here
 * DLManagedTensor dataset;
 * DLManagedTensor queries;
 * DLManagedTensor neighbors;
 *
 * // Create default search params
 * cuvsCagraSearchParams_t params;
 * cuvsError_t params_create_status = cuvsCagraSearchParamsCreate(&params);
 *
 * // Search the `index` built using `cuvsCagraBuild`
 * cuvsError_t search_status = cuvsCagraSearch(res, params, index, &queries, &neighbors,
 * &distances);
 *
 * // de-allocate `params` and `res`
 * cuvsError_t params_destroy_status = cuvsCagraSearchParamsDestroy(params);
 * cuvsError_t res_destroy_status = cuvsResourcesDestroy(res);
 * @endcode
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] params cuvsCagraSearchParams_t used to search CAGRA index
 * @param[in] index cuvsCagraIndex which has been returned by `cuvsCagraBuild`
 * @param[in] queries DLManagedTensor* queries dataset to search
 * @param[out] neighbors DLManagedTensor* output `k` neighbors for queries
 * @param[out] distances DLManagedTensor* output `k` distances for queries
 * @param[in] filter cuvsFilter input filter that can be used
              to filter queries and neighbors based on the given bitset.
 */
cuvsError_t cuvsDynamicBatchingSearch(cuvsResources_t res,
                                      cuvsDynamicBatchingSearchParams_t params,
                                      cuvsDynamicBatchingIndex_t index,
                                      DLManagedTensor* queries,
                                      DLManagedTensor* neighbors,
                                      DLManagedTensor* distances);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
