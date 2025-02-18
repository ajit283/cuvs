#include "cuvs/core/c_api.h"
#include "cuvs/distance/distance.h"
#include "cuvs/neighbors/cagra.h"
#include "cuvs/neighbors/dynamic_batching.h"
#include "dlpack/dlpack.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(DynamicBatchingC, BuildSearch)
{
  // Create cuvs resources.
  cuvsResources_t res;
  cuvsError_t err = cuvsResourcesCreate(&res);
  ASSERT_EQ(err, CUVS_SUCCESS);

  cudaStream_t stream;
  err = cuvsStreamGet(res, &stream);
  ASSERT_EQ(err, CUVS_SUCCESS);

  /********** Build the upstream (CAGRA) index **********/
  // Define a small dataset: 4 vectors of dimension 2.
  float dataset[4][2] = {{0.74021935f, 0.9209938f},
                         {0.03902049f, 0.9689629f},
                         {0.92514056f, 0.4463501f},
                         {0.6673192f, 0.10993068f}};

  // Set up a DLManagedTensor for the dataset (host memory is acceptable).
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;
  dataset_tensor.manager_ctx                  = nullptr;
  dataset_tensor.deleter                      = nullptr;

  // Create a CAGRA index.
  cuvsCagraIndex_t cagra_index;
  err = cuvsCagraIndexCreate(&cagra_index);
  ASSERT_EQ(err, CUVS_SUCCESS);

  // Create CAGRA index build parameters.
  cuvsCagraIndexParams_t cagra_index_params;
  err = cuvsCagraIndexParamsCreate(&cagra_index_params);
  ASSERT_EQ(err, CUVS_SUCCESS);

  // Build the upstream index.
  err = cuvsCagraBuild(res, cagra_index_params, &dataset_tensor, cagra_index);
  ASSERT_EQ(err, CUVS_SUCCESS);

  // Create CAGRA search parameters.
  cuvsCagraSearchParams_t cagra_search_params;
  err = cuvsCagraSearchParamsCreate(&cagra_search_params);
  ASSERT_EQ(err, CUVS_SUCCESS);

  // Pack the upstream into a dynamic batching upstream structure.
  struct cuvsDynamicBatchingUpstream dynb_upstream;
  dynb_upstream.type                = CAGRA;
  dynb_upstream.cagra.index         = cagra_index;
  dynb_upstream.cagra.search_params = cagra_search_params;

  /********** Create the dynamic batching index **********/
  // Allocate and configure dynamic batching index parameters.
  cuvsDynamicBatchingIndexParams_t dynb_index_params;
  err = cuvsDynamicBatchingIndexParamsCreate(&dynb_index_params);
  ASSERT_EQ(err, CUVS_SUCCESS);
  // Use K=1 (single neighbor search) and maximum batch size = 4.
  dynb_index_params->k                     = 1;
  dynb_index_params->max_batch_size        = 4;
  dynb_index_params->n_queues              = 1;
  dynb_index_params->convervative_dispatch = false;

  // Use no filter.
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = 0;

  // Create the dynamic batching index (wraps the upstream CAGRA index).
  cuvsDynamicBatchingIndex_t dynb_index;
  err = cuvsDynamicBatchingIndexCreate(res, &dynb_index, dynb_index_params, &dynb_upstream, filter);
  ASSERT_EQ(err, CUVS_SUCCESS);

  /********** Prepare query and output tensors on device **********/
  // Define queries: 4 queries, each of dimension 2.
  float h_queries[4][2] = {{0.48216683f, 0.0428398f},
                           {0.5084142f, 0.6545497f},
                           {0.51260436f, 0.2643005f},
                           {0.05198065f, 0.5789965f}};

  // Allocate device memory for queries.
  float* d_queries;
  size_t queries_size  = 4 * 2 * sizeof(float);
  cudaError_t cuda_err = cudaMalloc(&d_queries, queries_size);
  ASSERT_EQ(cuda_err, cudaSuccess);
  cuda_err = cudaMemcpy(d_queries, h_queries, queries_size, cudaMemcpyHostToDevice);
  ASSERT_EQ(cuda_err, cudaSuccess);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = d_queries;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;
  queries_tensor.manager_ctx                  = nullptr;
  queries_tensor.deleter                      = nullptr;

  // Allocate device memory for output neighbors (4 queries × 1 neighbor).
  uint32_t* d_neighbors;
  size_t neighbors_size = 4 * sizeof(uint32_t);
  cuda_err              = cudaMalloc(&d_neighbors, neighbors_size);
  ASSERT_EQ(cuda_err, cudaSuccess);
  cuda_err = cudaMemset(d_neighbors, 0, neighbors_size);
  ASSERT_EQ(cuda_err, cudaSuccess);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = d_neighbors;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;
  neighbors_tensor.manager_ctx                  = nullptr;
  neighbors_tensor.deleter                      = nullptr;

  // Allocate device memory for output distances.
  float* d_distances;
  size_t distances_size = 4 * sizeof(float);
  cuda_err              = cudaMalloc(&d_distances, distances_size);
  ASSERT_EQ(cuda_err, cudaSuccess);
  cuda_err = cudaMemset(d_distances, 0, distances_size);
  ASSERT_EQ(cuda_err, cudaSuccess);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = d_distances;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;
  distances_tensor.manager_ctx                  = nullptr;
  distances_tensor.deleter                      = nullptr;

  // Create dynamic batching search parameters.
  cuvsDynamicBatchingSearchParams_t dynb_search_params;
  err = cuvsDynamicBatchingSearchParamsCreate(&dynb_search_params);
  ASSERT_EQ(err, CUVS_SUCCESS);

  /********** Run dynamic batching search **********/
  err = cuvsDynamicBatchingSearch(
    res, dynb_search_params, dynb_index, &queries_tensor, &neighbors_tensor, &distances_tensor);
  if (err != CUVS_SUCCESS) {
    printf("%s", cuvsGetLastErrorText());
    exit(1);
  }
  ASSERT_EQ(err, CUVS_SUCCESS);

  // Copy the results back to host.
  uint32_t h_neighbors[4] = {0};
  float h_distances[4]    = {0.0f};
  cuda_err = cudaMemcpy(h_neighbors, d_neighbors, neighbors_size, cudaMemcpyDeviceToHost);
  ASSERT_EQ(cuda_err, cudaSuccess);
  cuda_err = cudaMemcpy(h_distances, d_distances, distances_size, cudaMemcpyDeviceToHost);
  ASSERT_EQ(cuda_err, cudaSuccess);

  // Expected results (from the CAGRA C test).
  uint32_t expected_neighbors[4] = {3, 0, 3, 1};
  float expected_distances[4]    = {0.03878258f, 0.12472608f, 0.04776672f, 0.15224178f};

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(h_neighbors[i], expected_neighbors[i]) << "Neighbor mismatch at query " << i;
    EXPECT_NEAR(h_distances[i], expected_distances[i], 0.001f)
      << "Distance mismatch at query " << i;
  }

  /********** Cleanup **********/
  err = cuvsDynamicBatchingSearchParamsDestroy(dynb_search_params);
  ASSERT_EQ(err, CUVS_SUCCESS);
  err = cuvsDynamicBatchingIndexParamsDestroy(dynb_index_params);
  ASSERT_EQ(err, CUVS_SUCCESS);
  err = cuvsDynamicBatchingIndexDestroy(dynb_index);
  ASSERT_EQ(err, CUVS_SUCCESS);
  err = cuvsCagraSearchParamsDestroy(cagra_search_params);
  ASSERT_EQ(err, CUVS_SUCCESS);
  err = cuvsCagraIndexParamsDestroy(cagra_index_params);
  ASSERT_EQ(err, CUVS_SUCCESS);
  err = cuvsCagraIndexDestroy(cagra_index);
  ASSERT_EQ(err, CUVS_SUCCESS);
  err = cuvsResourcesDestroy(res);
  ASSERT_EQ(err, CUVS_SUCCESS);

  cudaFree(d_queries);
  cudaFree(d_neighbors);
  cudaFree(d_distances);
}