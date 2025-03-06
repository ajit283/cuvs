package hnsw_test

import (
	"math/rand/v2"
	"path/filepath"
	"testing"

	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/cagra"
	"github.com/rapidsai/cuvs/go/hnsw"
)

// Rename the file to hnsw_test.go

func TestHnswFromCagra(t *testing.T) {
	const (
		nDataPoints = 1024
		nFeatures   = 16
		nQueries    = 4
		k           = 4
		epsilon     = 0.001
	)

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	// Create random dataset
	testDataset := make([][]float32, nDataPoints)
	for i := range testDataset {
		testDataset[i] = make([]float32, nFeatures)
		for j := range testDataset[i] {
			testDataset[i][j] = rand.Float32()
		}

	}

	dataset, err := cuvs.NewTensor(testDataset)
	if err != nil {
		t.Fatalf("error creating dataset tensor: %v", err)
	}
	defer dataset.Close()

	// Set up CAGRA parameters and index
	cagraIndexParams, err := cagra.CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating CAGRA index params: %v", err)
	}
	defer cagraIndexParams.Close()

	cagraIndex, err := cagra.CreateIndex()
	if err != nil {
		t.Fatalf("error creating CAGRA index: %v", err)
	}
	defer cagraIndex.Close()

	// Use the first 4 points as queries
	queries, err := cuvs.NewTensor(testDataset[:nQueries])
	if err != nil {
		t.Fatalf("error creating queries tensor: %v", err)
	}
	defer queries.Close()

	neighborsInputSlice := make([][]uint64, nQueries)
	for i := range neighborsInputSlice {
		neighborsInputSlice[i] = make([]uint64, k)
	}

	neighbors, err := cuvs.NewTensor[uint64](neighborsInputSlice)
	if err != nil {
		t.Fatalf("error creating neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	distancesInputSlice := make([][]float32, nQueries)
	for i := range distancesInputSlice {
		distancesInputSlice[i] = make([]float32, k)
	}

	distances, err := cuvs.NewTensor[float32](distancesInputSlice)
	if err != nil {
		t.Fatalf("error creating distances tensor: %v", err)
	}
	defer distances.Close()

	// Move dataset to device and build CAGRA index
	if _, err := dataset.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset to device: %v", err)
	}

	if err := cagra.BuildIndex(resource, cagraIndexParams, &dataset, cagraIndex); err != nil {
		t.Fatalf("error building CAGRA index: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Now convert CAGRA to HNSW
	hnswIndexParams, err := hnsw.CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating HNSW index params: %v", err)
	}
	defer hnswIndexParams.Close()

	// Set hierarchy type to CPU (for extendability)
	if _, err := hnswIndexParams.SetHierarchy(hnsw.HierarchyCPU); err != nil {
		t.Fatalf("error setting HNSW hierarchy: %v", err)
	}

	// Set reasonable values for construction
	if _, err := hnswIndexParams.SetEfConstruction(200); err != nil {
		t.Fatalf("error setting ef_construction: %v", err)
	}

	if _, err := hnswIndexParams.SetNumThreads(4); err != nil {
		t.Fatalf("error setting num_threads: %v", err)
	}

	hnswIndex, err := hnsw.CreateIndex()
	if err != nil {
		t.Fatalf("error creating HNSW index: %v", err)
	}
	defer hnswIndex.Close()
	t.Log("CAGRA index built")

	// Convert CAGRA to HNSW
	if err := hnsw.FromCagra(resource, hnswIndexParams, cagraIndex, hnswIndex); err != nil {
		t.Fatalf("error converting CAGRA to HNSW: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Create HNSW search parameters
	hnswSearchParams, err := hnsw.CreateSearchParams()
	if err != nil {
		t.Fatalf("error creating HNSW search params: %v", err)
	}
	defer hnswSearchParams.Close()

	// Set search parameters
	if _, err := hnswSearchParams.SetEf(100); err != nil {
		t.Fatalf("error setting ef: %v", err)
	}

	if _, err := hnswSearchParams.SetNumThreads(4); err != nil {
		t.Fatalf("error setting search threads: %v", err)
	}

	// Move queries to device
	// if _, err := queries.ToDevice(&resource); err != nil {
	// 	t.Fatalf("error moving queries to device: %v", err)
	// }

	// Search HNSW index
	err = hnsw.SearchIndex(resource, hnswSearchParams, hnswIndex, &queries, &neighbors, &distances)
	if err != nil {
		t.Fatalf("error searching HNSW index: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Validate results
	neighborsSlice, err := neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}

	for i := range neighborsSlice {
		if neighborsSlice[i][0] != uint64(i) {
			t.Error("wrong neighbor, expected", i, "got", neighborsSlice[i][0])
		}
	}

	distancesSlice, err := distances.Slice()
	if err != nil {
		t.Fatalf("error getting distances slice: %v", err)
	}

	for i := range distancesSlice {
		if distancesSlice[i][0] >= epsilon || distancesSlice[i][0] <= -epsilon {
			t.Error("distance should be close to 0, got", distancesSlice[i][0])
		}
	}
}

func TestHnswExtend(t *testing.T) {
	const (
		nDataPoints       = 1024
		nAdditionalPoints = 128
		nFeatures         = 16
		nQueries          = 4
		k                 = 4
		epsilon           = 0.001
	)

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	// Create initial dataset
	testDataset := make([][]float32, nDataPoints)
	for i := range testDataset {
		testDataset[i] = make([]float32, nFeatures)
		for j := range testDataset[i] {
			testDataset[i][j] = rand.Float32()
		}
	}

	// Create additional dataset for extension
	additionalDataset := make([][]float32, nAdditionalPoints)
	for i := range additionalDataset {
		additionalDataset[i] = make([]float32, nFeatures)
		for j := range additionalDataset[i] {
			additionalDataset[i][j] = rand.Float32()
		}
	}

	// Create queries from the additional dataset
	testQueries := make([][]float32, nQueries)
	for i := range testQueries {
		testQueries[i] = make([]float32, nFeatures)
		copy(testQueries[i], additionalDataset[i])
	}

	dataset, err := cuvs.NewTensor(testDataset)
	if err != nil {
		t.Fatalf("error creating dataset tensor: %v", err)
	}
	defer dataset.Close()

	additionalTensor, err := cuvs.NewTensor(additionalDataset)
	if err != nil {
		t.Fatalf("error creating additional dataset tensor: %v", err)
	}
	defer additionalTensor.Close()

	// Create CAGRA index and parameters
	cagraIndexParams, err := cagra.CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating CAGRA index params: %v", err)
	}
	defer cagraIndexParams.Close()

	cagraIndex, err := cagra.CreateIndex()
	if err != nil {
		t.Fatalf("error creating CAGRA index: %v", err)
	}
	defer cagraIndex.Close()

	// Move dataset to device and build CAGRA index
	if _, err := dataset.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset to device: %v", err)
	}

	if err := cagra.BuildIndex(resource, cagraIndexParams, &dataset, cagraIndex); err != nil {
		t.Fatalf("error building CAGRA index: %v", err)
	}

	// Create HNSW index and parameters
	hnswIndexParams, err := hnsw.CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating HNSW index params: %v", err)
	}
	defer hnswIndexParams.Close()

	// Set hierarchy to CPU for extendability
	if _, err := hnswIndexParams.SetHierarchy(hnsw.HierarchyCPU); err != nil {
		t.Fatalf("error setting HNSW hierarchy: %v", err)
	}

	hnswIndex, err := hnsw.CreateIndex()
	if err != nil {
		t.Fatalf("error creating HNSW index: %v", err)
	}
	defer hnswIndex.Close()

	// Convert CAGRA to HNSW
	if err := hnsw.FromCagra(resource, hnswIndexParams, cagraIndex, hnswIndex); err != nil {
		t.Fatalf("error converting CAGRA to HNSW: %v", err)
	}

	// Create extension parameters
	extendParams, err := hnsw.CreateExtendParams()
	if err != nil {
		t.Fatalf("error creating extend params: %v", err)
	}
	defer extendParams.Close()

	if _, err := extendParams.SetNumThreads(4); err != nil {
		t.Fatalf("error setting num threads for extension: %v", err)
	}

	// Move additional data to device
	if _, err := additionalTensor.ToDevice(&resource); err != nil {
		t.Fatalf("error moving additional dataset to device: %v", err)
	}

	// Extend the HNSW index
	if err := hnsw.ExtendIndex(resource, extendParams, &additionalTensor, hnswIndex); err != nil {
		t.Fatalf("error extending HNSW index: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Create tensors for queries and results
	queries, err := cuvs.NewTensor(testQueries)
	if err != nil {
		t.Fatalf("error creating queries tensor: %v", err)
	}
	defer queries.Close()

	// Create neighbors tensor (changed from uint32 to uint64 and on-host)
	neighborsInputSlice := make([][]uint64, nQueries)
	for i := range neighborsInputSlice {
		neighborsInputSlice[i] = make([]uint64, k)
	}

	neighbors, err := cuvs.NewTensor[uint64](neighborsInputSlice)
	if err != nil {
		t.Fatalf("error creating neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	// Create distances tensor (changed to on-host)
	distancesInputSlice := make([][]float32, nQueries)
	for i := range distancesInputSlice {
		distancesInputSlice[i] = make([]float32, k)
	}

	distances, err := cuvs.NewTensor[float32](distancesInputSlice)
	if err != nil {
		t.Fatalf("error creating distances tensor: %v", err)
	}
	defer distances.Close()

	// Create search parameters
	searchParams, err := hnsw.CreateSearchParams()
	if err != nil {
		t.Fatalf("error creating search params: %v", err)
	}
	defer searchParams.Close()

	// Move queries to device
	if _, err := queries.ToDevice(&resource); err != nil {
		t.Fatalf("error moving queries to device: %v", err)
	}

	// Search the extended index
	err = hnsw.SearchIndex(resource, searchParams, hnswIndex, &queries, &neighbors, &distances)
	if err != nil {
		t.Fatalf("error searching extended HNSW index: %v", err)
	}

	// Move results back to host
	if _, err := neighbors.ToHost(&resource); err != nil {
		t.Fatalf("error moving neighbors to host: %v", err)
	}

	if _, err := distances.ToHost(&resource); err != nil {
		t.Fatalf("error moving distances to host: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Validate results - extended points should find themselves
	neighborsSlice, err := neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}

	// The first nearest neighbor of each query should be the corresponding
	// point from the additional dataset (index = nDataPoints + i)
	for i := range neighborsSlice {
		expectedIdx := uint64(nDataPoints + i)
		if neighborsSlice[i][0] != expectedIdx {
			t.Error("wrong neighbor, expected", expectedIdx, "got", neighborsSlice[i][0])
		}
	}
}

// TestHnswSerialize - Adjusted version with uint64 and on-host tensors
func TestHnswSerialize(t *testing.T) {
	const (
		nDataPoints = 512
		nFeatures   = 16
		nQueries    = 4
		k           = 4
		epsilon     = 0.001
	)

	resource, _ := cuvs.NewResource(nil)
	defer resource.Close()

	// Create dataset
	testDataset := make([][]float32, nDataPoints)
	for i := range testDataset {
		testDataset[i] = make([]float32, nFeatures)
		for j := range testDataset[i] {
			testDataset[i][j] = rand.Float32()
		}
	}

	dataset, err := cuvs.NewTensor(testDataset)
	if err != nil {
		t.Fatalf("error creating dataset tensor: %v", err)
	}
	defer dataset.Close()

	// Build CAGRA index
	cagraIndexParams, err := cagra.CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating CAGRA index params: %v", err)
	}
	defer cagraIndexParams.Close()

	cagraIndex, err := cagra.CreateIndex()
	if err != nil {
		t.Fatalf("error creating CAGRA index: %v", err)
	}
	defer cagraIndex.Close()

	if _, err := dataset.ToDevice(&resource); err != nil {
		t.Fatalf("error moving dataset to device: %v", err)
	}

	if err := cagra.BuildIndex(resource, cagraIndexParams, &dataset, cagraIndex); err != nil {
		t.Fatalf("error building CAGRA index: %v", err)
	}

	// Create and convert to HNSW index
	hnswIndexParams, err := hnsw.CreateIndexParams()
	if err != nil {
		t.Fatalf("error creating HNSW index params: %v", err)
	}
	defer hnswIndexParams.Close()

	// Using CPU hierarchy for this test
	if _, err := hnswIndexParams.SetHierarchy(hnsw.HierarchyCPU); err != nil {
		t.Fatalf("error setting HNSW hierarchy: %v", err)
	}

	hnswIndex, err := hnsw.CreateIndex()
	if err != nil {
		t.Fatalf("error creating HNSW index: %v", err)
	}
	defer hnswIndex.Close()

	if err := hnsw.FromCagra(resource, hnswIndexParams, cagraIndex, hnswIndex); err != nil {
		t.Fatalf("error converting CAGRA to HNSW: %v", err)
	}

	// Serialize the index to a temporary file
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "hnsw_index.bin")
	if err := hnsw.SerializeIndex(resource, tempFile, hnswIndex); err != nil {
		t.Fatalf("error serializing HNSW index: %v", err)
	}

	// Create a new HNSW index for deserialization
	deserializedIndex, err := hnsw.CreateIndex()
	if err != nil {
		t.Fatalf("error creating deserialized HNSW index: %v", err)
	}
	defer deserializedIndex.Close()

	hnsw.SetDType[float32](deserializedIndex)

	// Deserialize the index
	if err := hnsw.DeserializeIndex(
		resource,
		hnswIndexParams,
		tempFile,
		nFeatures,
		cuvs.DistanceL2, // Use the appropriate metric that was used to build the index
		deserializedIndex,
	); err != nil {
		t.Fatalf("error deserializing HNSW index: %v", err)
	}

	// Test the deserialized index with queries
	queries, err := cuvs.NewTensor(testDataset[:nQueries])
	if err != nil {
		t.Fatalf("error creating queries tensor: %v", err)
	}
	defer queries.Close()

	// Create neighbors tensor (changed from uint32 to uint64 and on-host)
	neighborsInputSlice := make([][]uint64, nQueries)
	for i := range neighborsInputSlice {
		neighborsInputSlice[i] = make([]uint64, k)
	}

	neighbors, err := cuvs.NewTensor[uint64](neighborsInputSlice)
	if err != nil {
		t.Fatalf("error creating neighbors tensor: %v", err)
	}
	defer neighbors.Close()

	// Create distances tensor (changed to on-host)
	distancesInputSlice := make([][]float32, nQueries)
	for i := range distancesInputSlice {
		distancesInputSlice[i] = make([]float32, k)
	}

	distances, err := cuvs.NewTensor[float32](distancesInputSlice)
	if err != nil {
		t.Fatalf("error creating distances tensor: %v", err)
	}
	defer distances.Close()

	searchParams, err := hnsw.CreateSearchParams()
	if err != nil {
		t.Fatalf("error creating search params: %v", err)
	}
	defer searchParams.Close()

	// Search the deserialized index
	err = hnsw.SearchIndex(resource, searchParams, deserializedIndex, &queries, &neighbors, &distances)
	if err != nil {
		t.Fatalf("error searching deserialized HNSW index: %v", err)
	}

	if err := resource.Sync(); err != nil {
		t.Fatalf("error syncing resource: %v", err)
	}

	// Validate results
	neighborsSlice, err := neighbors.Slice()
	if err != nil {
		t.Fatalf("error getting neighbors slice: %v", err)
	}

	// The queries should find themselves as the top neighbors
	for i := range neighborsSlice {
		if neighborsSlice[i][0] != uint64(i) {
			t.Error("wrong neighbor from deserialized index, expected", i, "got", neighborsSlice[i][0])
		}
	}
}
