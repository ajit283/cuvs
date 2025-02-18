package dynamic_batching

// #include <cuvs/neighbors/dynamic_batching.h>
import "C"

import (
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
	cagra "github.com/rapidsai/cuvs/go/cagra"
)

// DynamicBatchingIndex wraps a cuvsDynamicBatchingIndex_t.
type DynamicBatchingIndex struct {
	index   C.cuvsDynamicBatchingIndex_t
	trained bool
}

// IndexParams wraps dynamic batching index parameters.
type IndexParams struct {
	params C.cuvsDynamicBatchingIndexParams_t
}

// CreateIndexParams allocates default dynamic batching index parameters.
func CreateIndexParams() (*IndexParams, error) {
	var params C.cuvsDynamicBatchingIndexParams_t
	if err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingIndexParamsCreate(&params))); err != nil {
		return nil, err
	}
	return &IndexParams{params: params}, nil
}

// Close releases the dynamic batching index parameters.
func (p *IndexParams) Close() error {
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingIndexParamsDestroy(p.params)))
}

// SearchParams wraps dynamic batching search parameters.
type SearchParams struct {
	params C.cuvsDynamicBatchingSearchParams_t
}

// CreateSearchParams allocates default dynamic batching search parameters.
func CreateSearchParams() (*SearchParams, error) {
	var params C.cuvsDynamicBatchingSearchParams_t
	if err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingSearchParamsCreate(&params))); err != nil {
		return nil, err
	}
	return &SearchParams{params: params}, nil
}

// Close releases the dynamic batching search parameters.
func (p *SearchParams) Close() error {
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingSearchParamsDestroy(p.params)))
}

// CreateIndex creates a dynamic batching index from an upstream (Cagra) index.
// The upstream index must already be built (trained). The provided filter (if non-nil)
// will be passed to the underlying API.
func CreateIndex(res cuvs.Resource, params *IndexParams, upstream *cagra.CagraIndex, allowList []uint32) (*DynamicBatchingIndex, error) {
	// Build the dynamic batching upstream structure.
	var dbUpstream C.cuvsDynamicBatchingUpstream_t
	// In this example we assume the upstream is a CAGRA index.
	dbUpstream._type = C.CAGRA
	dbUpstream.cagra.index = upstream.Index()                // e.g. returns C.cuvsCagraIndex_t
	dbUpstream.cagra.search_params = upstream.SearchParams() // likewise

	var filter C.cuvsFilter
	bitset := createBitset(allowList)
	allowListTensor, err := cuvs.NewVector[uint32](bitset)
	if err != nil {
		return err
	}
	defer allowListTensor.Close()
	_, err = allowListTensor.ToDevice(&Resources)
	if err != nil {
		return err
	}
	if allowList == nil {
		filter = C.cuvsFilter{
			_type: C.NO_FILTER,
			addr:  C.uintptr_t(0),
		}
	} else {
		filter = C.cuvsFilter{
			_type: C.BITSET,
			addr:  C.uintptr_t(uintptr(unsafe.Pointer(allowListTensor.C_tensor))),
		}
	}

	var dbIndex C.cuvsDynamicBatchingIndex_t
	if err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingIndexCreate(
		C.cuvsResources_t(res.Resource),
		&dbIndex,
		params.params,
		&dbUpstream,
		filter,
	))); err != nil {
		return nil, err
	}
	return &DynamicBatchingIndex{
		index:   dbIndex,
		trained: true,
	}, nil
}

func createBitset(allowList []uint32) []uint32 {
	// Calculate size needed for the bitset array
	// Each uint32 handles 32 bits, so we divide the max ID by 32 (shift right by 5)
	maxID := uint32(0)
	for _, id := range allowList {
		if id > maxID {
			maxID = id
		}
	}
	size := (maxID >> 5) + 1 // Division by 32, add 1 to handle remainder
	bitset := make([]uint32, size)
	for _, id := range allowList {
		// Calculate which uint32 in our array (divide by 32)
		arrayIndex := id >> 5
		// Calculate bit position within that uint32 (mod 32)
		bitPosition := id & 31 // equivalent to id % 32
		// Set the bit
		bitset[arrayIndex] |= 1 << bitPosition
	}
	return bitset
}

// SearchIndex performs a dynamic batching search on the index.
// The provided tensors (queries, neighbors, distances) must be allocated on device.
func (db *DynamicBatchingIndex) SearchIndex(res cuvs.Resource, searchParams *SearchParams, queries, neighbors, distances *cuvs.Tensor[float32]) error {
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingSearch(
		C.cuvsResources_t(res.Resource),
		searchParams.params,
		db.index,
		(*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)),
		(*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)),
		(*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)),
	)))
}

// Close destroys the dynamic batching index.
func (db *DynamicBatchingIndex) Close() error {
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsDynamicBatchingIndexDestroy(db.index)))
}
