package hnsw

// #include <stdlib.h>
// #include <dlpack/dlpack.h>
// #include <cuvs/neighbors/hnsw.h>
import "C"

import (
	"errors"
	"unsafe"

	cuvs "github.com/rapidsai/cuvs/go"
	"github.com/rapidsai/cuvs/go/cagra"
)

// HNSW ANN Index
type HnswIndex struct {
	index   C.cuvsHnswIndex_t
	trained bool
}

// Hierarchy for HNSW index when converting from CAGRA index
type HnswHierarchy int

const (
	// Flat hierarchy, search is base-layer only
	HierarchyNone HnswHierarchy = iota
	// Full hierarchy is built using the CPU
	HierarchyCPU
	// Full hierarchy is built using the GPU
	HierarchyGPU
)

// Creates a new empty HNSW Index
func CreateIndex() (*HnswIndex, error) {
	var index C.cuvsHnswIndex_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswIndexCreate(&index)))
	if err != nil {
		return nil, err
	}

	return &HnswIndex{index: index}, nil
}

func SetDType[T cuvs.TensorNumberType](index *HnswIndex) error {
	var zero T
	switch any(zero).(type) {
	case int64:
		index.index.dtype = C.DLDataType{
			bits:  C.uchar(64),
			lanes: C.ushort(1),
			code:  C.kDLInt,
		}
		return nil
	case uint64:
		index.index.dtype = C.DLDataType{
			bits:  C.uchar(64),
			lanes: C.ushort(1),
			code:  C.kDLUInt,
		}
		return nil
	case uint32:
		index.index.dtype = C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLUInt,
		}
		return nil
	case float32:
		index.index.dtype = C.DLDataType{
			bits:  C.uchar(32),
			lanes: C.ushort(1),
			code:  C.kDLFloat,
		}
		return nil
	}
	panic("unreachable")
}

// Destroys the HNSW Index
func (index *HnswIndex) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswIndexDestroy(index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Convert a CAGRA Index to an HNSW index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - HNSW index parameters
// * `cagraIndex` - Cagra index to convert
// * `hnswIndex` - HNSW index to return
func FromCagra[T any](Resources cuvs.Resource, params *IndexParams, cagraIndex *cagra.CagraIndex, hnswIndex *HnswIndex) error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswFromCagra(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		(C.cuvsCagraIndex_t)(unsafe.Pointer(*cagraIndex.GetIndex())),
		hnswIndex.index,
	)))
	if err != nil {
		return err
	}

	hnswIndex.trained = true
	return nil
}

// Add new vectors to an HNSW index
// NOTE: The HNSW index can only be extended when the hierarchy is CPU
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters for extending the index
// * `additionalDataset` - A tensor in device memory with additional vectors
// * `index` - HNSW index to extend
func ExtendIndex[T any](Resources cuvs.Resource, params *ExtendParams, additionalDataset *cuvs.Tensor[T], index *HnswIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before calling extend")
	}
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswExtend(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		(*C.DLManagedTensor)(unsafe.Pointer(additionalDataset.C_tensor)),
		index.index)))
	if err != nil {
		return err
	}
	return nil
}

// Perform a search on the HNSW Index
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - Parameters to use in searching the index
// * `index` - HNSW index to search
// * `queries` - A tensor in device memory to query for
// * `neighbors` - Tensor in device memory that receives the indices of the nearest neighbors
// * `distances` - Tensor in device memory that receives the distances of the nearest neighbors
func SearchIndex[T any](Resources cuvs.Resource, params *SearchParams, index *HnswIndex, queries *cuvs.Tensor[T], neighbors *cuvs.Tensor[uint64], distances *cuvs.Tensor[T]) error {
	if !index.trained {
		return errors.New("index needs to be built before calling search")
	}
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswSearch(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		index.index,
		(*C.DLManagedTensor)(unsafe.Pointer(queries.C_tensor)),
		(*C.DLManagedTensor)(unsafe.Pointer(neighbors.C_tensor)),
		(*C.DLManagedTensor)(unsafe.Pointer(distances.C_tensor)))))
}

// Serialize a HNSW index to a file
//
// # Arguments
//
// * `Resources` - Resources to use
// * `filename` - Path to save the index
// * `index` - HNSW index to serialize
func SerializeIndex(Resources cuvs.Resource, filename string, index *HnswIndex) error {
	if !index.trained {
		return errors.New("index needs to be built before serializing")
	}
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	return cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswSerialize(
		C.cuvsResources_t(Resources.Resource),
		cFilename,
		index.index)))
}

// Deserialize a HNSW index from a file
//
// # Arguments
//
// * `Resources` - Resources to use
// * `params` - HNSW index parameters
// * `filename` - Path to the serialized index
// * `dim` - Dimension of the vectors in the index
// * `metric` - Distance metric used to build the index
// * `index` - HNSW index to load into
func DeserializeIndex(Resources cuvs.Resource, params *IndexParams, filename string, dim int, metric cuvs.Distance, index *HnswIndex) error {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswDeserialize(
		C.cuvsResources_t(Resources.Resource),
		params.params,
		cFilename,
		C.int(dim),
		C.cuvsDistanceType(metric),
		index.index)))
	if err != nil {
		return err
	}
	index.trained = true
	return nil
}

// Helper method to get the underlying C index
func (index *HnswIndex) GetIndex() C.cuvsHnswIndex_t {
	return index.index
}
