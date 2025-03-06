package hnsw

// #include <cuvs/neighbors/hnsw.h>
import "C"

import (
	"errors"

	cuvs "github.com/rapidsai/cuvs/go"
)

// Parameters for building a HNSW index
type IndexParams struct {
	params C.cuvsHnswIndexParams_t
}

// Creates a new IndexParams with default values
func CreateIndexParams() (*IndexParams, error) {
	var params C.cuvsHnswIndexParams_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswIndexParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &IndexParams{params: params}, nil
}

// Destroys the IndexParams
func (p *IndexParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswIndexParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}

// Sets the hierarchy type for the HNSW index
func (p *IndexParams) SetHierarchy(hierarchy HnswHierarchy) (*IndexParams, error) {
	CHierarchy := uint32(C.NONE)

	switch hierarchy {
	case HierarchyNone:
		CHierarchy = C.NONE
	case HierarchyCPU:
		CHierarchy = C.CPU
	case HierarchyGPU:
		CHierarchy = C.GPU
	default:
		return nil, errors.New("unsupported hierarchy")
	}

	p.params.hierarchy = CHierarchy
	return p, nil
}

// Sets the ef_construction parameter for CPU-based hierarchy construction
func (p *IndexParams) SetEfConstruction(ef int) (*IndexParams, error) {
	p.params.ef_construction = C.int(ef)
	return p, nil
}

// Sets the number of threads to use for HNSW construction
// When set to 0, the maximum number of available threads will be used
func (p *IndexParams) SetNumThreads(numThreads int) (*IndexParams, error) {
	p.params.num_threads = C.int(numThreads)
	return p, nil
}
