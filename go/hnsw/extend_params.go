package hnsw

// #include <cuvs/neighbors/hnsw.h>
import "C"

import (
	cuvs "github.com/rapidsai/cuvs/go"
)

// Parameters for extending a HNSW index
type ExtendParams struct {
	params C.cuvsHnswExtendParams_t
}

// Creates a new ExtendParams with default values
func CreateExtendParams() (*ExtendParams, error) {
	var params C.cuvsHnswExtendParams_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswExtendParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &ExtendParams{params: params}, nil
}

// Destroys the ExtendParams
func (p *ExtendParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswExtendParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}

// Sets the number of threads used to extend additional vectors
func (p *ExtendParams) SetNumThreads(numThreads int) (*ExtendParams, error) {
	p.params.num_threads = C.int(numThreads)
	return p, nil
}
