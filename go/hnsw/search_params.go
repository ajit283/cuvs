package hnsw

// #include <cuvs/neighbors/hnsw.h>
import "C"

import (
	cuvs "github.com/rapidsai/cuvs/go"
)

// Parameters for searching a HNSW index
type SearchParams struct {
	params C.cuvsHnswSearchParams_t
}

// Creates a new SearchParams with default values
func CreateSearchParams() (*SearchParams, error) {
	var params C.cuvsHnswSearchParams_t
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswSearchParamsCreate(&params)))
	if err != nil {
		return nil, err
	}

	return &SearchParams{params: params}, nil
}

// Destroys the SearchParams
func (p *SearchParams) Close() error {
	err := cuvs.CheckCuvs(cuvs.CuvsError(C.cuvsHnswSearchParamsDestroy(p.params)))
	if err != nil {
		return err
	}
	return nil
}

// Sets the ef parameter (size of the candidate list during search)
func (p *SearchParams) SetEf(ef int) (*SearchParams, error) {
	p.params.ef = C.int32_t(ef)
	return p, nil
}

// Sets the number of threads used for search
func (p *SearchParams) SetNumThreads(numThreads int) (*SearchParams, error) {
	p.params.num_threads = C.int32_t(numThreads)
	return p, nil
}
