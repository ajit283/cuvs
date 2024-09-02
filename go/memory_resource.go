package cuvs

// #include <cuda_runtime_api.h>
// #include <cuvs/core/c_api.h>
// #include <cuvs/distance/pairwise_distance.h>
// #include <cuvs/neighbors/brute_force.h>
// #include <cuvs/neighbors/ivf_flat.h>
// #include <cuvs/neighbors/cagra.h>
// #include <cuvs/neighbors/ivf_pq.h>
import "C"

func EnablePoolMemoryResource(initial_pool_size_percent int, max_pool_size_percent int, managed bool) error {

	managed_c := C.int(0)
	if managed {
		managed_c = C.int(1)
	}
	return CheckCuvs(CuvsError(C.cuvsRMMPoolMemoryResourceEnable(C.int(initial_pool_size_percent), C.int(max_pool_size_percent), managed_c)))
}

func ResetMemoryResource() error {
	return CheckCuvs(CuvsError(C.cuvsRMMMemoryResourceReset()))
}
