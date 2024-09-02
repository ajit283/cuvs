package cuvs_test

import (
	"math/rand"
	"testing"
	"time"

	cuvs "github.com/rapidsai/cuvs/go"
)

func TestDlPack(t *testing.T) {

	resource, _ := cuvs.NewResource(nil)

	rand.Seed(time.Now().UnixNano())

	NDataPoints := 2560
	NFeatures := 16

	TestDataset := make([][]float32, NDataPoints)
	for i := range TestDataset {
		TestDataset[i] = make([]float32, NFeatures)
		for j := range TestDataset[i] {
			// TestDataset[i][j] = rand.Float32()
			TestDataset[i][j] = float32(i)
		}
	}

	dataset, err := cuvs.NewTensor(true, TestDataset, true)

	if err != nil {
		panic(err)
	}

	_, err = dataset.ToDevice(&resource)

	if err != nil {
		panic(err)
	}

	// _, err = dataset.Expand(&resource, TestDataset[127:])

	// if err != nil {
	// 	panic(err)
	// }

	println(dataset.GetShape()[1])

	// p := (*int64)(unsafe.Pointer(uintptr(neighbors.c_tensor.dl_tensor.data) + uintptr(K*8*3)))
	dataset.ToHost(&resource)
	arr, err := dataset.GetArray()
	if err != nil {
		panic(err)
	}
	println(arr)
	for i := range arr {
		for j := range arr[i] {

			if arr[i][j] != TestDataset[i][j] {
				t.Error("wrong neighbor, expected", i, "got", arr[i][j])
			}
		}

		// arr_dist, _ := distances.GetArray()
		// for i := range arr_dist {
		// 	if arr_dist[i][0] >= float32(0.001) || arr_dist[i][0] <= float32(-0.001) {
		// 		t.Error("wrong distance, expected", float32(i), "got", arr_dist[i][0])
		// 	}
		// }

	}
}
