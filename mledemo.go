package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/diff/fd"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"
)

func main() {

	gaussian := distuv.Normal{
		Mu:    5.0,
		Sigma: 3.0,
	}

	data := make([]float64, 200)
	for i := range data {
		data[i] = gaussian.Rand()
	}

	theoreticalMean := stat.Mean(data, nil)

	theoreticalSigma := stat.StdDev(data, nil)

	fmt.Println("theoretical mu:", theoreticalMean, " theoretical sigma:", theoreticalSigma)

	logLL := func(x []float64) float64 {
		dist := distuv.Normal{
			Mu:    x[0],
			Sigma: x[1],
		}
		var logLikelihood float64
		for _, pt := range data {
			logLikelihood += dist.LogProb(pt)
		}

		return -logLikelihood

	}

	mle := optimize.Problem{
		Func: logLL,
		Grad: func(grad []float64, x []float64) {
			_ = fd.Gradient(grad, logLL, x, nil)
		},
	}

	result, err := optimize.Minimize(mle, []float64{1, 1}, nil, &optimize.LBFGS{
		GradStopThreshold: 1e-5,
	})

	log.Println("result:", result.X)

	if err != nil {
		log.Fatal(err)
	}

}
