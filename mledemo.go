package main

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat/distuv"
	"gorgonia.org/gorgonia"
)

func main() {

	gaussian := distuv.Normal{
		Mu:    5.0,
		Sigma: 3.0,
	}

	var counter int

	data := make([]float64, 200)
	for i := range data {
		data[i] = gaussian.Rand()
	}

	fmt.Println("data:", data)

	mle := optimize.Problem{
		Func: func(x []float64) float64 {
			dist := distuv.Normal{
				Mu:    x[0],
				Sigma: x[1],
			}
			var logLikelihood float64
			for _, pt := range data {
				logLikelihood += dist.LogProb(pt)
			}

			fmt.Println("value in Func:", -logLikelihood)
			return -logLikelihood

		},
		Grad: func(grad []float64, x []float64) {

			counter += 1
			g := gorgonia.NewGraph()
			mu := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("mu"))
			sig := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("sig"))

			logLikeliHoodTerm := func(point float64) *gorgonia.Node {
				gauss_const := -0.5 * (math.Log(2.0) + math.Log(math.Pi))
				data_pt := gorgonia.NewConstant(point)
				gauss_coeff := gorgonia.NewConstant(gauss_const)

				diff_term, _ := gorgonia.Sub(data_pt, mu)
				diff_term_by_sig, _ := gorgonia.Div(diff_term, sig)

				square_term, _ := gorgonia.Square(diff_term_by_sig)

				square_term, _ = gorgonia.Mul(gorgonia.NewConstant(-0.5), square_term)

				log_sig, _ := gorgonia.Log(sig)
				neg_log_sig, _ := gorgonia.Mul(gorgonia.NewConstant(-1.0), log_sig)

				nodes := gorgonia.Nodes([]*gorgonia.Node{gauss_coeff, square_term, neg_log_sig})

				logLikelihood, _ := gorgonia.ReduceAdd(nodes)

				return logLikelihood

			}

			loglikelihood := logLikeliHoodTerm(data[0])

			for _, pt := range data[1:] {
				loglikelihood, _ = gorgonia.Add(loglikelihood, logLikeliHoodTerm(pt))
			}

			negloglikelihood, err := gorgonia.Mul(gorgonia.NewConstant(-1.0), loglikelihood)
			gorgonia.Let(mu, x[0])
			gorgonia.Let(sig, x[1])
			m := gorgonia.NewLispMachine(g)
			defer m.Close()

			err = m.RunAll()

			if err != nil {
				log.Fatal(err)
			}
			fmt.Println("value:", negloglikelihood.Value())
			mu_grad, _ := mu.Grad()
			sig_grad, _ := sig.Grad()

			log.Println("mu_grad:", mu_grad)
			log.Println("sig_grad:", sig_grad)

			grad[0] = mu_grad.Data().(float64)
			grad[1] = sig_grad.Data().(float64)

			log.Println("grad:", grad)

			log.Println("params:counter=", counter, " values=", x)

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
