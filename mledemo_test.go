package main

import "testing"

import "gorgonia.org/gorgonia"

func TestMulByNegOne(t *testing.T) {
	g := gorgonia.NewGraph()

	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))

	z, err := gorgonia.Add(x, y)

	if err != nil {
		t.Fatal("error adding:", err)
	}

	z, err = gorgonia.Mul(gorgonia.NewConstant(-1.0), z)

	if err != nil {
		t.Fatal("error in Multiply with -1:", err)
	}

	gorgonia.Let(x, 2.5)
	gorgonia.Let(y, 2.0)

	m := gorgonia.NewLispMachine(g)

	defer m.Close()

	err = m.RunAll()

	if err != nil {
		t.Fatal("error in running the lisp machine:", err)
	}

	t.Log("value of z:", z.Value())

	xgrad, err := x.Grad()

	if err != nil {
		t.Fatal("error in getting the xgrad:", err)
	}

	ygrad, err := y.Grad()

	if err != nil {
		t.Fatal("error in getting the ygrad:", err)
	}

	actualxgrad := xgrad.Data().(float64)

	if actualxgrad == 0.0 {
		t.Fatal("zero xgrad")
	}

	actualygrad := ygrad.Data().(float64)

	if actualygrad == 0.0 {
		t.Fatal("zero ygrad")
	}

	t.Log("xgrad=", actualxgrad, "ygrad=", actualygrad)

}

//test which fails due to presence of Neg operator
func TestNegOp(t *testing.T) {
	g := gorgonia.NewGraph()

	x := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y := gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))

	z, err := gorgonia.Add(x, y)

	if err != nil {
		t.Fatal("error adding:", err)
	}

	z, err = gorgonia.Neg(z)

	if err != nil {
		t.Fatal("error in Negating:", err)
	}

	// z, err = gorgonia.Mul(gorgonia.NewConstant(2.0), z)

	// if err != nil {
	// 	t.Fatal("error in Multiply by -1.0 :", err)
	// }

	gorgonia.Let(x, 2.5)
	gorgonia.Let(y, 2.0)

	m := gorgonia.NewLispMachine(g)

	defer m.Close()

	err = m.RunAll()

	if err != nil {
		t.Fatal("error in running the lisp machine:", err)
	}

	t.Log("value of z:", z.Value())

	xgrad, err := x.Grad()

	if err != nil {
		t.Fatal("error in getting the xgrad:", err)
	}

	ygrad, err := y.Grad()

	if err != nil {
		t.Fatal("error in getting the ygrad:", err)
	}

	actualxgrad := xgrad.Data().(float64)

	actualygrad := ygrad.Data().(float64)

	if actualxgrad == 0.0 {
		t.Log("xgrad=", actualxgrad, "ygrad=", actualygrad)
		t.Fatal("zero xgrad")
	}

	if actualygrad == 0.0 {
		t.Fatal("zero ygrad")
	}

	t.Log("xgrad=", actualxgrad, "ygrad=", actualygrad)

}
