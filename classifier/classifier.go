package classifier

import "fmt"

// type Attribute struct {
// 	name          string
// 	dtp           arff.DataType
// 	nominalValues []string
// 	frequencies   map[interface{}]map[string]int
// 	probabilities map[interface{}]map[string]float64
// }

// func NewAttribute(name string, dtp arff.DataType, values []string) (Attribute, error) {
// 	if dtp == arff.DataTypeNominal && len(values) == 0 {
// 		return Attribute{}, fmt.Errorf("no nominal values")
// 	}
// 	return Attribute{
// 		name:          name,
// 		dtp:           dtp,
// 		nominalValues: values,
// 	}, nil
// }

// func (a *Attribute) IncrFreq(class string, value interface{}) error {
// 	_, ok := a.frequencies[class]
// 	if !ok {
// 		log.Println("freq", a.frequencies)
// 		return fmt.Errorf("no class %s", class)
// 	}
// 	a.frequencies[class][value]++
// 	return nil
// }

// func (a *Attribute) CalculateProbabilities() error {
// 	pClass := 1 / len(a.frequencies)

// 	return nil
// }

type Naive struct {
	relationName string
	classes      map[string]int
	probClass    float64
	attributesF  []map[int]map[string]int
	attributesP  []map[int]map[string]float64
}

// func (n *Naive) IncrFreq(attribute int, class string, value interface{}) error {
// 	return n.attributes[attribute].IncrFreq(class, value)
// }

func NewNaive(name string, classes []string, numAttributes int) (Naive, error) {
	res := Naive{relationName: name}
	classesMap := map[string]int{}
	for _, class := range classes {
		classesMap[class] = 0
	}
	res.classes = classesMap
	res.attributesF = make([]map[int]map[string]int, numAttributes)
	res.attributesP = make([]map[int]map[string]float64, numAttributes)
	res.probClass = 1 / float64(len(classes))
	return res, nil
}

func (n *Naive) IncreFreq(attribute int, attribVal int, class string) error {
	if n.attributesF[attribute] == nil {
		n.attributesF[attribute] = make(map[int]map[string]int)
	}
	if n.attributesF[attribute][attribVal] == nil {
		n.attributesF[attribute][attribVal] = make(map[string]int)
		for k := range n.classes {
			n.attributesF[attribute][attribVal][k] = 0
		}
	}

	n.attributesF[attribute][attribVal][class]++
	return nil
}

func (n *Naive) Debug() {
	fmt.Println("class prob:", n.probClass)
	for i, a := range n.attributesF {
		fmt.Println("attrib num", i)
		fmt.Println(a)
	}
	for i, a := range n.attributesP {
		fmt.Println("attrib num", i)
		fmt.Println(a)
	}
}

func (n *Naive) Build() {
	for i, attrib := range n.attributesF {
		n.attributesP[i] = make(map[int]map[string]float64)
		for k, attribVal := range attrib {
			n.attributesP[i][k] = make(map[string]float64)
			var total float64 = 0
			for _, count := range attribVal {
				total += float64(count)
			}
			for class, count := range attribVal {
				probDenorm := float64(count) / total
				n.attributesP[i][k][class] = (total*probDenorm + n.probClass) / (total + 1)
			}

		}
	}
}

func (n *Naive) Classify(values []int) (string, error) {
	if len(values) != len(n.attributesP) {
		return "", fmt.Errorf("not enough values")
	}
	probs := make(map[string]float64)

	for k := range n.classes {
		probs[k] = n.probClass
	}

	for i, v := range values {
		for k := range n.classes {
			_, ok := n.attributesP[i][v]
			if !ok {
				probs[k] *= n.probClass
			} else {
				probs[k] *= n.attributesP[i][v][k]
			}

		}

	}
	var max float64 = 0
	var maxClass string = ""
	for k, v := range probs {
		if v > max {
			max = v
			maxClass = k
		}
	}
	return maxClass, nil
}
