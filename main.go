package main

import (
	"fmt"
	"log"
	"os"

	"github.com/asstronom/IADlab2/classifier"
	"github.com/bsm/arff"
)

func main() {
	var err error
	_, err = os.Open("./breast.w.arff")
	if err != nil {
		log.Fatalln("error opening text file", err)
	}
	data, err := arff.Open("./breast.w.arff")
	defer data.Close()
	if err != nil {
		log.Fatalln("error opening arff file", err)
	}
	fmt.Println(data.Relation.Name)
	fmt.Println(data.Relation.Attributes)

	classes := []string{}
	for _, v := range data.Relation.Attributes[len(data.Relation.Attributes)-1].NominalValues {
		if v == "" {
			continue
		}
		classes = append(classes, v)
	}

	//fmt.Println(data.Relation.Name, classes, attributes)

	naive, err := classifier.NewNaive(data.Relation.Name, classes, len(data.Relation.Attributes)-1)
	if err != nil {
		log.Fatalln("error creating naive", err)
	}

	// err = naive.IncrFreq(0, "benign", 0)
	// if err != nil {
	// 	log.Fatalln("error incrementing", err)
	// }

	for i := 0; i < 300; i++ {
		if data.Next() {
			//fmt.Println(data.Row().Values)
			curClass := data.Row().Values[len(data.Row().Values)-1]
			for i2, v := range data.Row().Values {
				if i2 == len(data.Row().Values)-1 {
					break
				}
				err := naive.IncreFreq(i2, int(v.(float64)), curClass.(string))
				if err != nil {
					log.Fatalln("error incrementing", err)
				}
			}
		} else {
			break
		}
	}

	naive.Build()

	var count int
	var total int
	for i := 0; i < 100; i++ {
		if data.Next() {
			total++
			valInter := data.Row().Values[0 : len(data.Row().Values)-1]
			valInt := make([]int, len(valInter))
			for i2, v := range valInter {
				valInt[i2] = int(v.(float64))
			}
			response, err := naive.Classify(valInt)
			if err != nil {
				log.Fatalln("error classifying")
			}

			if response == data.Row().Values[len(data.Row().Values)-1] {
				fmt.Println(i, data.Row().Values[len(data.Row().Values)-1], response)
				count++
			} else {
				log.Println(i, data.Row().Values[len(data.Row().Values)-1], response)
				log.Println(data.Row().Values)
			}
		}
	}

	fmt.Println("Precision:", float64(count)/float64(total))

	//naive.Debug()

}
