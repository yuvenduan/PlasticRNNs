#!/bin/bash
cd ..
rm -r sbatch
cd experiments
rm -r !("MNIST"|"CNN_calssification")
cd ..
