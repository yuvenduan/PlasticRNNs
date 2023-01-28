#!/bin/bash
for jobid in {27744436..27744459}
do
    scancel $jobid
done