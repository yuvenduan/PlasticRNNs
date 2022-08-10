#!/bin/bash
for jobid in {25507079..25507094}
do
    scancel $jobid
done