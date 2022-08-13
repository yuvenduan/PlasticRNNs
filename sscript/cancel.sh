#!/bin/bash
for jobid in {25600250..25600328}
do
    scancel $jobid
done