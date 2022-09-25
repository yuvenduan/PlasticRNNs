#!/bin/bash
for jobid in {25811063..25811160}
do
    scancel $jobid
done