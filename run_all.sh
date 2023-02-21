#!/bin/bash
function max5 {
   while [ `jobs | wc -l` -ge 5 ]
   do
      sleep 5
   done
}

for filename in /scratch2/2016-ITS-BrnoCompSpeed/dataset/*/; do
    [ -e "$filename" ] || continue
    [[ $filename != *session0* ]] || continue
    echo "Run $filename"
    max5; docker run --rm \
        --gpus '"device=1"' -v /home/fsauerwald/p2:/storage -v /scratch2:/scratch2 \
        -t cv-cuda python3 /storage/speed_estimation/speed_estimation.py "$filename" &
done