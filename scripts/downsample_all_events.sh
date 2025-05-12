#!/bin/bash

#!/bin/bash

# 定义根目录
DSEC_ROOT="/home/handsomexd/EventAD/data/detector/ROL"

# 遍历 val 文件夹
for split in val; do
    for sequence in $DSEC_ROOT/$split/*/; do
        infile=$sequence/events/left/events.h5
        outfile=$sequence/events/left/events_2x.h5

        # 输出处理的信息
        echo "Processing sequence: $sequence"

        # 运行 Python 脚本进行 downsample
        python /home/handsomexd/EventAD/scripts/downsample_events.py --input_path $infile --output_path $outfile

    done
done