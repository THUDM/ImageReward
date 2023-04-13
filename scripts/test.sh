#! /bin/bash

# Download dataset.
cd data
if [ ! -d "test_images" ]; then
    wget -c "https://cloud.tsinghua.edu.cn/f/9bd245027652422499f4/?dl=1" -O test_images.zip
    unzip test_images.zip
fi
cd ..

# test
run_cmd="python test.py"

echo ${run_cmd}
eval ${run_cmd}
set +x