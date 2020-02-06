#!/bin/bash
work_space=/home/cocoa_kang/no_where/
task_name=gen_result
python torch_render_test.py $work_space $task_name
echo "---------------------------"
echo "[TEST CORRECTION]" 
python compare_result.py $work_space