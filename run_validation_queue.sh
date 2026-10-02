#!/usr/bin/env bash

# ============================================================
# Config
# ============================================================

MAX_PARALLEL=2
POLL_SECONDS=5

mkdir -p logs


# ============================================================
# Jobs
#
# 格式：
# session_name|script|adapter_path|output_path
# ============================================================

jobs=(
"s2_seed43_epoch1|run_occurrence_validation.py|checkpoints/s2_occurrence_lora_seed44/epoch_1|outputs/s2_seed44_epoch1_validation_predictions.jsonl"

"s2_seed43_epoch2|run_occurrence_validation.py|checkpoints/s2_occurrence_lora_seed44/epoch_2|outputs/s2_seed44_epoch2_validation_predictions.jsonl"

"s2_seed43_epoch3|run_occurrence_validation.py|checkpoints/s2_occurrence_lora_seed43/epoch_3|outputs/s2_seed43_epoch3_validation_predictions.jsonl"

"s2_seed44_epoch1|run_occurrence_validation.py|checkpoints/s2_occurrence_lora_seed44/epoch_1|outputs/s2_seed44_epoch1_validation_predictions.jsonl"

"s2_seed44_epoch2|run_occurrence_validation.py|checkpoints/s2_occurrence_lora_seed44/epoch_2|outputs/s2_seed44_epoch2_validation_predictions.jsonl"

"s2_seed44_epoch3|run_occurrence_validation.py|checkpoints/s2_occurrence_lora_seed44/epoch_3|outputs/s2_seed44_epoch3_validation_predictions.jsonl"

"s3_seed43_epoch1|run_grouped_occurrence_validation.py|checkpoints/s3_grouped_occurrence_lora_seed43/epoch_1|outputs/s3_seed43_epoch1_validation_predictions.jsonl"

"s3_seed43_epoch2|run_grouped_occurrence_validation.py|checkpoints/s3_grouped_occurrence_lora_seed43/epoch_2|outputs/s3_seed43_epoch2_validation_predictions.jsonl"

"s3_seed43_epoch3|run_grouped_occurrence_validation.py|checkpoints/s3_grouped_occurrence_lora_seed43/epoch_3|outputs/s3_seed43_epoch3_validation_predictions.jsonl"

"s3_seed44_epoch1|run_grouped_occurrence_validation.py|checkpoints/s3_grouped_occurrence_lora_seed44/epoch_1|outputs/s3_seed44_epoch1_validation_predictions.jsonl"

"s3_seed44_epoch2|run_grouped_occurrence_validation.py|checkpoints/s3_grouped_occurrence_lora_seed44/epoch_2|outputs/s3_seed44_epoch2_validation_predictions.jsonl"

"s3_seed44_epoch3|run_grouped_occurrence_validation.py|checkpoints/s3_grouped_occurrence_lora_seed44/epoch_3|outputs/s3_seed44_epoch3_validation_predictions.jsonl"

)


# ============================================================
# Count currently running validation processes
# ============================================================

count_running()
{
    ps -eo comm=,args= \
    | awk '
        $1 ~ /^python/ && ($0 ~ /run_occurrence_validation\.py/ || $0 ~ /run_grouped_occurrence_validation\.py/) {
            count++
        }

        END {
            print count + 0
        }
    '
}


# ============================================================
# Launch one job in its own tmux session
# ============================================================

launch_job()
{
    session_name="$1"
    script="$2"
    adapter_path="$3"
    output_path="$4"

    log_path="logs/${session_name}.log"


    # 防止误覆盖已经存在的结果
    if [ -e "$output_path" ]; then

        echo
        echo "[SKIP]"
        echo "Output already exists:"
        echo "$output_path"
        echo

        return
    fi


    # 防止创建同名 tmux session
    if tmux has-session \
        -t "$session_name" \
        2>/dev/null
    then

        echo
        echo "[SKIP]"
        echo "tmux session already exists:"
        echo "$session_name"
        echo

        return
    fi


    echo
    echo "============================================================"
    echo "Launching:"
    echo "session : $session_name"
    echo "adapter : $adapter_path"
    echo "output  : $output_path"
    echo "log     : $log_path"
    echo "============================================================"
    echo


    tmux new-session \
        -d \
        -s "$session_name" \
        "bash -lc '
            set -o pipefail

            HF_HUB_OFFLINE=1 \
            PYTHONUNBUFFERED=1 \
            python \"$script\" \
                --adapter_path \"$adapter_path\" \
                --output_path \"$output_path\" \
                2>&1 \
                | tee \"$log_path\"

            status=\$?

            echo
            echo \"[FINISHED] $session_name exit_code=\$status\" \
                | tee -a \"$log_path\"

            exit \$status
        '"
}


# ============================================================
# Queue
# ============================================================

echo
echo "============================================================"
echo "Validation queue started"
echo "Maximum parallel inference jobs: $MAX_PARALLEL"
echo "============================================================"
echo


for job in "${jobs[@]}"
do

    IFS="|" read -r \
        session_name \
        script \
        adapter_path \
        output_path \
        <<< "$job"


    while true
    do

        running=$(
            count_running
        )


        if [ "$running" -lt "$MAX_PARALLEL" ]; then
            break
        fi


        echo \
            "[WAIT] running inference jobs: " \
            "$running / $MAX_PARALLEL"

        sleep "$POLL_SECONDS"

    done


    launch_job \
        "$session_name" \
        "$script" \
        "$adapter_path" \
        "$output_path"


    # 给新进程一点启动时间，
    # 避免下一轮检查时 Python 进程尚未出现。
    sleep 2

done


# ============================================================
# Wait for final jobs
# ============================================================

echo
echo "All jobs have been submitted."
echo "Waiting for the remaining inference jobs..."
echo


while true
do

    running=$(
        count_running
    )


    if [ "$running" -eq 0 ]; then
        break
    fi


    echo \
        "[WAIT] remaining running jobs: " \
        "$running"

    sleep "$POLL_SECONDS"

done


echo
echo "============================================================"
echo "All validation inference jobs finished."
echo "============================================================"