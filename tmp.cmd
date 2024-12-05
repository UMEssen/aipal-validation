python -m aipal_validation --step test --root_dir /nvme/merengelke/aipal

python -m aipal_validation --step sampling --root_dir /data
python -m aipal_validation --step test --root_dir /data

python -m aipal_validation --step test --root_dir /data --eval_all
