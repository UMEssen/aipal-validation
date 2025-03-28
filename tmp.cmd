python -m aipal_validation --step test --root_dir /nvme/merengelke/aipal

python -m aipal_validation --step sampling --root_dir /data
python -m aipal_validation --step test --root_dir /data

python -m aipal_validation --step test --root_dir /data --eval_all

# train outlier
python -m aipal_validation --step train --root_dir /data --config aipal_validation/config/config_outlier.yaml

# test outlier
python -m aipal_validation --sample aipal_validation/config/sample.json --model_dir aipal_validation/outlier --config aipal_validation/config/config_outlier.yaml
