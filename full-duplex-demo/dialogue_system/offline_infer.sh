script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname "$script_dir")

cd ${root_dir}
python dialogue_system/offline_infer.py \
    --config_path ${root_dir}/config/config.yaml \
    --eval_dirs "Full-Duplex-Bench-zh" \
    --prefix "clean_" \
    # --streaming_tts