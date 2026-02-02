cd ../src

# zhuang to chinese
python eval.py \
        --output_path ../output/Qwen3-32B_za2zh_exact138.jsonl  \
        --lang zh \
        --leveled

# chinese to zhuang
python eval.py \
        --output_path ../output/Qwen3-32B_zh2za_d.jsonl  \
        --lang mn \
        --leveled
