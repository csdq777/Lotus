export CUDA_VISIBLE_DEVICES=0,1

cd ../src

# please modify --model_path before running the following commands

# zhuang to chinese
python3 main.py \
--src_lang mn \
--tgt_lang zh \
--dict_path ../data/mn_zh_dict.jsonl \
--corpus_path ../data/zh_mn_train.json \
--test_data_path ../data/zh_mn_test.json \
--model_name Qwen3-32B \
--model_path ../models/Qwen/Qwen3-32B \
--prompt_type za2zh \
--num_parallel_sent 3 \
# --no_vllm \
--output_path ../output/Qwen3-32B_mn2zh_fuzzy2 \


chinese to zhuang
python3 main.py \
--src_lang zh \
--tgt_lang mn \
--dict_path ../data/zh_mn_dict.jsonl \
--corpus_path ../data/zh_mn_train.json \
--test_data_path ../data/zh_mn_test.json \
--model_name Qwen3-32B \
--model_path ../models/Qwen/Qwen3-32B \
--prompt_type zh2za \
--num_parallel_sent 3 \
# --no_vllm \
--output_path ../output/Qwen3-32B_zh2mn_fuzzy2\

