export CUDA_VISIBLE_DEVICES=4,5,6,7


# model name: s1-32B-0.8, s1.1-32B-0.8, limo-32B-0.8
python main.py \
    --model limo-32B-0.8 \
    --dataset limo_split \
    --batch_size 16 \
    --half \
    --clip_inf
    