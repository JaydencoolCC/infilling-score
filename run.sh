export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5

# model name: s1-32B-0.8, s1.1-32B-0.8, limo-32B-0.8
python main.py \
    --model s1-32B-0.8 \
    --dataset s1K_split  \
    --batch_size 8 \
    --half \
    --clip_inf