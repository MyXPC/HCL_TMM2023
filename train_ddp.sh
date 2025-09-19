# 使用torch.distributed.launch启动DDP训练（推荐方式）
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12355 main_ddp.py \
    --bs 15 \
    --net 'resnet50' \
    --data "../dataset/train/400" \
    --val "../dataset/val/val400" \
    --save_dir "./runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3

# 或者使用torchrun（PyTorch 1.9+推荐）
torchrun --nproc_per_node=4 --master_port=12355 main_ddp.py \
    --bs 8 \
    --net 'resnet50' \
    --data "../../dataset/train400" \
    --val "../../dataset/val400" \
    --save_dir "./runs" \
    --epochs 100 \
    --drop_rate 0.25 \
    --gpu 0,1,2,3
