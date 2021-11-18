python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    tools/train.py \
    -c /home/liuxin/Documents/CVMI_Sementic_Segmentation/configs/base_demo.yml
    