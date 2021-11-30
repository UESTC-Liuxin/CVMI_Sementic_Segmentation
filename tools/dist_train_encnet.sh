python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29501 \
    tools/train.py \
    -c /home/liuxin/Documents/CVMI_Sementic_Segmentation/configs/encnet/encnet_skmt_baseline.yml


# ps -ef | grep -v grep | grep tools/train.py  | awk '{print $2}' | xargs kill -9

