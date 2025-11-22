# # export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=ALL
# # export NCCL_ASYNC_ERROR_HANDLING=1
# # export NCCL_P2P_LEVEL=NVL
# # export NCCL_IB_DISABLE=1
# # export NCCL_SOCKET_IFNAME=^lo,docker0
echo $CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 val.py --base_dir <base_dir> --batch_size 14 --log_dir ./training_logs/validation --load_from <model_checkpoint_path>