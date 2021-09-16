python -m torch.distributed.launch --nproc_per_node=4 --master_port=12895 train.py\
        --batch=32\
        --n_sample=25\
        --size=256\
        --use_CAN\
