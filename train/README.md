# Training instruction for HQ-SAM

> [**Segment Anything in High Quality**](https://arxiv.org/abs/2306.01567)           
> Lei Ke, Mingqiao Ye, Martin Danelljan, Yifan Liu, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu \
> ETH Zurich & HKUST

> Added **EvenSampler** to draw samples from all datasets \
> by Hongyuan Zhang, Mingqiao Ye and Lei Ke \
> ETH Zuirch & CMU


We organize the training folder as follows.
```
train
|____data
|____pretrained_checkpoint
|____train.py
|____utils
| |____dataloader.py
| |____misc.py
| |____loss_mask.py
|____segment_anything_training
|____work_dirs
```

## 1. Data Preparation

Datasets following standard MSCOCO formts. <br>
In this example, we used 6 datasets [ClearStain_Brightfield, Imprints_Brightfield, Imprints_DIC, Leaf_Brightfield, Peels_Brightfield, Peels_SEM] <br>
**You have to modify the `train.py` file accordingly to define the path of datastes**

### Expected datasets structure

```
data
|____ClearStain_Brightfield
    |___train_sahi
    |___val_sahi
|____Imprints_Brightfield
    |___train_sahi
    |___val_sahi
|____Imprints_DIC
    |___train_sahi
    |___val_sahi
|____Leaf_Brightfield
    |___train_sahi
    |___val_sahi
|____Peels_Brightfield
    |___train_sahi
    |___val_sahi
|____Peels_SEM
    |___train_sahi
    |___val_sahi
```

## 2. Init Checkpoint
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b_01ec64.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l_0b3195.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h_4b8939.pth

```

## 3. Training
Example training code <br>

Note that `total_samples` defines the total number of samples to be evenly drawn from all dataset in the mode of `even_sampling`

```
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12346 \
    train.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --output work_dirs/hq_sam_h \
    --max_epoch_num 24 \
    --lr_drop_epoch 18 \
    --batch_size_train 4 \
    --learning_rate 1e-4 \
    --total_samples 200 \
    --even_sampling \
    --find_unused_params
```

## 4. Evaluation
Example validation code

```
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12345 \
    train.py \
    --checkpoint ./pretrained_checkpoint/sam_vit_h_4b8939.pth \
    --model-type vit_h \
    --output work_dirs/hq_sam_h \
    --eval \
    --restore-model work_dirs/hq_sam_h/epoch_38.pth \
    --visualize
```
