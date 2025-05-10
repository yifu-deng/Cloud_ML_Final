# Benchmarking Deep Vision Models on A100: ResNet50, ConvNeXt, TinyViT

## Team Members

- **Yifu Deng (yd2686)** – ResNet50-v2 BiT, Performance Analysis (Time & Throughput)
- **Zhehan Qi (zq2201)** – ConvNeXt, Performance Analysis (NCU Roofline Analysis)  
- **Boshan Chen (bc3603)** – TinyViT, Accuracy Comparison & Inference Demo

## Performance Analysis Instructions

### 1. ResNet50-v2 BiT

```bash
python main.py /path/to/imagenet-100/ \
  -a resnet50 \
  --batch-size 128 \
  --epochs 1 \
  --workers 8 \
  --gpu 0
```

### 2. ConvNeXt-Tiny

```bash
python main.py \
  --model convnext_tiny \
  --drop_path 0.1 \
  --batch_size 256 \
  --lr 2e-4 \
  --epochs 1 \
  --use_amp true \
  --data_set image_folder \
  --disable_eval true \
  --data_path /path/to/imagenet-100/train/ \
  --output_dir ./save_results \
  --nb_classes 25 \
  --num_workers 8 \
  --dist_url 'tcp://localhost:10001' \
  --warmup_steps 0 \
  --warmup_epochs 0 \
  --auto_resume false
```

### 3. TinyViT-21M

```bash
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345

python main.py \
  --cfg configs/1k/tiny_vit_21m.yaml \
  --data-path /path/to/imagenet-100/ \
  --batch-size 256 \
  --output ./output \
  --local_rank 0 \
  --opts TRAIN.BASE_LR 2.5e-4 TRAIN.EPOCHS 1 TRAIN.WARMUP_EPOCHS 0 MODEL.NUM_CLASSES 25 TRAIN.AUTO_RESUME False
```

---

## Nsight Compute Profiling Instructions

### 1. ResNet50-v2 BiT

```bash
ncu --profile-from-start off \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum \
  --csv \
  --page raw \
  --log-file resnet50-a100.csv \
  --target-processes all \
  python main2.py /path/to/imagenet-100/ -a resnet50 --epochs 1 -b 64 --gpu 0
```

### 2. ConvNeXt-Tiny

```bash
ncu --profile-from-start on \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum \
  --csv \
  --page raw \
  --log-file convnext_tiny-a100.csv \
  --target-processes all \
  python forward_train.py
```

### 3. TinyViT-21M

```bash
ncu --profile-from-start on \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_fp16_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__inst_executed_pipe_tensor.sum \
  --csv \
  --page raw \
  --log-file tinyvit-a100.csv \
  --target-processes all \
  python forward_train.py \
  --cfg configs/1k/tiny_vit_21m.yaml \
  --batch-size 64 \
  --opts MODEL.NUM_CLASSES 25
```

## Roofline Model and Accuracy Analysis Instructions

Run all .ipynb files in Jupyter Notebook.

## Reference Repos:
ResNet-V2-BiT: https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k_ft_in1k

ConvNeXt: https://github.com/facebookresearch/ConvNeXt

TinyViT: https://github.com/wkcn/TinyViT
