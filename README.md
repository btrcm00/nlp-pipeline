# NLP training pipeline
- Base pipeline for training NLP task
- Help to create a training pipeline quickly

# How to use
1. Clone this repo: `git clone https://github.com/btrcm00/nlp-pipeline.git`.
2. Fill lines `raise "This func need implementation"` by your implementation that suitable with your model.
3. Training your model
    1. Note!!!
       - Pass `model_class` to init class Processor.
       - Pass `n_folds` != None if you want to training with K-fold validation
       - If you use Bloom, you should pass `use_lora=True`
    2. Export environment variables: `while read LINE; do export "$LINE"; done < .env`
    3. Run training: `PRETRAINED_PATH=XXXX CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 --master-port=30000 pipeline/processor.py`