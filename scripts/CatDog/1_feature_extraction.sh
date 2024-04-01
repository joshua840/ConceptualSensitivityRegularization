# ############################################################### CatDog feature extractor #################################################################
# YAMLS="--model configs/FeatureExtractor/LitFeatureExtraction.yaml --trainer configs/FeatureExtractor/trainer.yaml "
# DATA_ARGS="--model.dataset catdog --model.data_dir /media/disk1/Data --model.save_root /media/disk2/Data/Features "

# CUDA_VISIBLE_DEVICES=0 python3 -m csr.jtt_weight_generator $YAMLS $DATA_ARGS \
#  --seed_everything 1234 --trainer.max_epochs 1000 --trainer.check_val_every_n_epoch 1000 --trainer.num_sanity_val_steps -1 \


# ############################################################### catdog_concepts feature extractor #################################################################
CUDA_VISIBLE_DEVICES=0 python3 -m csr.main_run fit \
 --trainer configs/FeatureExtractor/trainer.yaml \
 --model configs/FeatureExtractor/LitFeatureExtraction.yaml \
 --model.dataset catdog_concepts \
 --model.data_dir /media/disk2/Data \
 --model.save_root /media/disk2/Data/Features \
 --model.nimg_per_concept 128 \
 --seed_everything 1234 \
 --trainer.max_epochs 1000 \
 --trainer.check_val_every_n_epoch 1000 \
 --trainer.num_sanity_val_steps -1 

