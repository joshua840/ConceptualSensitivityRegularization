############################################################### Dogs feature extractor #################################################################
YAMLS="--model configs/FeatureExtractor/LitFeatureExtraction.yaml --trainer configs/FeatureExtractor/trainer.yaml "
DATA_ARGS="--model.dataset catdog --model.data_dir /media/disk1/Data --model.save_root /media/disk2/Data/Features "
CUDA_VISIBLE_DEVICES=0 python3 -m csr.jtt_weight_generator $YAMLS $DATA_ARGS \
 --seed_everything 1234 --trainer.max_epochs 1000 --trainer.check_val_every_n_epoch 1000 --trainer.num_sanity_val_steps -1 \


# ############################################################### Dogs_concepts feature extractor #################################################################
# OUTPUT_DIR='./output'
# SPURIOUS_ROOT='/home/data/'
# CONCEPT_ROOT='/home/data/'

# MAIN_ARGS="--seed 1234 --dataset waterbirds_concepts --tr_epochs 1000 --tr_shuffle False --data_module default" #--ckpt_path PATH --save_file_name FN
# MODEL_ARGS="--model convnext_t --imagenet_pretrained True --freeze_model True --freezing_target_layer classifier.2 " #--model_path --set_last_layer
# DATA_ARGS="--data_dir /home/data/Places205/data/vision/torralba/deeplearning/images256 --data_seed 1234 --num_workers 1 --batch_size_train 128 --batch_size_test 100 --minor_ratio 1"

# CUDA_VISIBLE_DEVICES=3 python3 smoothAttributionPriorNew/feature_extractor.py $MAIN_ARGS $MODEL_ARGS $DATA_ARGS \
# --target_layer 'classifier.1' \
# --save_root /data/Features

