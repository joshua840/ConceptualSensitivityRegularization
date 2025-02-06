WORKSPACE_NAME="REQUIRED"
PROJECT_NAME="REQUIRED"
DATA_PATH="PATH/TO/DATA"

MAIN_ARGS="--seed 1234 --loggername neptune --dataset dogs --litmodule feature_jtt_meta "
TRAINER_ARGS="--default_root_dir ${ROOT_DIR} --max_epochs 60 --accelerator gpu --devices 1 --log_every_n_steps=30 --check_val_every_n_epoch 1 "
MODEL_ARGS="--model convnext_t --imagenet_pretrained True --freeze_model True --freezing_target_layer classifier.2 --h_activation_fn softplus --h_softplus_beta 10 --set_last_layer three_layer" #--model_path --set_last_layer
CLS_ARGS="--scheduler cosineannealing --milestones 9999 --optimizer adamw --criterion bce --learning_rate 3e-3 --weight_decay 1e-2 " #1e-3
F_ARGS="--del_backbone True "
DATA_ARGS="--batch_size_train 32 --batch_size_test 32 --data_seed 1234 --num_workers 2 --data_dir ${DATA_ROOT} "




#################################################################################################################################
# # Metadata generation for JTT
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPrior/jtt_weight_generator.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS $F_ARGS \
--minor_ratio 0 &
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPrior/jtt_weight_generator.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS $F_ARGS \
--minor_ratio 0.005 &
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPrior/jtt_weight_generator.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS $F_ARGS \
--minor_ratio 0.01 &
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPrior/jtt_weight_generator.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS $F_ARGS \
--minor_ratio 0.05 &


MAIN_ARGS=" --loggername neptune --dataset dogs --litmodule feature_jtt "
TRAINER_ARGS="--default_root_dir ${ROOT_DIR} --max_epochs 200 --accelerator gpu --devices 1 --log_every_n_steps=30 --check_val_every_n_epoch 1 "

# JTT
for SEED in 1234 1235 1236
do
for UC in 2 5 10 20
do
for EPOCH in 1 2 5 10
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPrior/spurious_feature_main.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS \
--minor_ratio 0 --upsample_count $UC --upsample_indices_path "./output/${PROJECT_NAME}/DOG-318/jtt_meta_${EPOCH}.pt" --seed $SEED --data_seed $SEED &
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPrior/spurious_feature_main.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS \
--minor_ratio 0.005 --upsample_count $UC --upsample_indices_path "./output/${PROJECT_NAME}/DOG-316/jtt_meta_${EPOCH}.pt"  --seed $SEED --data_seed $SEED &
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPrior/spurious_feature_main.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS \
--minor_ratio 0.01 --upsample_count $UC --upsample_indices_path "./output/${PROJECT_NAME}/DOG-317/jtt_meta_${EPOCH}.pt"  --seed $SEED --data_seed $SEED &
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPrior/spurious_feature_main.py $MAIN_ARGS $TRAINER_ARGS $MODEL_ARGS $CLS_ARGS $DATA_ARGS \
--minor_ratio 0.05 --upsample_count $UC --upsample_indices_path "./output/${PROJECT_NAME}/DOG-315/jtt_meta_${EPOCH}.pt"  --seed $SEED --data_seed $SEED &
done
wait
done
done
##################################################################################################################################
