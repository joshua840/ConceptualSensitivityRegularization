WORKSPACE_NAME="REQUIRED"
PROJECT_NAME="REQUIRED"
DATA_PATH="PATH/TO/DATA"
DEFAULTS="\
--model configs/FeatureRRC_stage2.yaml \
--model.dataset waterbirds \
--model.data_dir ${DATA_PATH} \
--model.cs_method dot_sq \
--model.set_last_layer three_layer \
--trainer configs/trainer.yaml \
--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} \
--trainer.logger.name ${PROJECT_NAME} \
--trainer.max_epochs 600 "



G_PATH='/PATH/TO/CKPT.pt'
MODEL_PATH='/PATH/TO/CKPT.ckpt'
LL=three_layer
MR=0.0
BS=16
LR=0.003
LCS=1

python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs $LCS --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH
