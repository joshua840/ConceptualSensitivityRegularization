WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="temp"
DATA_PATH='/media/disk2/Data'
G_PATH="/home/jj/Research/ConceptualSensitivityRegularization/data/cavs/catdog_concepts_convnext_t_signal.pt"
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/temp/TEM-578/checkpoints/last.ckpt'
DEFAULTS="\
--model configs/FeatureRRC_stage2.yaml \
--model.dataset waterbirds \
--model.data_dir ${DATA_PATH} \
--model.g_ckpt_path ${G_PATH} \
--model.cs_method dot_sq \
--model.model_path ${MODEL_PATH} \
--model.set_last_layer three_layer \
--trainer configs/trainer.yaml \
--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} \
--trainer.logger.name ${PROJECT_NAME} \
--trainer.max_epochs 600 "



G_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/data/cavs/catdog_concepts_convnext_t_svm.pt'
LL=three_layer
MR=0.0
BS=16
LR=0.003

SEED=1234
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/.neptune/240401CatDog/CAT3-37/checkpoints/last.ckpt'
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1e-06 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.0001 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.01 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
wait
SEED=1235
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/.neptune/240401CatDog/CAT3-27/checkpoints/last.ckpt'
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1e-06 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.0001 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.01 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
wait
SEED=1236
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/.neptune/240401CatDog/CAT3-40/checkpoints/last.ckpt'
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1e-06 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.0001 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.01 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
wait
LL=three_layer
MR=0.05
BS=16
LR=0.003

SEED=1234
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/.neptune/240401CatDog/CAT3-10/checkpoints/last.ckpt'
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1e-06 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.0001 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.01 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
wait
SEED=1235
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/.neptune/240401CatDog/CAT3-11/checkpoints/last.ckpt'
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1e-06 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.0001 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.01 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
wait
SEED=1236
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/ConceptualSensitivityRegularization/.neptune/240401CatDog/CAT3-5/checkpoints/checkpt-epoch=206-valid_valid_worst_acc=0.969.ckpt'
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1e-06 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.0001 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 0.01 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
 CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS     --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL     --model.lamb_cs 100000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH &
wait