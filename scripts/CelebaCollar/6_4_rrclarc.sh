WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="240406CelebaCollar"
# PROJECT_NAME="temp"
DATA_PATH='/media/disk2/Data'
DEFAULTS="\
--model configs/FeatureRRC_stage2.yaml \
--model.dataset celeba_collar \
--model.data_dir ${DATA_PATH} \
--model.cs_method dot_sq \
--model.set_last_layer three_layer \
--trainer configs/trainer.yaml \
--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} \
--trainer.logger.name ${PROJECT_NAME} \
--trainer.max_epochs 100 "


# # Debug run
# G_PATH='/home/jj/Research/ConceptualSensitivityRegularization/data/cavs/celeba_collar_concepts_convnext_t_signal.pt'
# # G_PATH='/home/jj/Research/ConceptualSensitivityRegularization/data/cavs/celeba_collar_concepts_convnext_t_svm.pt'

# LL=three_layer
# MR=0.0
# BS=16
# LR=0.003
# SEED=1234
# MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-134/checkpoints/last.ckpt'
# for CS_METHOD in dot_sq
# do
# CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD 
# done

# G_PATH='/home/jj/Research/ConceptualSensitivityRegularization/data/cavs/celeba_collar_concepts_convnext_t_signal.pt'
G_PATH='/home/jj/Research/ConceptualSensitivityRegularization/data/cavs/celeba_collar_concepts_convnext_t_svm.pt'

LL=three_layer
MR=0.0
BS=16
LR=0.003
SEED=1234
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-134/checkpoints/last.ckpt'
for CS_METHOD in dot_sq
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

LL=three_layer
MR=0.0
BS=16
LR=0.003
SEED=1235
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-133/checkpoints/last.ckpt'
for CS_METHOD in dot_sq
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

LL=three_layer
MR=0.0
BS=16
LR=0.003
SEED=1236
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-124/checkpoints/last.ckpt'
for CS_METHOD in dot_sq
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

LL=three_layer
MR=0.05
BS=16
LR=0.003
SEED=1234
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-98/checkpoints/last.ckpt'
for CS_METHOD in dot_sq
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

LL=three_layer
MR=0.05
BS=16
LR=0.003
SEED=1235
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-108/checkpoints/last.ckpt'
for CS_METHOD in dot_sq
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

LL=three_layer
MR=0.05
BS=16
LR=0.003
SEED=1236
MODEL_PATH='/home/jj/Research/ConceptualSensitivityRegularization/.neptune/240406CelebaCollar/CEL2-105/checkpoints/last.ckpt'
for CS_METHOD in dot_sq
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done
