export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="

WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="temp"
DATA_PATH='/media/disk2/Data'
DEFAULTS="\
--model configs/FeatureCGR_stage1.yaml \
--model.dataset waterbirds_concepts_v2 \
--model.cgr_stage stage1 \
--model.data_dir ${DATA_PATH} \
--model.g_model three_layer \
--model.set_last_layer three_layer \
--trainer configs/trainer.yaml \
--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} \
--trainer.logger.name ${PROJECT_NAME} \
--trainer.max_epochs 200 "

for LL in three_layer
do
for BS in 4
do
for SEED in 1234
do
for LR in 1e-3
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1 --model.lamb_cav 1
done
done
done
done
