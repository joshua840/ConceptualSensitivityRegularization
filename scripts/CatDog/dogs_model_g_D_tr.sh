export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="

WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="temp"
DATA_PATH='/data/'
YAMLS="--model configsNew/FeatureERM.yaml --trainer configsNew/trainer_logger_updated.yaml "
DEFAULTS="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 2 --model.dataset catdog --model.data_dir ${DATA_PATH}"

# Debug run
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main_temp.py $YAMLS $DEFAULTS \
 --seed_everything 1234 --model.minor_ratio 0.05 --model.batch_size_train 32 --model.learning_rate 1e-3 --model.set_last_layer three_layer 
