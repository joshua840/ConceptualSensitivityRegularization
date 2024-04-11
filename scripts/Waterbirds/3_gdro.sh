export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="

WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="230920CatDog"
# PROJECT_NAME="temp"
DATA_PATH='/data/'
YAMLS="--model configsNew/FeatureGDRO.yaml --trainer configsNew/trainer.yaml "
DEFAULTS="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 --model.dataset catdog --model.data_dir ${DATA_PATH}"


# # Debug run
for LL in three_layer
do
for WD in 1e-2
do
for MR in 0.05 0.01 0.005 0
do
for BS in 16 32 # 16 32
do
for SEED in 1234 1235 1236
do
for LR in 1e-3 3e-3 3e-4 1e-4
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.eta 0.01 &
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.eta 0.03 &
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.eta 0.1 &
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.eta 0.3 &
done
wait
done
done
done
done
done

