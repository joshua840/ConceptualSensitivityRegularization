export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="

WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="230920CatDog"
# PROJECT_NAME="temp"
DATA_PATH='/data/'
# DATA_PATH='/media/disk2/Data'
YAMLS="--model configsNew/FeatureCGR.yaml --trainer configsNew/trainer.yaml "
G_DEFAULTS="--model.target_layer classifier.1 --model.g_ckpt_path null --model.g_freeze False --model.g_model three_layer --model.g_num_classes 1 --model.g_activation softplus --model.g_softplus_beta 10 --model.g_criterion bce  --model.cgr_stage stage1 --model.grad_from logit "
DEFAULTS="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 20 --trainer.inference_mode False --model.dataset catdog_concepts --model.data_dir ${DATA_PATH} --model.set_last_layer three_layer "


########################################### Stage 1 ###########################################
# # Debug run
for LL in three_layer
do
for WD in 1e-2
do
for BS in 16 
do
for SEED in 1234 1235 1236
do
for LR in 1e-3
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS \
 --seed_everything $SEED --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1 --model.lamb_cav 1 &
done
done
done
done
done

