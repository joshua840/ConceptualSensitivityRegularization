export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhYTBjMGNjYS01MjI1LTQxZjgtYmRlZS1jMmYwYzgxNDE5ODEifQ=="

WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="230920CatDog"
DATA_PATH='/data/'
# PROJECT_NAME="temp"
# DATA_PATH='/media/disk2/Data'
G_PATH="/home/jj/Research/SmoothAttributionPrior/.neptune/temp/TEM-457/checkpoints/last.ckpt"
MODEL_PATH='/home/jj/Research/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-214/checkpoints/checkpt-epoch=25-valid_valid_worst_acc=0.775.ckpt'
YAMLS="--model configsNew/FeatureCGR.yaml --trainer configsNew/trainer.yaml "
G_DEFAULTS="--model.target_layer classifier.1 --model.g_ckpt_path ${G_PATH} --model.g_freeze True --model.g_model three_layer --model.g_num_classes 1 --model.g_activation softplus --model.g_softplus_beta 10 --model.g_criterion bce --model.cgr_stage stage2 --model.cs_method cross_dot_sq --model.grad_from logit --model.model_path ${MODEL_PATH}"
DEFAULTS="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 --model.dataset catdog --model.data_dir ${DATA_PATH} --model.set_last_layer three_layer "


# # # Debug run
# for LL in three_layer
# do
# for MR in 0.05
# do
# for BS in 16 
# do
# for SEED in 1234 
# do
# for LR in 1e-3
# do
# CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS \
#  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1 --model.lamb_cav 1
# done
# done
# done
# done
# done


for LL in three_layer
do
for BS in 16
do
for LR in 1e-3
do
      
      
G_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-1441/checkpoints/last.ckpt'
SEED=1234
MR=0
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-291/checkpoints/checkpt-epoch=222-valid_valid_worst_acc=0.550.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1234
MR=0.005
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-281/checkpoints/checkpt-epoch=37-valid_valid_worst_acc=0.650.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1234
MR=0.01
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-236/checkpoints/checkpt-epoch=25-valid_valid_worst_acc=0.667.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1234
MR=0.05
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-214/checkpoints/checkpt-epoch=25-valid_valid_worst_acc=0.775.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

G_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-1442/checkpoints/last.ckpt'
SEED=1235
MR=0
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-305/checkpoints/checkpt-epoch=07-valid_valid_worst_acc=0.492.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1235
MR=0.005
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-267/checkpoints/checkpt-epoch=211-valid_valid_worst_acc=0.758.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1235
MR=0.01
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-255/checkpoints/checkpt-epoch=57-valid_valid_worst_acc=0.683.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1235
MR=0.05
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-218/checkpoints/checkpt-epoch=155-valid_valid_worst_acc=0.825.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

G_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-1443/checkpoints/last.ckpt'
SEED=1236
MR=0
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-301/checkpoints/checkpt-epoch=06-valid_valid_worst_acc=0.525.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1236
MR=0.005
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-269/checkpoints/checkpt-epoch=60-valid_valid_worst_acc=0.600.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1236
MR=0.01
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-253/checkpoints/checkpt-epoch=14-valid_valid_worst_acc=0.750.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

SEED=1236
MR=0.05
MODEL_PATH='/mnt/ssd/jj/Research/Maxent/SmoothAttributionPrior/.neptune/230920CatDog/CAT1-228/checkpoints/checkpt-epoch=21-valid_valid_worst_acc=0.792.ckpt'
for CS_METHOD in dot_abs dot_sq cosine_sq cosine_abs
do
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0.1 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 10000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 100000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS $G_DEFAULTS  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 1000000.0 --model.lamb_cav 0 --model.g_ckpt_path $G_PATH --model.model_path $MODEL_PATH --model.cs_method $CS_METHOD & 
wait
done

done
done
done
