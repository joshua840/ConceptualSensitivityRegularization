####################################### Debug run - ERM raw #######################################
# WORKSPACE_NAME="SmoothAttributionPrior"
# PROJECT_NAME="temp"
# DATA_PATH='/media/disk1/Data'
# YAMLS="--model configs/FeatureERM.yaml --trainer configs/trainer.yaml "
# TRAINER="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 "
# MODEL="--model.dataset catdog --model.data_dir ${DATA_PATH} --model.input_type raw "

# CUDA_VISIBLE_DEVICES=1 python -m csr.main $YAMLS $TRAINER $MODEL \
#  --seed_everything 1234 --model.minor_ratio 0.0 --model.batch_size_train 32 --model.learning_rate 1e-3 --model.set_last_layer three_layer


####################################### Debug run - ERM feature ####################################
WORKSPACE_NAME="SmoothAttributionPrior"
PROJECT_NAME="temp"
DATA_PATH='/media/disk2/Data'
YAMLS="--model configs/FeatureERM.yaml --trainer configs/trainer.yaml "
TRAINER="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 "
MODEL="--model.dataset catdog --model.data_dir ${DATA_PATH} --model.input_type feature "

# Debug run
CUDA_VISIBLE_DEVICES=0 python -m csr.main $YAMLS $TRAINER $MODEL \
 --seed_everything 1234 --model.minor_ratio 0.0 --model.batch_size_train 32 --model.learning_rate 1e-3 --model.set_last_layer three_layer \


# ####################################### Debug run - ERM feature ####################################
# WORKSPACE_NAME="SmoothAttributionPrior"
# PROJECT_NAME="240401CatDog"
# DATA_PATH='/media/disk2/Data'
# YAMLS="--model configs/FeatureERM.yaml --trainer configs/trainer.yaml "
# TRAINER="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 "
# MODEL="--model.dataset catdog --model.data_dir ${DATA_PATH} --model.num_workers 8 --model.input_type feature "

# for LL in three_layer
# do
# for WD in 1e-2
# do
# for MR in 0.05 0.01 0.005 0
# do
# for SEED in 1234 1235 1236
# do
# for BS in 16 32
# do

# CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 1e-3 --model.set_last_layer $LL & 

# CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 3e-3 --model.set_last_layer $LL & 
 
# CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 1e-4 --model.set_last_layer $LL & 

# CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 3e-4 --model.set_last_layer $LL & 


# done
# done
# wait
# done
# done
# done




# # raw run
# WORKSPACE_NAME="SmoothAttributionPrior"
# PROJECT_NAME="230912CatDog"
# PROJECT_NAME="temp"
# DATA_PATH='/home/data/'
# YAMLS="--model configsNew/FeatureERM.yaml --trainer configsNew/trainer.yaml "
# DEFAULTS="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 --model.dataset catdog2 --model.data_dir ${DATA_PATH} --model.input_type raw "


# CUDA_VISIBLE_DEVICES=0 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything 1234 --model.minor_ratio 0.05 --model.batch_size_train 32 --model.learning_rate 3e-3 --model.set_last_layer three_layer &

# CUDA_VISIBLE_DEVICES=1 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything 1234 --model.minor_ratio 0.05 --model.batch_size_train 32 --model.learning_rate 1e-3 --model.set_last_layer three_layer &

# CUDA_VISIBLE_DEVICES=2 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything 1234 --model.minor_ratio 0.05 --model.batch_size_train 32 --model.learning_rate 3e-4 --model.set_last_layer three_layer &

# CUDA_VISIBLE_DEVICES=3 python smoothAttributionPriorNew/main.py $YAMLS $DEFAULTS \
#  --seed_everything 1234 --model.minor_ratio 0.05 --model.batch_size_train 32 --model.learning_rate 1e-4 --model.set_last_layer three_layer &
