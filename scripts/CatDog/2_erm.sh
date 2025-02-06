WORKSPACE_NAME="REQUIRED"
PROJECT_NAME="REQUIRED"
DATA_PATH="PATH/TO/DATA"
YAMLS="--model configs/FeatureERM.yaml --trainer configs/trainer.yaml "
TRAINER="--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} --trainer.logger.name ${PROJECT_NAME} --trainer.max_epochs 600 "
MODEL="--model.dataset catdog --model.data_dir ${DATA_PATH} --model.input_type feature "

for LL in three_layer linear
do
for WD in 1e-2
do
for MR in 0.05 0
do
for SEED in 1234 1235 1236
do
for BS in 16 32
do

CUDA_VISIBLE_DEVICES=0 python -m csr.main $YAMLS $TRAINER $MODEL \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 1e-3 --model.set_last_layer $LL & 

CUDA_VISIBLE_DEVICES=1 python -m csr.main $YAMLS $TRAINER $MODEL \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 3e-3 --model.set_last_layer $LL & 
 
CUDA_VISIBLE_DEVICES=2 python -m csr.main $YAMLS $TRAINER $MODEL \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 1e-4 --model.set_last_layer $LL & 

CUDA_VISIBLE_DEVICES=3 python -m csr.main $YAMLS $TRAINER $MODEL \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate 3e-4 --model.set_last_layer $LL & 


done
done
wait
done
done
done

