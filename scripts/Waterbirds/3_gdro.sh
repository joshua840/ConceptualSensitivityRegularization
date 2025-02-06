WORKSPACE_NAME="REQUIRED"
PROJECT_NAME="REQUIRED"
DATA_PATH="PATH/TO/DATA"
DEFAULTS="\
--model configs/FeatureGDRO.yaml \
--model.dataset waterbirds \
--model.data_dir ${DATA_PATH} \
--model.set_last_layer three_layer \
--trainer configs/trainer.yaml \
--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} \
--trainer.logger.name ${PROJECT_NAME} \
--trainer.max_epochs 600 "


for MR in 0.05 0
do
for BS in 16
do
for SEED in 1234 1235 1236
do
for LR in 1e-3 3e-3 3e-4 1e-4
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 0.01 &
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 0.03 &
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 0.1 &
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 0.3 &
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 1 &
CUDA_VISIBLE_DEVICES=1 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 3 &
CUDA_VISIBLE_DEVICES=2 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 10 &
CUDA_VISIBLE_DEVICES=3 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.minor_ratio $MR --model.batch_size_train $BS --model.learning_rate $LR --model.eta 30 &
done
wait
done
done
done
