WORKSPACE_NAME="REQUIRED"
PROJECT_NAME="REQUIRED"
DATA_PATH="PATH/TO/DATA"
DEFAULTS="\
--model configs/FeatureCGR_stage1.yaml \
--model.dataset celeba_collar_concepts_v2 \
--model.cgr_stage stage1 \
--model.data_dir ${DATA_PATH} \
--model.g_model three_layer \
--model.set_last_layer three_layer \
--trainer configs/trainer.yaml \
--trainer.logger.project ${WORKSPACE_NAME}/${PROJECT_NAME} \
--trainer.logger.name ${PROJECT_NAME} \
--trainer.max_epochs 100 "

for LL in three_layer
do
for BS in 4 16
do
for SEED in 1234
do
for LR in 1e-3 3e-3
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main $DEFAULTS \
 --seed_everything $SEED --model.batch_size_train $BS --model.learning_rate $LR --model.set_last_layer $LL --model.lamb_cs 0 --model.lamb_cav 1 &
done
done
done
done

