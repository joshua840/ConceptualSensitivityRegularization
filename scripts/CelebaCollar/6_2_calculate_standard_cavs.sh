for CAV_TYPE in svm signal
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main_cav \
--dataset celeba_collar_concepts_v2 \
--model_name convnext_t \
--cav_type $CAV_TYPE \
--root /PATH/TO/DATA \
--save_path ./data/cavs

done





