for CAV_TYPE in svm signal
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main_cav \
--dataset catdog_concepts \
--model_name convnext_t \
--cav_type $CAV_TYPE \
--root /media/disk2/Data \
--save_path ./data/cavs

done





