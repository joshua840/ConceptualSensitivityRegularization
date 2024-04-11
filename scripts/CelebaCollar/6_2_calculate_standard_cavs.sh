# argument lists of main_cav is as follow.
# parser.add_argument("--dataset", type=str, default="catdog")
# parser.add_argument("--model_name", type=str, default="convnext_t")
# parser.add_argument("--cav_type", type=str, default="svm")
# parser.add_argument("--root", type=str, default="/home/data/Features")
# parser.add_argument("--save_path", type=str, default="/home/data/Features")

for CAV_TYPE in svm signal
do
CUDA_VISIBLE_DEVICES=0 python -m csr.main_cav \
--dataset celeba_collar_concepts_v2 \
--model_name convnext_t \
--cav_type $CAV_TYPE \
--root /media/disk2/Data \
--save_path ./data/cavs

done





