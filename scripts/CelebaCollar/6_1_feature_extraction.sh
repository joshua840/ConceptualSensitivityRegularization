# ############################################################### catdog_concepts feature extractor #################################################################
CUDA_VISIBLE_DEVICES=0 python3 -m csr.main_run fit \
 --seed_everything 1234 \
 --trainer configs/FeatureExtractor/trainer.yaml \
 --model configs/FeatureExtractor/LitFeatureExtraction.yaml \
 --model.dataset celeba_collar_concepts_v2 \
 --model.data_dir /PATH/TO/DATA \
 --model.save_root /SAVE/PATH \
 --model.nimg_per_concept 128 \
 --trainer.max_epochs 100 \
 --trainer.check_val_every_n_epoch 100 \
 --trainer.num_sanity_val_steps -1 \
 --model.num_workers 8 \


