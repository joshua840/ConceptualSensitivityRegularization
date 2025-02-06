# ############################################################### catdog_concepts feature extractor #################################################################
CUDA_VISIBLE_DEVICES=0 python3 -m csr.main_run fit \
 --trainer configs/FeatureExtractor/trainer.yaml \
 --model configs/FeatureExtractor/LitFeatureExtraction.yaml \
 --model.dataset catdog \
 --model.data_dir /PATH/TO/DATA \
 --model.save_root /SAVE/PATH \
 --model.nimg_per_concept 128 \
 --trainer.max_epochs 1000 \
 --trainer.check_val_every_n_epoch 1000 \
 --trainer.num_sanity_val_steps -1 