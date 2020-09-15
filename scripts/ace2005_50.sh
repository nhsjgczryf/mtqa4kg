python train.py \
--eval \
--train_path ./data/cleaned_data/ACE2005/50_mini_train.json \
--dev_path ./data/cleaned_data/ACE2005/50_mini_train.json  \
--pretrained_model_path bert-base-cased \
--warmup_ratio -1  \
--lr 8e-6