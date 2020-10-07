python train.py \
--train_path "/content/drive/My Drive/mtqa4kg/data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_300_threshold_4_max_distance_-1/train.json" \
--dev_path "/content/drive/My Drive/mtqa4kg/data/cleaned_data/ACE2005/bert-base-uncased_overlap_15_window_300_threshold_4_max_distance_-1/test.json" \
--train_batch 20 
--amp  
--eval 
--turn2_down_sample_ratio -1
--max_len 350  
--max_distance -1
