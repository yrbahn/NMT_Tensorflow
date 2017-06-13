python train.py --source_vocab_path=data/vocab_en.txt  \
 --target_vocab_path=data/vocab_kr.txt \
 --source_files=data/kaist_corpus_bpe.en \
 --target_files=data/kaist_corpus_bpe.kr \
 --dev_source_files=data/kaist_dev_bpe.en \
 --dev_target_files=data/kaist_dev_bpe.kr \
 --schedule=evaluate \
 --output_dir=./model_1
