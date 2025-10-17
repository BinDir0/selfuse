python train.py --multirun \
    experiment=train_vq_tokenizer_multirun \
    tokenizer.states.wrist.config.quantizer_config.nb_code=1024,2048,4096,1024,2048,4096,2048,4096 \
    tokenizer.states.wrist.config.quantizer_config.codebook_dim=256,256,256,512,512,512,512,512 \
    tokenizer.states.wrist.config.quantizer_config.num_groups=1,1,1,1,1,1,2,2 \
    tokenizer.actions.wrist.config.quantizer_config.nb_code=1024,2048,4096,1024,2048,4096,2048,4096 \
    tokenizer.actions.wrist.config.quantizer_config.codebook_dim=256,256,256,512,512,512,512,512 \
    tokenizer.actions.wrist.config.quantizer_config.num_groups=1,1,1,1,1,1,2,2 \
