class Config:
    # 基本参数
    score_scale = 1e-3
    exemplar_size = 127
    instance_size = 255
    score_size = 17
    # 数据集处理相关参数
    frame_range = 100
    context_amount = 0.5
    # 训练相关参数
    epoch_num = 100
    batch_size = 8
    num_workers = 8
    beg_lr = 1e-2
    end_lr = 1e-5
    weight_decay = 5e-4
    momentum = 0.9
    stride = 8
    pos_radius = 16
    neg_weight = 1.0
    model_save_dir = '/home/snorlax/Projects/SiamFC-pytorch/models/'
    log_dir = '/home/snorlax/Projects/SiamFC-pytorch/runs/'
    # 跟踪相关参数
    upscale_size = 272
    scale_num = 3
    scale_step = 1.0375
    scale_lr = 0.59
    scale_penalty = 0.9745
    window_influence = 0.176

cfg = Config()