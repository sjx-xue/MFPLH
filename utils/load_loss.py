from utils.loss import *


def getLoss(loss_func_name, class_freq, train_num):
    if loss_func_name == 'BCE':
        loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=False, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'FL':
        loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'CBloss':  # CB
        loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'R-BCE-Focal':  # R-FL
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'NTR-Focal':  # NTR-FL
        loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'DBloss-noFocal':  # DB-0FL
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                                 focal=dict(focal=False, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                                 class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'CBloss-ntr':  # CB-NTR
        loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=0.8),
                                 CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                 class_freq=class_freq, train_num=train_num)
        # loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
        #                          focal=dict(focal=True, alpha=0.5, gamma=2),
        #                          logit_reg=dict(init_bias=0.05, neg_scale=2.0),
        #                          CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
        #                          class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'DBloss':  # DB
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 map_param=dict(alpha=0.1, beta=25, gamma=0.9),
                                 class_freq=class_freq, train_num=train_num)
        # loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
        #                          focal=dict(focal=True, alpha=0.5, gamma=2),
        #                          logit_reg=dict(init_bias=0.05, neg_scale=2.0),
        #                          map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
        #                          class_freq=class_freq, train_num=train_num)

    if loss_func_name == 'FD':
        # print("使用的损失函数为： FD")
        loss_func = FocalDiceLoss(clip_pos=0.7, clip_neg=0.5, pos_weight=0.3)

    if loss_func_name == 'FD_smooth':
        loss_func = FocalDiceLoss_smooth(clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, smoothing=0.2)

    if loss_func_name == 'ASL':
        # loss_func = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        loss_func = AsymmetricLoss(gamma_neg=2.5, gamma_pos=1, clip=0, disable_torch_grad_focal_loss=True)

    if loss_func_name == 'RAL':
        print("使用的损失函数为： RAL")
        loss_func = Ralloss(gamma_neg=2, gamma_pos=1, clip=0, eps=1e-8, lamb=1.5, epsilon_neg=0.0, epsilon_pos=1.0, epsilon_pos_pow=-2.5, disable_torch_grad_focal_loss=False)

    if loss_func_name == 'Hill':
        print("使用的损失函数为： Hill")
        loss_func = Hill()

    if loss_func_name == 'FD_ours':
        print("使用的损失函数为： FD_ZLPR")
        # loss_func = ours_1(clip_pos=0.7, clip_neg=0.5, pos_weight=0.3)
        loss_func = FD_ZLPR(clip_pos=0.7, clip_neg=0.5, pos_weight=0.3)
        # loss_func = ours_1(clip_pos=0.7, clip_neg=0.5, pos_weight=0.3)
        # loss_func = FD_ours_2(clip_pos=0.7, clip_neg=0.5, pos_weight=0.3)

    if loss_func_name == 'ZLPR':
        print("使用的损失函数为： ZLPR")
        loss_func = multilabel_categorical_crossentropy()

    return loss_func