_base_ = './mask_rcnn_prime_while_finetune_swin-t-p4-w7_fpn_ms-crop-2x_coco.py'
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
