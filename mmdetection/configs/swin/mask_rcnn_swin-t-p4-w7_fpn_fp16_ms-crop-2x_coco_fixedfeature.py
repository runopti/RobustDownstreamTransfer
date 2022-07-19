_base_ = './mask_rcnn_swin-t-p4-w7_fpn_ms-crop-2x_coco_fixedfeature.py'
# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
)
