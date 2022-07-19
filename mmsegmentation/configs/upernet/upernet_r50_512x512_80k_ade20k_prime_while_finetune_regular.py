_base_ = [
    '../_base_/models/upernet_r50_regular.py', '../_base_/datasets/ade20k_prime.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k_with_prime.py'
]
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))
