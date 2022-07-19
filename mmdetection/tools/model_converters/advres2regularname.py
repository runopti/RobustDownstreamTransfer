import torch
import torch.nn as nn
import argparse
from collections import OrderedDict

def convert_advres(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('module.attacker') or k.startswith('module.normalizer'):
            continue
        elif k.startswith('module.model'):
            new_v = v
            new_k = k.replace('module.model.', '')
        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
   parser = argparse.ArgumentParser(
       description='Convert keys in official pretrained swin models to'
       'MMDetection style.')
   parser.add_argument('src', help='src detection model path')
   # The dst path must be a full path of the new checkpoint.
   parser.add_argument('dst', help='save path')
   args = parser.parse_args()

   checkpoint = torch.load(args.src, map_location='cpu')
   # module.model -> backbone
   
   if 'state_dict' in checkpoint:
       state_dict = checkpoint['state_dict']
   elif 'model' in checkpoint:
       state_dict = checkpoint['model']
   else:
       state_dict = checkpoint

   weight = convert_advres(state_dict)
   del checkpoint['optimizer']
   del checkpoint['epoch']
   del checkpoint['amp']
   del checkpoint['schedule']

    
   with open(args.dst, 'wb') as f:
       torch.save(weight, f)
    
if __name__ == '__main__':
   main()
