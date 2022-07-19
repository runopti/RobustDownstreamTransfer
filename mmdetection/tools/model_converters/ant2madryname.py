import torch
import torch.nn as nn
import argparse
from collections import OrderedDict

def convert_advres(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        #if k.endswith('fc.weight') or k.endswith('fc.bias'):
        new_v = v
        new_k = k #.replace('module.', '')
        new1_k = 'module.model.' + new_k 
        new2_k = 'module.attacker.model.' + new_k 
        new_ckpt[new1_k] = v
        new_ckpt[new2_k] = v
    new_ckpt['module.normalizer.new_std'] = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
    new_ckpt['module.normalizer.new_mean'] = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
    new_ckpt['module.attacker.normalize.new_std'] = torch.tensor([0.229, 0.224, 0.225]).reshape(3,1,1)
    new_ckpt['module.attacker.normalize.new_mean'] = torch.tensor([0.485, 0.456, 0.406]).reshape(3,1,1)
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
   elif 'model_state_dict' in checkpoint:
       state_dict = checkpoint['model_state_dict']
   else:
       state_dict = checkpoint

   weight = convert_advres(state_dict)

    
   checkpoint_new = {}
   checkpoint_new['model'] = weight
   checkpoint_new['epoch'] = 0
   with open(args.dst, 'wb') as f:
       torch.save(checkpoint_new, f)
    
if __name__ == '__main__':
   main()
