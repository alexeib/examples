import gc
import torch

# Code by James Bradbury - https://github.com/jekbradbury

def print_obj_tree(min_elements=None):
    objlist = [obj for obj in gc.get_objects() if torch.is_tensor(obj) or isinstance(obj, torch.autograd.Variable)]
    for obj in objlist:
        if min_elements and obj.nelement() < min_elements: continue
        referrers = [r for r in gc.get_referrers(obj) if r is not objlist]
        print(f'{id(obj)} {obj.__class__.__qualname__} of size {tuple(obj.size())} with references held by:')
        for referrer in referrers: 
            if torch.is_tensor(referrer) or isinstance(referrer, torch.autograd.Variable):
                infostr = f' of size {tuple(referrer.size())}'
            #elif type(referrer) == list: #isinstance(referrer, list):
            #        infostr = f' in which it is indexed {referrer.index(obj)}'
            elif isinstance(referrer, dict):
                infostr = f' in which its key is {next(k for k, v in referrer.items() if v is obj)}'
            else:
                infostr = ''
            print(f'  {id(referrer)} {referrer.__class__.__qualname__}{infostr}')

