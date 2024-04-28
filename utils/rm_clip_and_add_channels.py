import torch
pretrained_model_path='../checkpoints/original/model.ckpt'
ckpt_file=torch.load(pretrained_model_path,map_location='cpu')

# add input conv mask channel
new_input_weight=torch.zeros(320,1,3,3)
ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight']=torch.cat((torch.cat((ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'][:,:4], new_input_weight), dim = 1),ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'][:,4:]),dim = 1)

# add input conv pose channel
new_input_weight=torch.zeros(320,8,3,3)
ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight']=torch.cat((ckpt_file['state_dict']['model.diffusion_model.input_blocks.0.0.weight'],new_input_weight),dim=1)

# add output conv mask channel
new_output_weight=torch.zeros(1,320,3,3)
ckpt_file['state_dict']['model.diffusion_model.out.2.weight']=torch.cat((ckpt_file['state_dict']['model.diffusion_model.out.2.weight'],new_output_weight),dim=0)
new_output_bias=torch.zeros(1)
ckpt_file['state_dict']['model.diffusion_model.out.2.bias']=torch.cat((ckpt_file['state_dict']['model.diffusion_model.out.2.bias'],new_output_bias),dim=0)


state_dict = ckpt_file['state_dict']

new_state_dict = {}

for key, value in state_dict.items():
    if not key.startswith('cond_stage_model'):
        new_state_dict[key] = value

ckpt_file['state_dict'] = new_state_dict


torch.save(ckpt_file,"../checkpoints/original/model_prepared.ckpt")

