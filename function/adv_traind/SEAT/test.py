from seat import seat

args_dict = {'epochs': 120,
             'arch': 'resnet18',
             'num_classes': 10,
             'lr': 0.01,
             'loss_fn': 'cent',
             'attack_method': 'pgd',
             'attack_method_list': 'pgd',
             'log_step': 7,
             'num_classes': 10,
             'epsilon': 0.031,
             'num_steps': 10,
             'step_size': 0.007,
             'resume': False,
             'out_dir': "/data/user/WZT/models/SEAT_out",
             'ablation': ''}


seat(args_dict)


