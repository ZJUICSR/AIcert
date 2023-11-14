# from .test import run
from function.adversarial_test.flow_line.test import run as flow_line_run
from function.adversarial_test.flow_line_rules.test import run as flow_rule_run
from function.adversarial_test.graph_knowledge.test import run as graph_run
from function.adversarial_test.graph_knowledge_rules.test import run as graph_rule_run
from function.adversarial_test.rules.test import run as rule_run

def run_adv_test(model, auto_method, dataloader=None, batch_size=128, eps=16, attack_methods=['fgsm'], 
                 param_hash='', save_path='./output/results',log_func=None, device='cuda',**kwargs):
    model = model.to(device)
    if auto_method == "flow":
        if 'adv_loader' in kwargs:
            defend_info = flow_line_run(model=model,
                attack_methods=attack_methods,
                param_hash=param_hash,
                save_path=save_path,
                log_func=log_func,
                device=device,
                ori_loader=dataloader,
                adv_loader=kwargs['adv_loader'])
        else:
            defend_info = flow_line_run(model=model,
                                        eps=eps,
                                        attack_methods=attack_methods,
                                        param_hash=param_hash,
                                        save_path=save_path,
                                        log_func=log_func,
                                        device=device,
                                        ori_loader=dataloader)
    elif auto_method == "flow_rule": 
        defend_info = flow_rule_run(model=model,
                                    dataloader = dataloader,
                                    eps=eps,
                                    attack_methods=attack_methods,
                                    param_hash=param_hash,
                                    save_path=save_path,
                                    log_func=log_func,
                                    device=device)
    elif auto_method == "graph":
        params = {'attack_mode': kwargs['attack_mode'],
              'attack_type': kwargs['attack_type'],
              'data_type': kwargs['data_type'],
              'defend_algorithm': kwargs['defend_algorithm'],
              'device': device,
              'out_path': save_path}
        print(kwargs)
        defend_info = graph_run(model, dataset=kwargs['dataset'], num_classes=10, test_acc={}, param_hash=param_hash, params=params, log_func=log_func)
    elif auto_method == "graph_rule":
        params = {'attack_mode': kwargs['attack_mode'],
              'attack_type': kwargs['attack_type'],
              'data_type': kwargs['data_type'],
              'defend_algorithm': kwargs['defend_algorithm'],
              'device': device,
              'out_path': save_path}
        defend_info = graph_rule_run(model, dataloader, num_classes=10, test_acc={}, param_hash=param_hash, params=params, log_func=log_func)
    elif auto_method == "rule":
        defend_info = rule_run(model=model,
                      dataloader=dataloader,
                      batch_size=batch_size,
                      eps=eps,
                      attack_methods=attack_methods,
                      param_hash=param_hash,
                      save_path=save_path,
                      log_func=log_func,
                      device=device)
    return defend_info