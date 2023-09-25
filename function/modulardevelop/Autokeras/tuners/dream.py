# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import kerastuner
import numpy as np

from autokeras.engine import tuner as tuner_module


class TrieNode(object):
    def __init__(self):
        super().__init__()
        self.num_leaves = 0
        self.children = {}
        self.hp_name = None

    def is_leaf(self):
        return len(self.children) == 0


class Trie(object):
    def __init__(self):
        super().__init__()
        self.root = TrieNode()

    def insert(self, hp_name):
        names = hp_name.split("/")

        new_word = False
        current_node = self.root
        nodes_on_path = [current_node]
        for name in names:
            if name not in current_node.children:
                current_node.children[name] = TrieNode()
                new_word = True
            current_node = current_node.children[name]
            nodes_on_path.append(current_node)
        current_node.hp_name = hp_name

        if new_word:
            for node in nodes_on_path:
                node.num_leaves += 1

    @property
    def nodes(self):
        return self._get_all_nodes(self.root)

    def _get_all_nodes(self, node):
        ret = [node]
        for key, value in node.children.items():
            ret += self._get_all_nodes(value)
        return ret

    def get_hp_names(self, node):
        if node.is_leaf():
            return [node.hp_name]
        ret = []
        for key, value in node.children.items():
            ret += self.get_hp_names(value)
        return ret


class DreamOracle(kerastuner.Oracle):
    """An oracle combining random search and Dream algorithm.

    It groups the HyperParameters into several categories, namely, HyperGraph,
    Preprocessor, Architecture, and Optimization. The oracle tunes each group
    separately using random search. In each trial, it use a greedy strategy to
    generate new values for one of the categories of HyperParameters and use the best
    trial so far for the rest of the HyperParameters values.

    # Arguments
        initial_hps: A list of dictionaries in the form of
            {HyperParameter name (String): HyperParameter value}.
            Each dictionary is one set of HyperParameters, which are used as the
            initial trials for the search. Defaults to None.
        seed: Int. Random seed.
    """

    def __init__(self, initial_hps=None, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.initial_hps = initial_hps or []
        self._tried_initial_hps = [False] * len(self.initial_hps)

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "initial_hps": self.initial_hps,
                "tried_initial_hps": self._tried_initial_hps,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.initial_hps = state["initial_hps"]
        self._tried_initial_hps = state["tried_initial_hps"]

    # modify
    def get_best_trial_id(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].trial_id
        else:
            return None

    def get_best_hp_dir(self,save_dir):

        import os
        import pickle
        import sys
        sys.path.append('./utils')
        from load_test_utils import check_move
    
        
        trial_id_path=os.path.join(save_dir,'trial_id.pkl')
        if not os.path.exists(trial_id_path):
            return None
        
        trial_id=self.get_best_trial_id()
        if trial_id==None:
            return check_move(save_dir)

        with open(trial_id_path, 'rb') as f:#input,bug type,params
            trial_id_dict = pickle.load(f)
        return trial_id_dict[trial_id]

    def get_previous_dir(self,save_dir,method='last',beam_size=None):
        # last: use the last beam hps

        import os
        import pickle
        import sys
        sys.path.append('./utils')
        from load_test_utils import traversalDir_FirstDir

        trial_id_path=os.path.join(save_dir,'trial_id.pkl')
        if not os.path.exists(trial_id_path):
            cur_num=0
            with open(trial_id_path, 'wb') as f:
                pickle.dump(cur_num, f)
            return []
        else:
            with open(trial_id_path, 'rb') as f:#input,bug type,params
                cur_num = pickle.load(f)
            dir_list=traversalDir_FirstDir(save_dir)
            
            save_dir_list=[]
            if method=='last':
                for dirs in dir_list:
                    base_name=os.path.basename(dirs)
                    if cur_num ==int(base_name.split('-')[0]):
                        save_dir_list.append(dirs)
                cur_num+=1
                with open(trial_id_path, 'wb') as f:
                    pickle.dump(cur_num, f)
                return save_dir_list
            elif method=='best' and beam_size!=None:
                acc_list=[]
                for dirs in dir_list:
                    base_name=os.path.basename(dirs)
                    acc_list.append((float(base_name.split('-')[1]),dirs))
                cur_num+=1
                new_list=sorted(acc_list, key=lambda k: k[0], reverse=True)
                for i in range(min(beam_size,len(new_list))):
                    save_dir_list.append(new_list[i][1])
                with open(trial_id_path, 'wb') as f:
                    pickle.dump(cur_num, f)
                return save_dir_list


    def obtain_new_hps(self,
                        save_dir='./Test_dir/demo_result'):

        # a greedy sequential search, demo of form

        import os
        import pickle
        import time
        import sys
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict,update_candidate,select_action
        start=time.time()

        new_save_dir=self.get_best_hp_dir(save_dir)
        
        if new_save_dir==None:
            return None
        
        # step 1:# read current state and judge the training condition, return
        
        algw_path=os.path.join(new_save_dir,'algw.pkl')

        tmp_t=0
        while tmp_t<10:
            if os.path.exists(algw_path):
                with open(algw_path, 'rb') as f:#input,bug type,params
                    algw = pickle.load(f)
                break
            else:
                tmp_t+=1
                time.sleep(10)
        print('==={}==='.format(tmp_t))
        if tmp_t>=10:
            print("=========No ALGW!!==========")
            arch,loss,grad,wgt=judge_dirs(new_save_dir)
            algw="{}-{}-{}-{}".format(arch,loss,grad,wgt)
            print(algw)
        # step 2:# load evaluation results, if not return None,or get the operation and corresponding weights
        opt_wgt_dict,opt_list=load_evaluation(algw,evaluation_pkl=os.path.abspath('./utils/priority_all_0113.pkl'))
        
        time1=time.time()
        print(time1-start)
        if opt_wgt_dict==None:
            print(1)
            return None

        # step 3:

        values=self.generate_hp_values(opt_list,save_dir=save_dir)

        print('================We have generated the values! Ready to TRAIN================')
        return values

    def obtain_beam_hps(self,
                        save_dir='./Test_dir/demo_result',beam_size=None,seed_size=None):

        # if use beam search, we need to specify the seed size and the beam size

        import os
        import pickle
        import time
        import sys
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict,update_candidate,select_action,choose_random_select
        start=time.time()

        # new_save_dir_list=self.get_previous_dir(save_dir)
        new_save_dir_list=self.get_previous_dir(save_dir,method='best',beam_size=3)# choose 3 best trial
        
        # new_save_dir=check_move(save_dir)
        # if no save dir, return None
        candidate_dict_path=os.path.join(save_dir,'candidate.pkl')
        if new_save_dir_list==[]:
            # generate random seeds to start training
            # values_list=self.generate_random_hp_list(seed_size=seed_size)# list, element is (value,modify)
            
            # with open(candidate_dict_path, 'wb') as f:
            #     pickle.dump(values_list, f)
            # return values_list[0][0][0]

            return None
        
        # step 1:# read current state and judge the training condition, return
        # algw contidion sign
        
        if os.path.exists(candidate_dict_path):
            os.remove(candidate_dict_path)

        for new_save_dir in new_save_dir_list:
        
            algw_path=os.path.join(new_save_dir,'algw.pkl')
            if os.path.exists(algw_path):
                with open(algw_path, 'rb') as f:#input,bug type,params
                    algw = pickle.load(f)
            else:
                arch,loss,grad,wgt=judge_dirs(new_save_dir)
                algw="{}-{}-{}-{}".format(arch,loss,grad,wgt)
            # step 2:# load evaluation results, if not return None,or get the operation and corresponding weights
            # opt_wgt_dict,opt_list=load_evaluation(algw)
            # print('Architecture Condition is {}; Convergence Condition is {}; Gradient Condition is {}; Weight Condition is {}'.format(arch,loss,grad,wgt))
            opt_wgt_dict,opt_list=load_evaluation(algw,evaluation_pkl=os.path.abspath('./utils/priority_all.pkl'))

            # if opt_wgt_dict==None:
            #     print(1)
                # return None# bug return None tuner not fix

        
            # combine opt with model path, save the dict in candidate dict path
            update_candidate(opt_wgt_dict,opt_list,new_save_dir,candidate_dict_path,beam_size=beam_size)
        # choose actions and model
        action_list=select_action(candidate_dict_path,beam_size=beam_size)
        if action_list==None:
            return None # for the special conditions that no priority for the training..
        # generate new values list(modify previous function)
        random_select=choose_random_select(new_save_dir_list)# adjust random select value according to the best history accuracy
        values_list=self.generate_hp_values(action_list,beam_size=beam_size,random_select=random_select)# list, element is (value,modify)
        values=values_list[0][0]
        for j in range(beam_size):
            values_list[j]=(values_list[j],0)# 0 is status sign, if trained, turn to 1.
        with open(candidate_dict_path, 'wb') as f:
            pickle.dump(values_list, f)

        
        
        print('================We have generated the values! Ready to TRAIN================')

        return values

    def _get_best_action(self,
                        best_hps,
                        operation_list,
                        collision=0,
                        collision_list=[],
                        best_hash_path='./Test_dir/demo_result/hash.pkl',
                        method='normal'):

        # modify
        # 0111 add new hps here
        additional_hp_list=['step_1_ratio','step_2_lr_scale','step_1_freeze','end_learning_rate','weight_decay_rate','momentum','multi_step','triple_train']
        
        import os
        import pickle
        import sys
        import copy
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict,append_list

        
        best_hps_key_list=list(best_hps.values.keys())
        best_hps_hash=self._compute_values_hash(best_hps.values)
        if not os.path.exists(best_hash_path):
            best_hash_dict={}
        else:
            with open(best_hash_path, 'rb') as f:#input,bug type,params
                best_hash_dict = pickle.load(f)
        best_hp_name=None
        best_hp_value=None

        # clean operation list based on collision_list
        
        tmp_operation_list=copy.deepcopy(operation_list)
        if collision_list!=[]:
            for coll in collision_list:
                for opt in tmp_operation_list:
                    # operation_list.remove(str(coll[0].split('/')[-1])+'-'+str(coll[1]))
                    action=opt.split('-')[0]
                    value=opt.replace('{}-'.format(action), '')
                    if action in coll[0] and value==str(coll[1]):
                        operation_list.remove(opt)

        if best_hps_hash not in best_hash_dict.keys():
            for opt in operation_list:
                
                if collision>0 and collision_list==[]:
                    collision-=1
                    print("======Collision {}====".format(opt))
                    continue
                if len(opt.split('-'))==3:#TODO: back
                    continue
                action=opt.split('-')[0]
                value=opt.replace('{}-'.format(action), '')
                if special_action(action):
                    append_list(best_hps_hash,best_hash_dict,opt,best_hash_path)
                    print('======0======')
                    return action, value, "Special"

                # add special determine for new hp like step_1_ratio
                if action in additional_hp_list:
                    best_hp_value=get_true_value(value)
                    append_list(best_hps_hash,best_hash_dict,opt,best_hash_path)
                    print('======1======')
                    return action,best_hp_value,None

                # if action not in key, we need find the parent and modify
                # parent and itself:
                with open(os.path.abspath('./utils/hp_relation.pkl'), 'rb') as f:#input,bug type,params
                    relation = pickle.load(f)

                for hp_key in relation.keys():
                    if action in hp_key:
                        if len(relation[hp_key]['parents'])<2:
                            continue
                        else:
                            true_pare=True
                            modify_pare_list=[]
                            for pare in relation[hp_key]['parents']:
                                hp_name=pare.split('--')[0]
                                if hp_name not in best_hps_key_list:
                                    true_pare=False
                                    break
                                pare_value=pare.split('--')[1]
                                # special values
                                if pare_value=='1':
                                    pare_value=True
                                elif pare_value=='0':
                                    pare_value==False
                                if best_hps.values[hp_name]!=pare_value:
                                    # best_hps.values[hp_name]=pare_value
                                    modify_pare_list.append((hp_name,pare_value))
                            if true_pare and modify_pare_list!=[]:
                                best_hp_value=get_true_value(value)
                                return hp_key,best_hp_value,None

                for key in best_hps_key_list:
                    if action in key:
                        best_hp_name=key
                        best_hp_value=get_true_value(value)
                        if best_hps.values[key]==best_hp_value:
                            # continue
                            break
                        append_list(best_hps_hash,best_hash_dict,opt,best_hash_path)
                        print('======2======')
                        return best_hp_name,best_hp_value,None

        else:
            for opt in operation_list:
                if opt in best_hash_dict[best_hps_hash]:
                    continue
                if collision>0:
                    collision-=1
                    continue
                if len(opt.split('-'))==3:#TODO: back
                    continue
                action=opt.split('-')[0]
                value=opt.replace('{}-'.format(action), '')
                
                if action in additional_hp_list:
                    best_hp_value=get_true_value(value)
                    append_list(best_hps_hash,best_hash_dict,opt,best_hash_path)
                    print('======3======')
                    return action,best_hp_value,None
                
                if special_action(action):
                    append_list(best_hps_hash,best_hash_dict,opt,best_hash_path)
                    print('======4======')
                    return action, value, "Special"
                for key in best_hps_key_list:
                    if action in key:
                        best_hp_name=key
                        best_hp_value=get_true_value(value)
                        if best_hps.values[key]==best_hp_value:
                            continue
                        append_list(best_hps_hash,best_hash_dict,opt,best_hash_path)
                        print('======5======')
                        return best_hp_name,best_hp_value,None

    
    def generate_hp_values(self,operation_list,beam_size=None,random_select=0.2,save_dir='./Test_dir/demo_result'):
        # modify

        import os
        import pickle
        import random
        import sys
        sys.path.append('./utils')
        from load_test_utils import judge_dirs,load_evaluation,check_move,get_true_value,special_action,write_opt,sort_opt_wgt_dict,multi_action_search,get_opti_value
        additional_hp_list=['step_1_ratio','step_2_lr_scale','step_1_freeze','end_learning_rate','weight_decay_rate','momentum','triple_train']

        save_log_path=os.path.join(os.path.abspath(save_dir),'log.pkl')
        hash_path=os.path.join(os.path.abspath(save_dir),'hash.pkl')
        if beam_size==None:
            self._max_collision=10
            with open(save_log_path, 'rb') as f:#input,bug type,params
                log_dict = pickle.load(f)
       
            tmp_status,tmp_values=get_opti_value(log_dict)
            if tmp_status:
                return tmp_values

            best_hps = self._get_best_hps()

            collisions = 0
            collision_list=[]
            while True:
                best_hp_name,best_hp_value,special_sign=self._get_best_action(best_hps,operation_list,collisions,collision_list,best_hash_path=hash_path)
                print('=======best_hp:{}, Value:{}=========='.format(best_hp_name,best_hp_value))
                best_hps=self.prepare_parents(best_hps,best_hp_name,best_hp_value)
                
                try:
                    if best_hps[best_hp_name]==best_hp_value:
                        values=best_hps.values
                        values=multi_action_search(values,best_hp_name,best_hp_value)
                        values_hash = self._compute_values_hash(values)
                        if values_hash in self._tried_so_far:
                            print("======collision! {}-{}=======".format(best_hp_name,best_hp_value))
                            collision_list.append((best_hp_name,best_hp_value))

                            if 'collision_list' not in log_dict.keys():
                                log_dict['collision_list']=[]
                            log_dict['collision_list'].append((values,values_hash))
                            with open(save_log_path, 'wb') as f:
                                pickle.dump(log_dict, f)

                            collisions += 1
                            if collisions <= self._max_collisions:
                                continue
                            return None
                        self._tried_so_far.add(values_hash)
                        break
                except Exception as e:
                    print("========Error in Multi action=========")
                    print(e)
                
                if special_sign !=None:
                    print("====special_sign===========")
                    write_opt(best_hp_name,best_hp_value)
                    return best_hps.values
                hps = kerastuner.HyperParameters()
                # Generate a set of random values.
                trigger_count=0# if over 1 in generation(error situation), then use greedy
                for hp in self.hyperparameters.space:
                    hps.merge([hp])
                    # if not active, do nothing.
                    # if active, check if selected to be changed.
                    if hps.is_active(hp):
                        # if was active and not selected, do nothing.
                        if best_hps.is_active(hp.name) and hp.name != best_hp_name:
                            hps.values[hp.name] = best_hps.values[hp.name]
                            continue
                        # if was not active or selected, sample.
                        elif hp.name == best_hp_name:
                            hps.values[hp.name] = best_hp_value#hp.random_sample(self._seed_state)
                            print("=========={}-Trigger==========".format(hp.name))
                            trigger_count+=1
                        else:
                            
                            if hp.name in additional_hp_list:
                                print('==========DefaultValue:{}============'.format(hp.name))
                                continue
                            print('==========RandomValue:{}============'.format(hp.name))
                            hps.values[hp.name] = hp.random_sample(self._seed_state)
                            
                            
                        if trigger_count>1:
                            print("============Multi_trigger=======")
                            return None
                        self._seed_state += 1
                        
                # 0111 special change for some hp not in existing hps
                if trigger_count==0 and best_hp_name not in hps.values.keys():
                    hps.values[best_hp_name]=best_hp_value
                
                values = hps.values
                try:
                    values=multi_action_search(values,best_hp_name,best_hp_value)
                except:
                    print("========Error in Multi action=========")
                # Keep trying until the set of values is unique,
                # or until we exit due to too many collisions.
                values_hash = self._compute_values_hash(values)
                if values_hash in self._tried_so_far:
                    print("======collision! {}-{}=======".format(best_hp_name,best_hp_value))
                    collision_list.append((best_hp_name,best_hp_value))

                    if 'collision_list' not in log_dict.keys():
                        log_dict['collision_list']=[]
                    log_dict['collision_list'].append((values,values_hash))
                    with open(save_log_path, 'wb') as f:
                        pickle.dump(log_dict, f)

                    collisions += 1
                    print("======collision! {}-{}=======".format(best_hp_name,best_hp_value))
                    if collisions <= self._max_collisions:
                        continue
                    return None
                self._tried_so_far.add(values_hash)
                break
            # print(trigger_count)
            return values
        else:
            value_list=[]
            for opt in operation_list:
                if opt==None:
                    # None means algw is not covered
                    hp_names = self._select_hps()
                    values = self._generate_hp_values(hp_names)
                    value_list.append((values,'random'))
                    continue
                # best_hps = self._get_best_hps()
                hps_path=os.path.join(os.path.dirname(opt[0]),'param.pkl')
                with open(hps_path, 'rb') as f:#input,bug type,params
                    best_hps = pickle.load(f)
                action_list=opt[2]
                while 1:
                    if action_list[0]!=opt[1]:
                        action_list.remove(action_list[0])
                    else:
                        break
                

                collisions = 0
                collision_list=[]
                while True:
                    
                    best_hp_name,best_hp_value,special_sign=self._get_best_action(best_hps,action_list,collisions,collision_list,best_hash_path=hash_path)
                    print('=======best_hp:{}, Value:{}=========='.format(best_hp_name,best_hp_value))
                    best_hps=self.prepare_parents(best_hps,best_hp_name,best_hp_value)
                    # if new_best_hps.values!=best_hps.values:
                    try:
                        if best_hps[best_hp_name]==best_hp_value:
                            values=multi_action_search(values,best_hp_name,best_hp_value)
                            values_hash = self._compute_values_hash(values)
                            if values_hash in self._tried_so_far:
                                collisions += 1
                                print("======collision! {}-{}=======".format(best_hp_name,best_hp_value))
                                collision_list.append((best_hp_name,best_hp_value))
                                if collisions <= self._max_collisions:
                                    continue
                                return None
                            self._tried_so_far.add(values_hash)
                            break
                    except Exception as e:
                        print("========Error in Multi action=========")
                        print(e)
                    # else:
                    if special_sign !=None:
                        print("====special_sign===========")
                        write_opt(best_hp_name,best_hp_value)
                        return best_hps.values
                    hps = kerastuner.HyperParameters()
                    # Generate a set of random values.
                    trigger_count=0# if over 1 in generation(error situation), then use greedy
                    for hp in self.hyperparameters.space:
                        hps.merge([hp])
                        # if not active, do nothing.
                        # if active, check if selected to be changed.
                        if hps.is_active(hp):
                            # if was active and not selected, do nothing.
                            if best_hps.is_active(hp.name) and hp.name != best_hp_name and random.random()>random_select:# add random
                                hps.values[hp.name] = best_hps.values[hp.name]
                                continue
                            # if was not active or selected, sample.
                            elif hp.name == best_hp_name:
                                hps.values[hp.name] = best_hp_value#hp.random_sample(self._seed_state)
                                trigger_count+=1
                            else:
                                hps.values[hp.name] = hp.random_sample(self._seed_state)
                            
                            if trigger_count>1:
                                print("============Multi_trigger=======")
                                return None
                            self._seed_state += 1

                    if trigger_count==0 and best_hp_name not in hps.values.keys():
                        hps.values[best_hp_name]=best_hp_value
                    values = hps.values
                    # Keep trying until the set of values is unique,
                    # or until we exit due to too many collisions.

                    # modify: add pretrain and trainable for xception
                    try:
                        values=multi_action_search(values,best_hp_name,best_hp_value)
                    except:
                        print("========Error in Multi action=========")

                    values_hash = self._compute_values_hash(values)
                    if values_hash in self._tried_so_far:
                        print("======collision!=======")
                        collisions += 1
                        if collisions <= self._max_collisions:
                            continue
                        return None
                    self._tried_so_far.add(values_hash)
                    break
                # print(trigger_count)
                # return values
                value_list.append((values,"{}--{}".format(best_hp_name,best_hp_value)))
            return value_list

    def prepare_parents(self,best_hps,best_hp_name,best_hp_value):
        additional_hp_list=['step_1_ratio','step_2_lr_scale','step_1_freeze','end_learning_rate','weight_decay_rate','momentum','multi_step','triple_train']
        import os
        import pickle
        with open(os.path.abspath('./utils/hp_relation.pkl'), 'rb') as f:#input,bug type,params
            relation = pickle.load(f)
        best_hps_key_list=list(best_hps.values.keys())
        for hp_key in relation.keys():
            if best_hp_name == hp_key:
                
                    
                if len(relation[hp_key]['parents'])<2 and hp_key not in additional_hp_list:
                    return best_hps
                else:
                    modify_pare_list=[]
                    for pare in relation[hp_key]['parents']:
                        hp_name=pare.split('--')[0]
                        if hp_name not in best_hps_key_list and hp_name != 'multi_step':
                            return best_hps
                        if hp_name=='multi_step':
                            tmp_value_list=list(best_hps.values.keys())
                            for key in tmp_value_list:
                                if '/pretrained' in key:
                                    best_hps.values[key]=True
                                    trainable_setting=key.replace('/pretrained','/trainable')
                                    best_hps.values[trainable_setting]=True
                                    best_hps.values[hp_name]=True
                            best_hps.values[best_hp_name]=best_hp_value
                        pare_value=pare.split('--')[1]
                        # special values
                        if pare_value=='1':
                            pare_value=True
                        elif pare_value=='0':
                            pare_value==False
                        if best_hps.values[hp_name]!=pare_value:
                            # best_hps.values[hp_name]=pare_value
                            modify_pare_list.append((hp_name,pare_value))
                    if modify_pare_list!=[]:
                        for modi_p in modify_pare_list:
                            best_hps.values[modi_p[0]]=modi_p[1]
                        best_hps.values[best_hp_name]=best_hp_value
                        if 'trainable' in best_hp_name:
                            best_hps.values['learning_rate']=0.0001
                            # use special learning rate
                        return best_hps

        if best_hp_name=='multi_step' and best_hp_value==True:
            tmp_value_list=list(best_hps.values.keys())
            for key in tmp_value_list:
                if '/pretrained' in key:
                    best_hps.values[key]=True
                    trainable_setting=key.replace('/pretrained','/trainable')
                    best_hps.values[trainable_setting]=True
                    best_hps.values[best_hp_name]=best_hp_value
                    return best_hps
        print('No relation now')
        return best_hps
                



    def generate_random_hp_list(self,seed_size):
        value_list=[]
        for i in range(seed_size):
            hp_names = self._select_hps()
            values = self._generate_hp_values(hp_names)
            value_list.append(((values,0),0))
        return value_list
    # modify
    
    def _select_hps(self):#TODO::
        trie = Trie()
        best_hps = self._get_best_hps()
        for hp in best_hps.space:
            # Not picking the fixed hps for generating new values.
            if best_hps.is_active(hp) and not isinstance(
                hp, kerastuner.engine.hyperparameters.Fixed
            ):
                trie.insert(hp.name)
        all_nodes = trie.nodes

        if len(all_nodes) <= 1:
            return []

        probabilities = np.array([1 / node.num_leaves for node in all_nodes])
        sum_p = np.sum(probabilities)
        probabilities = probabilities / sum_p
        node = np.random.choice(all_nodes, p=probabilities)

        return trie.get_hp_names(node)

    def _next_initial_hps(self):
        for index, hps in enumerate(self.initial_hps):
            if not self._tried_initial_hps[index]:
                self._tried_initial_hps[index] = True
                return hps

    # def _populate_space(self, trial_id):
    #     if not all(self._tried_initial_hps):
    #         values = self._next_initial_hps()
    #         return {
    #             "status": kerastuner.engine.trial.TrialStatus.RUNNING,
    #             "values": values,
    #         }

    #     for i in range(self._max_collisions):
    #         hp_names = self._select_hps()
    #         values = self._generate_hp_values(hp_names)
    #         # Reached max collisions.
    #         if values is None:
    #             continue
    #         # Values found.
    #         return {
    #             "status": kerastuner.engine.trial.TrialStatus.RUNNING,
    #             "values": values,
    #         }
    #     # All stages reached max collisions.
    #     return {
    #         "status": kerastuner.engine.trial.TrialStatus.STOPPED,
    #         "values": None,
    #     }

    # modify 
    def _populate_space(self, trial_id,root_path=None):
        if not all(self._tried_initial_hps):
            values = self._next_initial_hps()
            return {
                "status": kerastuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }

        for i in range(self._max_collisions):
            if root_path!=None:
                values=self.obtain_new_hps(save_dir=root_path)
            else:
                values=self.obtain_new_hps()

            if values==None: # worst: use greedy
                print('============Fail to Generate!!!! USE GREEDY Method NOW!!!============')
                hp_names = self._select_hps()
                values = self._generate_hp_values(hp_names)
            # Reached max collisions.
            if values is None:
                continue
            # Values found.
            return {
                "status": kerastuner.engine.trial.TrialStatus.RUNNING,
                "values": values,
            }
        # All stages reached max collisions.
        return {
            "status": kerastuner.engine.trial.TrialStatus.STOPPED,
            "values": None,
        }
    # modify

    def _get_best_hps(self):
        best_trials = self.get_best_trials()
        if best_trials:
            return best_trials[0].hyperparameters.copy()
        else:
            return self.hyperparameters.copy()

    def _generate_hp_values(self, hp_names): 
        best_hps = self._get_best_hps()

        collisions = 0
        while True:
            hps = kerastuner.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                # if not active, do nothing.
                # if active, check if selected to be changed.
                if hps.is_active(hp):
                    # if was active and not selected, do nothing.
                    # modify modify architecture
                    if hp.name=='image_block_1/block_type':
                        hps.values[hp.name] = hp.random_sample(self._seed_state)
                        continue
                    if best_hps.is_active(hp.name) and hp.name not in hp_names:
                        hps.values[hp.name] = best_hps.values[hp.name]
                        continue
                    # if was not active or selected, sample.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                print("======collision!=======")
                collisions += 1
                if collisions <= self._max_collisions:
                    continue
                return None
            self._tried_so_far.add(values_hash)
            break
        return values


class Dream(tuner_module.AutoTuner):
    def __init__(
        self,
        hypermodel: kerastuner.HyperModel,
        objective: str = "val_loss",
        max_trials: int = 10,
        initial_hps: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        hyperparameters: Optional[kerastuner.HyperParameters] = None,
        tune_new_entries: bool = True,
        allow_new_entries: bool = True,
        **kwargs
    ):
        self.seed = seed
        oracle = DreamOracle(
            objective=objective,
            max_trials=max_trials,
            initial_hps=initial_hps,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)

# modify 
# def judge_dirs(target_dir):
#     params_path=os.path.join(target_dir,'param.pkl')
#     gw_path=os.path.join(target_dir,'gradient_weight.pkl')
#     his_path=os.path.join(target_dir,'history.pkl')

#     with open(params_path, 'rb') as f:#input,bug type,params
#         hyperparameters = pickle.load(f)
#     with open(his_path, 'rb') as f:#input,bug type,params
#         history = pickle.load(f)
#     with open(gw_path, 'rb') as f:#input,bug type,params
#         gw = pickle.load(f)

#     arch=get_arch(hyperparameters)
#     loss=get_loss(history)
#     grad,wgt=get_gradient(gw)
    

#     return "{}-{}-{}-{}".format(arch,loss,grad,wgt)