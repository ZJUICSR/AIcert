# -*- coding: utf-8 -*-
import sys
import argparse
import os
import pickle
import platform
import pandas
import re
import time
import pickle
# import winreg
import json

def closeReg(key):
    import winreg
    winreg.CloseKey(key)


def openReg(key):
    import winreg
    return winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)


def queryVal(key, name):
    import winreg
    (value, type) = winreg.QueryValueEx(key, name)
    return value


def find(keypath, softname):
    import winreg
    key = openReg(keypath)
    i = 0
    result = ""
    try:
        while 1:
            name = winreg.EnumKey(key, i)
            path = keypath + "\\" + name
            subkey = openReg(path)
            try:
                value = queryVal(subkey, 'DisplayName')
                print('{}\n'.format(value))
                if softname in value:
                    value = queryVal(subkey, 'DisplayVersion')
                    result = value
                    closeReg(subkey)
                    break
            except:
                pass
            finally:
                closeReg(subkey)
            i += 1
    except:
        pass
    closeReg(key)
    return result

def get_previous_dict(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            tmp = pickle.load(f)
    else:
        tmp={}
    return tmp

def cut_redundancy_part(key,value):
    slice_list=key.split(' ')
    for s in slice_list:
        if '.'in s and s in value:
            slice_list.remove(s)
    new_key=' '.join(slice_list)
    return new_key,value

def add_software_dict(key,value,tmp_dict):
    dict_key=list(tmp_dict.keys())
    if key not in dict_key:
        key,value=cut_redundancy_part(key,value)
        tmp_dict[key]=value
    return tmp_dict


def read_output(software_dict,tmp_save_path):
    f = open(tmp_save_path, 'r')
    lines=f.readlines()[1:]
    for line in lines:
        line_list=line.split(',now ')
        if len(line_list)==2:
            value=line_list[0].split(r'/')[0]
            version=line_list[1].split(' ')[0]
        else:
            line_list=line.split('/now ')
            value=line_list[0]
            version=line_list[1].split(' ')[0]

        if value not in software_dict.keys():
            software_dict[value]=version
    return software_dict

def read_centos_output(software_dict,tmp_save_path):
    f = open(tmp_save_path, 'r')
    lines=f.readlines()[1:]
    for line in lines:
        line=line.strip()
        line_list=line.split('-')
        for l in range(len(line_list)):
            if l==0: continue
            if '.' not in line_list[l]:
                continue
            else:
                version=line_list[l]
                value='-'.join(line_list[:l])
                if value not in software_dict.keys():
                    software_dict[value]=version
                break
    return software_dict



def traversalDir_FirstFile(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isfile(m)):
                tmplist.append(m)
    return tmplist

def traversalDir_FirstDir(path):
    tmplist = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path,file1)
            if (os.path.isdir(m)):
                tmplist.append(m)
    return tmplist



def sort_path_list(cve_path_list):
    cve_path_list.sort(key=lambda d:int(d.split('.csv')[0].split('/')[-1]))
    cve_path_list.reverse()# return a list and the newest year is at first.
    return cve_path_list[:1]

def update_potential_val(potential_val,key,des,cve_num):
    # print(1)
    if key not in potential_val.keys():
        potential_val[key]={}
    potential_val[key][cve_num]=des
    return potential_val

def judge_version(des,cur_version):
    # 匹配正则获取版本号
    searchObj = re.search( r'\s([0-9]+\.)+([0-9]+)\s', des, re.M|re.I)
    if not searchObj:
        return False
    bug_version=searchObj.group().strip()

    searchObj = re.search( r'([0-9]+\.)+([0-9]+)', cur_version, re.M|re.I)
    cur_version=searchObj.group()

    if not searchObj:
        print('error')
    cur_version=searchObj.group().strip()
    #对比记录的版本
    older_sign=False
    count=0
    bv=bug_version.split('.')
    cv=cur_version.split('.')
    max_count=min(len(bv),len(cv))
    while not older_sign:
        tmp_b=int(bv[count])
        tmp_c=int(cv[count])
        if tmp_b==tmp_c:
            count+=1
        elif tmp_b<tmp_c:
            return False
        elif tmp_b>tmp_c:
            return True
        if count>=max_count:
            if count>len(bv):
                return False
            else: return True

    # #返回是否TF
    # pass

def search_update(potential_val,df,val_dict):
    for key,value in val_dict.items():
        length=len(df)
        for l in range(length):
            des=df.iloc[l]['Description']
            name=df.iloc[l]['Name']
            if key in des:
                if judge_version(des,value):
                    potential_val=update_potential_val(potential_val,key,des,name)

    return potential_val


def judge_vulnerable(val_dict,cve_dir):
    potential_val={}
    cve_path_list=traversalDir_FirstFile(cve_dir)
    sort_cve_list=sort_path_list(cve_path_list)
    for cve in sort_cve_list:
        df = pandas.read_csv(cve)
        potential_val=search_update(potential_val,df,val_dict)
        print('finish {}'.format(cve))
        # length=len(df)
        # for l in range(length):

        #     if year in df.iloc[l]['Name']:
        #         year_path=os.path.join(args.output_dir,'{}.csv'.format(year))
        #         if not os.path.exists(year_path):
        #             df[l:l+1].to_csv(year_path,mode='w', header=True,index=None)
        #         else:
        #             df[l:l+1].to_csv(year_path,mode='a', header=False,index=None)

    return potential_val

def judge_vulnerable_new(val_dict,cve_path,method='hard'):
    potential_val={}
    with open(cve_path, 'rb') as f:
        cve_dict = pickle.load(f)
    if method=='soft':
        for key in cve_dict:
            for val in val_dict:
                for p in cve_dict[key]['product_list']:
                    if val in p:
                        for ver in cve_dict[key]['version']:
                            if judge_version(ver,val_dict[val]):
                                potential_val=update_potential_val(potential_val,val,cve_dict[key]['description'],key)
    elif method=='hard':
        for key in cve_dict:
            for val in val_dict:
                if val in cve_dict[key]['product_list']:
                    for ver in cve_dict[key]['version']:
                        if judge_version(ver,val_dict[val]):
                            potential_val=update_potential_val(potential_val,val,cve_dict[key]['description'],key)
    print('finish')
    return potential_val

def env_detection(args):
    
    keypath1 = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    keypath2 = r"SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall"

    print("--------------Start Extract System Information--------------")
    info=get_sys_info()
    # info is the system msg
    print("--------------Start Extract Lib Version--------------")
    if info=={}:
        os._exit(0)
    elif info['system']=='Windows':
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        val = win_get_software_list(keypath1, self.save_path)
        if val == "":
            val = win_get_software_list(keypath2, self.save_path)
    elif info['system']=='Linux':
        if 'ubuntu' in info['sys_version'].lower():
            val= ubuntu_get_software_list(self.save_path)
        elif 'debian' in info['sys_version'].lower():
            pass
    else:
        print('Not support System')
        os._exit(0)
    # val is the sys software version
    sys_msg={}
    sys_msg['software_lib_list']=val
    sys_msg['env_info']=info
    sys_msg['description']="The system architecture is {} with total of {} libs.\nThe detailed infomation of environment is shown in `env_info`.\
         And the libs and softwares in the system is shown in `Software_lib_list`.".format(info['system'],len(val))
    # save system msg
    with open(os.path.join(self.save_dir,'system_message.pkl'),'wb') as f:
        pickle.dump(sys_msg,f)
    

    start=time.time()
    print("--------------Start Detect System Vulnerable--------------")
    if method!='all':
        potential_problem=judge_vulnerable_new(val,self.cve_path,method=method)
        import pickle
        with open(os.path.join(self.save_dir,'potential_problem_{}_new.pkl'.format(method)),'wb') as f:
            pickle.dump(potential_problem,f)
            print(time.time()-start)
        report_dict={}
        report_dict['Risk']={}
        potential_lib_list=list(potential_problem.keys())
        report_dict['Risk']['Libs']=potential_lib_list
        report_dict['Risk']['CVE Reports']=potential_problem
        report_dict['Risk']['Test Method']=method
        report_dict['Risk']['description']="This dict shows the risk of the libs in current environment. We have list the lib name and the correspoding CVE reports for reference."

    else:
        hard_problem=judge_vulnerable_new(val,self.cve_path,method='hard')
        potential_problem=judge_vulnerable_new(val,self.cve_path,method='soft')
        import pickle
        with open(os.path.join(self.save_dir,'potential_problem_{}_new.pkl'.format(method)),'wb') as f:
            pickle.dump(potential_problem,f)
            print(time.time()-start)
        
        report_dict={}
        report_dict['Confirmable Risk']={}
        report_dict['Potential Risk']={}
        risk_lib_list=list(hard_problem.keys())
        report_dict['Confirmable Risk']['Libs']=risk_lib_list
        report_dict['Confirmable Risk']['CVE Reports']=hard_problem
        potential_lib_list=list(potential_problem.keys())
        report_dict['Potential Risk']['Libs']=potential_lib_list
        report_dict['Potential Risk']['CVE Reports']=potential_problem
        report_dict['Confirmable Risk']['description']="The Libs in this dict shows the list of the risk lib that can be comfirmed by the CVE reports. The CVE Reports dict shows the corresponding CVE report of each libs."
        report_dict['Potential Risk']['description']="This dict shows the potential risk of the libs in current environment. We have list the lib name and the correspoding CVE reports for reference."
        json_path=os.path.join(self.save_dir,'tmp.json')
        with open(json_path,"w") as f:
            f.write(json.dumps(report_dict,ensure_ascii=False,indent=4,separators=(',',':')))
    
    print("--------------Finish!--------------")

# def code_check(dir_check,dir_log):
#     from static_check import batch_check,save_json
#     import time
#     dir_to_check = dir_check
#     check_info_dir = dir_log

#     time_start = time.time()
#     line_counts, num_problems, num_vuls = batch_check(dir_to_check, check_info_dir)
#     time_end = time.time()
#     time_cost = time_end - time_start

#     time_cost_str = "Time costs :"+str(time_cost)+"s"
#     line_conunts_str = "Lines: {}".format(line_counts)
#     speed_str = "Speed: {:.2f} lines/s ({:d} lines/hour)".format(line_counts*1. / time_cost, int(line_counts*1. / time_cost*3600))

#     print("*" * 20)
#     print(time_cost_str)
#     print(line_conunts_str)
#     print(speed_str)
#     print("{} problem(s) found ({} vulnerability(ies).)".format(num_problems, num_vuls))

#     output_json = {}
#     output_json['file'] = dir_to_check
#     output_json['time_cost'] = time_cost
#     output_json['lines'] = line_counts
#     output_json['speed'] = int(line_counts * 1. / time_cost * 3600)
#     output_json['message'] =("{} problem(s) found ({} vulnerability(ies).)".format(num_problems, num_vuls))

#     save_json([output_json], os.path.join(check_info_dir, 'check_summary.json'))

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Test image classification')
#     parser.add_argument('--savepath','-sp',default='/home/zxy/main/Projects/2020_ZhongDian/Evironment/code_check/tmp.pkl', help='savepath')#'D:\\workspace\\project\\tmp.pkl'
#     parser.add_argument('--savedir','-sd',default='/home/zxy/main/Projects/2020_ZhongDian/Evironment/code_check/tmp', help='savedir')#'D:\\workspace\\project\\tmp.pkl'
#     parser.add_argument('--cvepath','-cvep',default="/data2/zxy/Projects/2020_ZhongDian/Evironment/cvelist-master/valid_extract.pkl", help='cve lib path')#'D:\\workspace\\project\\tmp.pkl' /home/zxy/main/Projects/2020_ZhongDian/Evironment/code_check/all_csv/out
#     parser.add_argument('--method','-mtd',default="hard", help='potential problem search method')#'D:\\workspace\\project\\tmp.pkl' /home/zxy/main/Projects/2020_ZhongDian/Evironment/code_check/all_csv/out
#     parser.add_argument('--code_check_enable', '-cce',default=True, help='The directory to save logs.')
#     parser.add_argument('--code_input', '-ci',default='/data2/zxy/Projects/2020_ZhongDian/Evironment/code_check/autokeras/autokeras', help='The directory to check.')
#     parser.add_argument('--detection_result', '-dr',default='/data2/zxy/Projects/2020_ZhongDian/Evironment/code_check/tmp_out', help='The directory to save logs.')

#     args = parser.parse_args()
#     env_detection(args)
#     # if args.code_check_enable:
#     #     print("--------------Start Code Check--------------")
#     #     code_check(args.code_input,args.detection_result)
#     print("Detection Finished!\nThe environment detection result is saved in
#     {}".format(self.save_dir))

def read_centos_sys(centos_sys_path,info):

    f = open(centos_sys_path, 'r')
    lines=f.readlines()
    for line in lines:
        line=line.strip()
        if 'Distributor ID' in line:
            info['system']=line.split(':')[-1].strip('\t')
        elif 'Release:' in line:
            info['sys_version']=line.split(':')[-1].strip('\t')
        # elif 'LSB Version:' in line:
        #     info['processor']=line.split(':',1)[-1].strip('\t')

    return info

class ENVT(object):
    def __init__(self,
        json_path,
        save_dir='./env',
        cve_path='./valid_extract.pkl'):
        keypath1 = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        keypath2 = r"SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall"
        
        self.save_dir=os.path.abspath(save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.json_path=json_path
        self.save_path=os.path.join(self.save_dir,'tmp.pkl')
        self.cve_path=os.path.abspath(cve_path)

        print("--------------Start Extract System Information--------------")
        info=self.get_sys_info()
        # info is the system msg
        print("--------------Start Extract Lib Version--------------")
        if info=={}:
            os._exit(0)
        elif info['system']=='Windows':
            if os.path.exists(self.save_path):
                os.remove(self.save_path)
            val = self.win_get_software_list(keypath1, self.save_path)
            if val == "":
                val = self.win_get_software_list(keypath2, self.save_path)
        elif info['system']=='Linux':
            if 'ubuntu' in info['sys_version'].lower():
                val= self.ubuntu_get_software_list(self.save_path)
            elif 'debian' in info['sys_version'].lower():
                pass
            else:
                import subprocess
                command="lsb_release -a >{}"
                centos_sys_path=os.path.join(save_dir,'tmp_env_sys')
                run_command=command.format(centos_sys_path)
                out_path=os.path.join(os.path.dirname(save_dir),'log')
                out_file = open(out_path, 'w')
                out_file.write('logs\n')
                subprocess.call(run_command, shell=True, stdout=out_file, stderr=out_file)

                info=read_centos_sys(centos_sys_path,info)
                val= self.centos_get_software_list(self.save_path)
        else:
            print('Not support System')
            os._exit(0)
        # val is the sys software version
        sys_msg={}
        sys_msg['software_lib_list']=val
        sys_msg['env_info']=info
        sys_msg['description']="The system architecture is {} with total of {} libs.\nThe detailed infomation of environment is shown in `env_info`.\
            And the libs and softwares in the system is shown in `Software_lib_list`.".format(info['system'],len(val))
        # save system msg
        self.sys_msg_path=os.path.join(self.save_dir,'system_message.pkl')
        with open(self.sys_msg_path,'wb') as f:
            pickle.dump(sys_msg,f)
        self.val=val
        
    def detection(self,method):
        start=time.time()
        print("--------------Start Detect System Vulnerable--------------")
        if method!='all':
            potential_problem=judge_vulnerable_new(self.val,self.cve_path,method=method)
            import pickle
            with open(os.path.join(self.save_dir,'potential_problem_{}_new.pkl'.format(method)),'wb') as f:
                pickle.dump(potential_problem,f)
                print(time.time()-start)
            report_dict={}
            report_dict['Risk']={}
            potential_lib_list=list(potential_problem.keys())
            report_dict['Risk']['Libs']=potential_lib_list
            report_dict['Risk']['CVE Reports']=potential_problem
            report_dict['Risk']['Test Method']=method
            report_dict['Risk']['description']="This dict shows the risk of the libs in current environment. We have list the lib name and the correspoding CVE reports for reference."

        else:
            hard_problem=judge_vulnerable_new(self.val,self.cve_path,method='hard')
            potential_problem=judge_vulnerable_new(self.val,self.cve_path,method='soft')
            import pickle
            with open(os.path.join(self.save_dir,'potential_problem_{}_new.pkl'.format(method)),'wb') as f:
                pickle.dump(potential_problem,f)
                print(time.time()-start)
            
            report_dict={}
            report_dict['Confirmable Risk']={}
            report_dict['Potential Risk']={}
            risk_lib_list=list(hard_problem.keys())
            report_dict['Confirmable Risk']['Libs']=risk_lib_list
            report_dict['Confirmable Risk']['CVE Reports']=hard_problem
            potential_lib_list=list(potential_problem.keys())
            report_dict['Potential Risk']['Libs']=potential_lib_list
            report_dict['Potential Risk']['CVE Reports']=potential_problem
            report_dict['Confirmable Risk']['description']="The Libs in this dict shows the list of the risk lib that can be comfirmed by the CVE reports. The CVE Reports dict shows the corresponding CVE report of each libs."
            report_dict['Potential Risk']['description']="This dict shows the potential risk of the libs in current environment. We have list the lib name and the correspoding CVE reports for reference."
        self.detection_json_path=os.path.join(self.save_dir,'detection.json')
        with open(self.detection_json_path,"w") as f:
            f.write(json.dumps(report_dict,ensure_ascii=False,indent=4,separators=(',',':')))
        
        if os.path.exists(self.json_path):
            with open(self.json_path,'r') as f:
                json_result = json.load(f)
        else:
            json_result={}
        json_result['env_test']={}
        json_result['env_test']['result_dir']=os.path.join('keti2',(os.path.abspath(self.save_dir).split('keti2/')[-1]))
        json_result['env_test']['detection_result']=os.path.join('keti2',(self.detection_json_path.split('keti2/')[-1]))
        json_result['env_test']['sys_msg']=os.path.join('keti2',(self.sys_msg_path.split('keti2/')[-1]))
        
        with open(self.json_path, 'w') as fw:
            json.dump(json_result,fw)
        print("--------------Finish!--------------")
        return json_result

    
    def get_sys_info(self):
        info={}
        all_info=platform.uname()
        info['system']=all_info.system
        info['sys_version']=all_info.version
        info['processor']=all_info.processor
        return info
    
    def win_get_software_list(self,keypath, pkl_path):
        
        key = openReg(keypath)
        i = 0
        result = ""
        software_dict=get_previous_dict(pkl_path)
        try:
            while 1:
                name = winreg.EnumKey(key, i)
                path = keypath + "\\" + name
                subkey = openReg(path)
                try:
                    value = queryVal(subkey, 'DisplayName')
                    version = queryVal(subkey, 'DisplayVersion')
                    closeReg(subkey)
                    software_dict=add_software_dict(value,version,software_dict)
                    # print('{}\n'.format(value))
                except Exception as e:
                    print(e)
                finally:
                    closeReg(subkey)
                i += 1
        except Exception as e:
            print(e)
        closeReg(key)
        with open(pkl_path, 'wb') as f:
            pickle.dump(software_dict, f)
        return software_dict

    def ubuntu_get_software_list(self,pkl_path):
        tmp_save_path= os.path.join(os.path.dirname(pkl_path),'tmp')
        # './safety-db/tmp'

        software_dict=get_previous_dict(pkl_path)
        out_path=os.path.join(os.path.dirname(pkl_path),'log')
        import subprocess
        command="apt list --installed >{}"
        run_command=command.format(tmp_save_path)
        out_file = open(out_path, 'w')
        out_file.write('logs\n')
        subprocess.call(run_command, shell=True, stdout=out_file, stderr=out_file)
        
        software_dict=read_output(software_dict,tmp_save_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(software_dict, f)
        return software_dict

    def centos_get_software_list(self,pkl_path):
        tmp_save_path= os.path.join(os.path.dirname(pkl_path),'tmp')
        # './safety-db/tmp'

        software_dict=get_previous_dict(pkl_path)
        out_path=os.path.join(os.path.dirname(pkl_path),'log')
        import subprocess
        command="rpm -qa | sort >{}"
        run_command=command.format(tmp_save_path)
        out_file = open(out_path, 'w')
        out_file.write('logs\n')
        subprocess.call(run_command, shell=True, stdout=out_file, stderr=out_file)
        
        software_dict=read_centos_output(software_dict,tmp_save_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(software_dict, f)
        return software_dict