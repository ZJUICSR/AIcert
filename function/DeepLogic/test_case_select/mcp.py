import numpy as np

def get_boundary_priority(output_probability):
    output = np.array(output_probability)
    sorted_index = output.argsort()[::-1] # sort in descending order
    max_value = output[sorted_index[0]]
    second_max_value = output[sorted_index[1]]
    bound_priority =  1.0* second_max_value / max_value
    return sorted_index[0], sorted_index[1], bound_priority


def select_from_firstsec_dic(selectsize, dicratio, dicindex, num_classes):
    selected_lst=[]
    tmpsize=selectsize
    #tmpsize保存的是采样大小，全程都不会变化
    
    noempty=no_empty_number(dicratio)
    #print(selectsize)
    #print(noempty)
    #待选择的数目大于非空的类别数(满载90类)，每一个都选一个
    while selectsize>=noempty:
        for i in range(num_classes*num_classes):
            if len(dicratio[i])!=0:#非空就选一个最大的出来
                tmp=max(dicratio[i])
                j = dicratio[i].index(tmp)
                #if tmp>=0.1:
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize=tmpsize-len(selected_lst)
        noempty=no_empty_number(dicratio)
        #print(selectsize)
    #selectsize<noempty
    #no_empty_number(dicratio)
    #print(selectsize)
    
    #剩下少量样本没有采样，比如还存在30类别非空，但是只要采样10个，此时我们取30个最大值中的前10大
    while len(selected_lst)!= tmpsize:
        max_tmp=[0 for i in range(selectsize)]#剩下多少就申请多少
        max_index_tmp=[0 for i in range(selectsize)]
        for i in range(num_classes*num_classes):
            if len(dicratio[i])!=0:
                tmp_max=max(dicratio[i])
                if tmp_max>min(max_tmp):
                    
                    index=max_tmp.index(min(max_tmp))
                    max_tmp[index]=tmp_max
                    #selected_lst.append()
                    #if tmp_max>=0.1:
                    max_index_tmp[index]=dicindex[i][dicratio[i].index(tmp_max)]#吧样本序列号存在此列表中
        if len(max_index_tmp)==0 and len(selected_lst)!= tmpsize:
            print('wrong!!!!!!')  
            break
        selected_lst=selected_lst+ max_index_tmp
        #print(len(selected_lst))
    #print(selected_lst)
    assert len(selected_lst)== tmpsize
    return selected_lst


#配对表情非空的数目。比如第一是3，第二是5，此时里面没有任何实例存在那么就是0
def no_empty_number(dicratio):
    no_empty=0
    for i in range(len(dicratio)):
        if len(dicratio[i])!=0:
            no_empty+=1
    return no_empty

