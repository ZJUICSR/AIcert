from .ConcolicShow import *
import shutil
import os
import os.path as osp
CURR = osp.dirname(osp.abspath(__file__))

def run_concolic(data_name, model_name, norm, times, out_path, logging=None):
    # res = ConcolicShow.show_results('a','a', model_name=model_name, data_name=data_name, norm=norm, basepath=CURR, out_path=out_path,Times=3)
    logging.info("Start generation")
    res = ConcolicShow.show_results('a','a', model_name=model_name, data_name=data_name, norm=norm, basepath=out_path.rsplit("/",2)[0]+"/cache", out_path=out_path,Times=times, logging=logging)
    logging.info("End generation")
    return res
    # return ConcolicShow.show_results('a','a', model_name=model_name, data_name=data_name, norm=norm, basepath=CURR, out_path=CURR+'/show_path',Times=3)


def run(params):
	
	data_name = params["concolic_dataset"].lower()
	model_name = params["concolic_model"].lower()
	norm_type = params["norm"]
	base_path = params["basepath"]
	try_times = params["Times"]
	out_path = params["outpath"]


	## if need to perform the concolic algorithm: run next line: maybe need to create a new thread for it in the platform to flash the results in dynamic
	#ConcolicShow.Dynamic_run(model='a',data='a',model_name='lenet',data_name='mnist',norm='linf',basepath=pathOfDemo,times=2)

	## the next function return: a number for all samples generated and a string for a new img to be show for example



	result = ConcolicShow.show_results('a','a', model_name=model_name, data_name=data_name, norm=norm_type, basepath=base_path, out_path=out_path,Times=try_times)

	return result


if __name__ == '__main__':

	params={
		"concolic_dataset":"mnist",
		"concolic_model":'lenet',
		"norm":'linf',
		"basepath":CURR,
		"outpath":CURR+'/show_path',
		"Times":3
	}
	print(run(params))