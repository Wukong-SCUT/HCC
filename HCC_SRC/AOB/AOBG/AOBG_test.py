from AOBG_main import AOBG

fun_ID = 6
seed = 42
dimension = 1000
lower_bound = -100
upper_bound = 100
sub_num = 20
subgroups = [50,50,25,25,100,100,25,25,50,25,100,25,100,50,25,25,25,100,50,25]
overlap_list = [10]*(len(subgroups)-1)
file_path = 'datafile/'

AOBG_ = AOBG(fun_ID, seed, dimension, lower_bound, upper_bound, sub_num, subgroups, overlap_list, file_path)
AOBG_.main()