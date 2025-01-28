from concurrent.futures import ProcessPoolExecutor
from AOB.AOB import Benchmark
from HCC.RDDSM import Decomposition
import numpy as np
from AOB.utils import plot_evaluation_curve, plot_evaluation_curve_best_so_far, evaluation_record, combine, remove_overlapping_groups
import time
import math
from HCC.NDAs.MMES.mmes import MMES
from HCC.OPT.CMAES.cmaes import CMAES

def optimization_task(fun_name, fun_id, best_individual, MaxFEs, grouping_result, info, overlapping_elements, overlap_groups):
    time_start = time.time()
    fun = bench.get_function(fun_name, fun_id)
    sumFEs = 0

    overlapping_ratio = len(overlap_groups) / info['dimension']

    if overlapping_ratio == 0:
        GloFEs = 0
    else:
        GloFEs = int(0.2*MaxFEs + (4/5)*overlapping_ratio*MaxFEs)

    if GloFEs != 0:
        '''global'''
        problem_ = {'fitness_function': fun,  # fitness function
        'ndim_problem': info['dimension'],  # dimension
        'lower_boundary': info['lower'] * np.ones((info['dimension'],)),  # lower search boundary
        'upper_boundary': info["upper"]* np.ones((info['dimension'],))}

        options_ = {'max_function_evaluations': GloFEs,  # to set optimizer options
            'mean': (best_individual,) ,
            'sigma': 0.5,
            'is_restart': False,
            'verbose': 1000} 
        optimizer = MMES(problem_, options_)
        results_ = optimizer.optimize()

        best_individual = results_['best_so_far_x'].copy()
        sumFEs += results_['n_function_evaluations']
        
    while sumFEs < MaxFEs:
        sub_num = len(grouping_result)
        subFEs = math.ceil((MaxFEs - sumFEs) / sub_num)
        fitness_delta_list = []
        for i, dims in enumerate(grouping_result):
            orignal_best_individual = best_individual.copy()
            orignal_best_fitness = fun(best_individual)
            OBJFUNC = lambda X_batch: fun(combine(X_batch, best_individual, dims))
            problem_CC = {'fitness_function': OBJFUNC,  # fitness function
            'ndim_problem': len(dims),  # dimension
            'lower_boundary': info['lower'] * np.ones((len(dims),)),  # lower search boundary
            'upper_boundary': info["upper"]* np.ones((len(dims),))}
            options_CC = {'max_function_evaluations': subFEs,  # to set optimizer options
                'mean': (best_individual[dims],) ,
                'sigma': 0.5,
                'is_restart': False,
                'verbose': 1000,
                'early_stopping_evaluations':1000}
            optimizer_CC = CMAES(problem_CC, options_CC)
            results_CC = optimizer_CC.optimize()
            best_individual[dims] = results_CC['best_so_far_x'].copy()
            sumFEs += results_CC['n_function_evaluations']
            fitness_delta_list.append(orignal_best_fitness-results_CC['best_so_far_y'])
            if i > 0:
                weight = fitness_delta_list[i-1] / (fitness_delta_list[i] + fitness_delta_list[i-1])
                best_individual[overlapping_elements[i-1]] = weight * best_individual[overlapping_elements[i-1]] + (1 - weight) * orignal_best_individual[overlapping_elements[i-1]]
    time_end = time.time()
    return fun.fitness_record, (time_end - time_start)

def parallel_optimization(fun_name, fun_id, best_individual, MaxFEs, cycle_num, grouping_result, info, output_data, overlapping_elements, overlap_groups):
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(cycle_num):
            futures.append(executor.submit(optimization_task, fun_name, fun_id, best_individual, MaxFEs, grouping_result, info, overlapping_elements, overlap_groups))
        
        Algorithm = f'{fun_name}_{fun_id}'
        average_time = 0
        for future in futures:
            result = future.result()
            output_data[Algorithm].append(result[0])
            average_time += result[1]
        
        return average_time / cycle_num

fun_name_list = ['elliptic','schwefel','rastrigin','ackley']

output_data_schwefel = {'schwefel_1': [], 'schwefel_1_time':[], 
                        'schwefel_2': [], 'schwefel_2_time':[], 
                        'schwefel_3': [], 'schwefel_3_time':[], 
                        'schwefel_4': [], 'schwefel_4_time':[], 
                        'schwefel_5': [], 'schwefel_5_time':[], 
                        'schwefel_6': [], 'schwefel_6_time':[]}
output_data_elliptic = {'elliptic_1': [], 'elliptic_1_time':[],
                        'elliptic_2': [], 'elliptic_2_time':[],
                        'elliptic_3': [], 'elliptic_3_time':[],
                        'elliptic_4': [], 'elliptic_4_time':[],
                        'elliptic_5': [], 'elliptic_5_time':[],
                        'elliptic_6': [], 'elliptic_6_time':[]}
output_data_rastrigin = {'rastrigin_1': [], 'rastrigin_1_time':[],
                        'rastrigin_2': [], 'rastrigin_2_time':[],
                        'rastrigin_3': [], 'rastrigin_3_time':[],
                        'rastrigin_4': [], 'rastrigin_4_time':[],
                        'rastrigin_5': [], 'rastrigin_5_time':[],
                        'rastrigin_6': [], 'rastrigin_6_time':[]}
output_data_ackley = {'ackley_1': [], 'ackley_1_time':[],
                        'ackley_2': [], 'ackley_2_time':[],
                        'ackley_3': [], 'ackley_3_time':[],
                        'ackley_4': [], 'ackley_4_time':[],
                        'ackley_5': [], 'ackley_5_time':[],
                        'ackley_6': [], 'ackley_6_time':[]}

output_data_map = {
    'schwefel': output_data_schwefel,
    'elliptic': output_data_elliptic,
    'rastrigin': output_data_rastrigin,
    'ackley': output_data_ackley
}

MaxFEs = 2E3
cycle_num = 5

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
for fun_name in fun_name_list:
    output_path = f'HCC_SRC/result/{timestamp}/{fun_name}/'
    bench = Benchmark(output_path)
    original_fitness_list = []
    output_data = output_data_map[fun_name]
    for fun_id in range(6):
        fun_id += 1

        # decomposition initialization
        file_path = f'HCC_SRC/AOB/AOBG/datafile/F{fun_id}-design.txt'
        design_matrix = np.loadtxt(file_path, delimiter=',')
        decomposition = Decomposition(design_matrix)
        grouping_result = decomposition.decomposition()
        _,overlap_groups,overlapping_elements = remove_overlapping_groups(grouping_result)
        info = bench.get_info(fun_name, fun_id)
        fun = bench.get_function(fun_name, fun_id)

        best_individual = np.zeros(info['dimension'])
        best_fitness = fun(best_individual)[0].copy()
        original_fitness_list.append(best_fitness)

        average_time = parallel_optimization(fun_name, fun_id, best_individual, MaxFEs, cycle_num, grouping_result, info, output_data, overlapping_elements, overlap_groups)
        print(f'{fun_name}_{fun_id} average time: {average_time}')

        output_data[f'{fun_name}_{fun_id}_time'].append(average_time)

    evaluation_record(output_data, output_path, record_FEs_list=[1.2E5, 2E5, 1E6, 2E6, 3E6])
    plot_evaluation_curve(output_data, output_path, font_size = 12, log_scale=True)
    plot_evaluation_curve_best_so_far(output_data, output_path, font_size = 12, log_scale=True, show_variance=True)
