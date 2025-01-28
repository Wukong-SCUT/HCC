from AOB.schwefel import schwefel
from AOB.elliptic import elliptic
from AOB.rastrigin import rastrigin
from AOB.ackley import ackley
import os

class Benchmark():
    def __init__(self, output_path):
        self.output_path = output_path
        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
    def get_function(self, func_name, func_id):
        if func_name == 'elliptic':
            return elliptic(func_id, self.output_path)
        elif func_name == 'rastrigin':
            return rastrigin(func_id, self.output_path)
        elif func_name == 'ackley':
            return ackley(func_id, self.output_path)
        elif func_name == 'schwefel':
            return schwefel(func_id, self.output_path)
        else:
            raise ValueError("Function name is wrong.")

    def get_info(self, func_name, func_id):
        if func_name == 'elliptic':
            fun_ = elliptic(func_id, self.output_path)
            return fun_.info()
        elif func_name == 'rastrigin':
            fun_ = rastrigin(func_id, self.output_path)
            return fun_.info()
        elif func_name == 'ackley':
            fun_ = ackley(func_id, self.output_path)
            return fun_.info()
        elif func_name == 'schwefel':
            fun_ = schwefel(func_id, self.output_path)
            return fun_.info()
        else:
            raise ValueError("Function name is wrong.")
    



