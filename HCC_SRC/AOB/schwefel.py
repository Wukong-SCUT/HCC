from AOB.Benchmarks import Benchmarks
import numpy as np
import yaml

class schwefel(Benchmarks):
    def __init__(self, ID, output_path):
        super().__init__(output_path)
        self.ID = ID
        info_file_path = f'HCC_SRC/AOB/AOBG/datafile/F{ID}-info.txt'
        
        # Initialize data for Schwefel function
        with open(info_file_path, "r") as file:
            data = yaml.safe_load(file)
        
        self.s_size = data['sub_num']
        self.dimension = data['dimension']  # Because of overlapping
        self.overlap = data['overlap_degree']
        
        # Read vectors and matrices (NumPy arrays instead of tensors)
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()

        
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        
        self.minX = data['lower_bound']
        self.maxX = data['upper_bound']
        
        # Initialize additional variables
        self.anotherz = np.zeros(self.dimension)  # NumPy array instead of PyTorch tensor
        self.cache_Rotation = {i: self.readR(i) for i in data['subgroups_type']}

    def __call__(self, x):
        return self.compute(x)

    def info(self):
        return {
            'best': 0.0,
            'dimension': self.dimension,
            'lower': self.minX,
            'threshold': 0,
            'upper': self.maxX
        }

    def compute(self, x):

        # Make sure x is a 2D array if it is 1D
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        
        result = np.zeros(x.shape[0])

        c = 0
        self.anotherz = x - self.Ovector  # Element-wise subtraction

        for i in range(self.s_size):
            anotherz1 = self.rotateVectorConform(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # Update c

        self.fitness_record.extend(result.tolist())

        return result

