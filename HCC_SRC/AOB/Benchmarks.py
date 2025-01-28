import numpy as np
import sys
import os

class Benchmarks:
    def __init__(self, output_path):
        self.data_dir = "HCC_SRC/AOB/AOBG/datafile"  # 数据文件夹
        self.dimension = 1000  # 维度


        # 基本量的设置, 不是准确的值，准确的值会在function中设置
        self.ID = None
        self.s_size = 20
        self.overlap = None
        self.minX = None
        self.maxX = None
        self.Ovector = None
        self.OvectorVec = None
        self.Pvector = None
        self.anotherz = np.zeros(self.dimension)
        self.anotherz1 = None
        self.best = 0
        self.cache_Rotation = {}

        self.maxevals = 3000000  # 最大评估次数
        self.numevals = 0  # 当前评估次数, 用于记录当前评估次数, cpp中设置的值是2*self.maxevals

        self.output_path = output_path
        if output_path is not None and not os.path.exists(output_path):
            os.makedirs(output_path)
        # self.record_evels = np.array([120000, 600000, 3000000])

        # 评估器
        self.fitness_record = []

    # 读取Ovector
    def readOvector(self):
        d = np.zeros(self.dimension)
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"
        
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = float(value)
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return d
    
    # 读取OvectorVec，根据子空间的大小分割，得到一个向量数组
    def readOvectorVec(self):
        d = [np.zeros(self.s[i]) for i in range(self.s_size)]
        file_path = f"{self.data_dir}/F{self.ID}-xopt.txt"

        try:
            with open(file_path, 'r') as file:
                c = 0  # index over 1 to dim
                i = -1  # index over 1 to s_size
                up = 0  # current upper bound for one group

                for line in file:
                    if c == up:  # out (start) of one group
                        i += 1
                        d[i] = np.zeros(self.s[i])
                        up += self.s[i]

                    values = line.strip().split(',')
                    for value in values:
                        d[i][c - (up - self.s[i])] = float(value)
                        c += 1
        except FileNotFoundError:
            print(f"Cannot open the OvectorVec datafiles '{file_path}'")

        return d
    
    # 读取PermVector
    def readPermVector(self):
        d = np.zeros(self.dimension, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-p.txt"
        
        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dimension:
                            d[c] = int(float(value)) - 1
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return d
    
    # 读取R，即为各个子空间的向量
    def readR(self, sub_dim):
        m = np.zeros((sub_dim, sub_dim))
        file_path = f"{self.data_dir}/F{self.ID}-R{sub_dim}.txt"

        try:
            with open(file_path, 'r') as file:
                i = 0
                for line in file:
                    values = line.strip().split(',')
                    for j, value in enumerate(values):
                        m[i, j] = float(value)
                    i += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return m

    # 读取S，即为各个子问题的维度
    def readS(self, num):
        self.s = np.zeros(num, dtype=int)
        file_path = f"{self.data_dir}/F{self.ID}-s.txt"

        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    self.s[c] = int(float(line.strip()))
                    c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return self.s

    # 读取W
    def readW(self, num):
        self.w = np.zeros(num)
        file_path = f"{self.data_dir}/F{self.ID}-w.txt"

        try:
            with open(file_path, 'r') as file:
                c = 0
                for line in file:
                    self.w[c] = float(line.strip())
                    c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_path}'")
        
        return self.w

    # 向量乘矩阵
    def multiply(self, vector, matrix):
        return np.matmul(matrix, vector.T).T

    # 旋转向量
    def rotateVector(self, i, c): 
        sub_dim = self.s[i]
        indices = self.Pvector[c:c + sub_dim]
        z = self.anotherz[:, indices]
        
        rotate_matrix = self.cache_Rotation[sub_dim]
        self.anotherz1 = self.multiply(z, rotate_matrix)

        return self.anotherz1
    
    def rotateVectorConform(self, i, c):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        indices = self.Pvector[start_index:end_index]
        z = self.anotherz[:, indices]
        
        rotate_matrix = self.cache_Rotation[sub_dim]
        self.anotherz1 = self.multiply(z, rotate_matrix)
    
        return self.anotherz1

    def rotateVectorConflict(self, i, c, x):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap

        indices = self.Pvector[start_index:end_index]
        z = x[indices] - self.OvectorVec[i]
        z = z.astype(float)
        
        rotate_matrix = self.cache_Rotation[sub_dim]
        self.anotherz1 = self.multiply(z, rotate_matrix)

        return self.anotherz1

    # basic function
    def sphere(self, x):
        return np.sum(x ** 2, axis=-1)

    def elliptic(self, x):
        nx = x.shape[-1]
        i = np.arange(nx)
        return np.sum(10 ** (6 * i / (nx - 1)) * (x ** 2), axis=-1)

    def rastrigin(self, x):
        return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10, axis=-1)

    def ackley(self, x):
        nx = x.shape[-1]
        sum1 = -0.2 * np.sqrt(np.sum(x ** 2, axis=-1) / nx)
        sum2 = np.sum(np.cos(2 * np.pi * x), axis=-1) / nx
        return -20 * np.exp(sum1) - np.exp(sum2) + 20 + np.e

    def schwefel(self, x):
        s1 = np.cumsum(x, axis=-1)
        return np.sum(s1 ** 2, axis=-1)

    def rosenbrock(self, x):
        x0 = x[:, :-1]
        x1 = x[:, 1:]
        t = x0 ** 2 - x1
        return np.sum(100.0 * t ** 2 + (x0 - 1.0) ** 2, axis=-1)
    
    def transform_osz(self, z):
        sign_z = np.sign(z)
        hat_z = np.where(z == 0, 0, np.log(np.abs(z)))
        c1_z = np.where(z > 0, 10, 5.5)
        c2_z = np.where(z > 0, 7.9, 3.1)
        sin_term = np.sin(c1_z * hat_z) + np.sin(c2_z * hat_z)
        return sign_z * np.exp(hat_z + 0.049 * sin_term)

    def transform_asy(self, z, beta=0.2):
        indices = np.arange(z.shape[-1])[None, :].repeat(z.shape[0], axis=0)
        positive_mask = z > 0
        z[positive_mask] = z[positive_mask] ** (1 + beta * indices[positive_mask] / (z.shape[-1] - 1) * np.sqrt(z[positive_mask]))
        return z
    
    def Lambda(self, z, alpha=10):
        dim = z.shape[-1]
        exponents = 0.5 * np.arange(dim) / (dim - 1)
        return z * (alpha ** exponents)
    
    def update(self, newfitness):
        if self.numevals > self.maxevals:
            if self.numevals >= 2 * self.maxevals:
                print("Error: nextRun was not run before compute")
                sys.exit(1)
            elif self.numevals > self.maxevals * 1.1:
                print("Error: many evaluations greater than maximum.")
                sys.exit(1)
            print("Warning: evaluations greater than maximum, will be ignored.")
            return

        if self.numevals == 0 or newfitness < self.best_fitness:
            self.best_fitness = newfitness
            if not self.output:
                self.output = f"results_f{self.ID}.csv"

        self.numevals += 1

        if self.numevals in self.record_evels:
            self.save_evals()
        
    def save_evals(self):
        with open(self.output, 'a') as f_output:
            f_output.write(f"{self.numevals}, {self.ID}, {self.best_fitness:.6e}\n")

    
    






        







    




    






