import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import random
from test import tests

random.seed(42)


class p_model():
    def __init__(self, k, mu ,lamb):
        self.k = k
        self.mu = mu
        self.lamb = lamb
        self.dapature_time = 60 / self.mu
        self.trashhold_arrival =  1 - np.exp(-self.lamb / 60)
        self.trashhold_depature = 1 - np.exp(-self.mu * self.k / 60)

    def simulate(self, time):

        requests_arrival_mnt = 0
        requests_depature_mnt = 0
        is_busy = False
        time_in_queue = []
        amount_of_people = 0
        total_time = 0
        total_requests = 0

        alpha = (np.sqrt(1 + 4 * self.lamb / (self.mu * self.k)) - 1) / 2
        P = {}
        N = set()

        for time in range(1, time + 1):
            r_a, r_d = np.random.exponential(size= 2).tolist()

            if r_a < self.trashhold_arrival:
                requests_arrival_mnt += 1
                requests_depature_mnt = requests_arrival_mnt

            if time % self.dapature_time == 0:
                is_busy = False
            else:
                is_busy = True


            if not(is_busy) and r_d < self.trashhold_depature:
                requests_depature_mnt = requests_arrival_mnt - self.k
                is_busy = True

            if requests_depature_mnt < 0:
                requests_depature_mnt = 0

            
            if r_d < self.trashhold_depature and time_in_queue:

                crop = min(len(time_in_queue), self.k)
                total_time += sum(list(map(lambda x: x / 60 + 1 / self.mu, time_in_queue[:crop + 1])))
                time_in_queue = time_in_queue[crop:]

            if time_in_queue:
                time_in_queue = list(map(lambda x: x + 1, time_in_queue)) 

            if r_a < self.trashhold_arrival:
                time_in_queue.append(0)
                amount_of_people += 1

            if(requests_arrival_mnt not in N):
                P['P' + str(requests_arrival_mnt)] = alpha ** requests_arrival_mnt * (1 - alpha)
                N.add(requests_arrival_mnt)

            total_requests += requests_arrival_mnt


        l_q = 0

        for i, p in  enumerate(P.values()):
            l_q += i * p

        w_q_formula = l_q / self.lamb
        w_s_formula = w_q_formula + 1 / self.mu

        w_q = total_time / amount_of_people
        w_s = w_q + 1 / self.mu
        l_q_formula = alpha / ( 1 - alpha)



        assert l_q  -  l_q_formula < 0.0001,  'l_q not equal l_q by formula'
        assert  1 - sum(P.values()) < 0.1, f'sum of P_i  {sum(P.values())} must be equal to 1 {self.k}, {self.lamb}, {self.mu}'
        assert abs(w_s  -  w_s_formula) < 0.1,  f'{w_q} not equal {w_s_formula} by formula'


        with open('test_log.txt', 'a') as f:
            f.write(f'########\n[{self.k}, {self.lamb}, {self.mu}],\n w_s = {w_s} w_s_formula = {w_s_formula}\nl_q = {l_q} l_q_formula = {l_q_formula}\nsum of P = {sum(P.values())}\n########\n\n')

        


                

def test():

    for test in tqdm(tests):

        k, lamb, mu = test
        
        model = p_model(k, mu, lamb)
        model.simulate(10 ** 5)

    print('ALL TEST WAS PASSED!')

    

