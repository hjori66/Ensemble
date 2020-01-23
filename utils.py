import numpy as np
import matplotlib.pyplot as plt

'''
If you want to reproduce of the results in the paper, the following parameters should be used
Data is generated by sampling from  'y = x^3 + error'

arg:
    num_data = 20
    x_ range = (-4, 4)
    std = 3.
'''

def generate_data(num_data=20, x_range=(-4,4), std=3.):
    x_data = [[np.random.uniform(*x_range)] for _ in range(num_data)]
    y_data = [[x[0]**3 +np.random.normal(0,std)] for x in x_data]

    return x_data ,y_data


def draw_graph(x,x_set,y_set,mean_predict,std):
    #_x = np.linspace(-6,6,100)
    y = x**3
    plt.plot(x,y,'b-', label = "Ground Truth")
    plt.plot(x_set, y_set,'ro', label = 'data points')
    plt.plot(x, mean_predict, label='Predicted mean', color='grey')
    plt.fill_between(x.reshape(-1), (mean_predict-3*std).reshape(100,), (mean_predict+3*std).reshape(100,),color='grey',alpha=0.3)
    plt.legend()
    plt.show()

