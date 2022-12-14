param_bounds={'x1':(-1,5),
               'x2':(0,4)}
def y_funtion(x1,x2):
    return -x1 **2-(x2-2) **2 +10

# pip install BayesianOptimization
from bayes_opt import BayesianOptimization

optimizer=BayesianOptimization(f=y_funtion, #찾고자 하는 함수를 넣는다
                               pbounds=param_bounds, #그 함수에 들어있는 파라미터를 딕셔너리로 넣는다
                               random_state=1234)

optimizer.maximize(init_points=2, #초기화 포인트
                   n_iter=20)
#결국 22번 돈다는이야기임

print(optimizer.max)
