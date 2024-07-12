# Instantiation of the random number generator class
from random import SystemRandom
rand = SystemRandom()

# Type alias for representing n-space vector data-points
from typing import Iterable
type VectorN = Iterable[float]

# Constant that stores the dimensionality of the data-space
DIMENSIONALITY = 2

SLOPE = rand.random()
INTERCEPT = rand.uniform(0, 100)


original_line: list[VectorN] = [
    (x, x * SLOPE + INTERCEPT)
    for x in range(100)
]

observations: list[VectorN] = [
    (x, x * SLOPE + INTERCEPT + rand.gauss(0, 10))
    for x in range(100)
]

def Linear_Regression(
    data_set: list[VectorN]
):
    from statistics import covariance, variance, mean
    
    def _Sum_Of_Square_Residuals(
        data_set: list[VectorN],
        slope: float,
        intercept: float
    ):
        return sum(
            (y - (intercept + slope * x))**2
            for x, y in data_set
        )
    xs, ys = zip(*data_set)
    
    slope = covariance(xs, ys) / variance(xs)
    intercept = mean(ys) - slope * mean(xs)
    
    return (slope, intercept)


slope_estimate, intercept_estimate = Linear_Regression(observations)

line = [
    (x, x * slope_estimate + intercept_estimate)
    for x in range(100)
]

# Plotting the solution
from matplotlib import pyplot as p

p.figure(figsize=(5, 5))
p.axis('equal')

p.plot(
    *zip(*observations),
    'ro'
)
p.plot(
    *zip(*line),
    color='blue'
)
p.plot(
    *zip(*original_line),
    color='green'
)

xs, ys = zip(*observations)
from statistics import variance
print(variance([1, 2]))


p.close()
