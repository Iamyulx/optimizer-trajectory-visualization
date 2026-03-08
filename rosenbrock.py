from os import GRND_RANDOM
import torch

def rosenbrock(x):
  a = 1
  b = 100
  return (a - x[0])**2 + b*(x[1] - x[0]**2)**2
