import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from scipy.interpolate import splrep, splev, interp1d, CubicSpline


class Graph:
    def __init__(self, name):
        self.graph = plt.figure()
        self.name = name
        plt.xlabel(r"Separation / $\mathrm{\AA}$")
        plt.ylabel(r"Tight Tolerance / $\frac{\hbar}{2}$")

    def plot(self, *args, **kwargs):
        self.plt = plt.plot(*args, **kwargs)

    def save(self):
        plt.savefig(self.name + ".png", dpi=680)

    def legend(self, *args, **kwargs):
        plt.legend(*args, **kwargs)


data = pd.read_csv("data.csv", sep=",")
separation = np.array(data["Separation"])
tight_tol = np.array(data["Tight Tol."])

separation_range = (np.min(separation), np.max(separation))


s = 3
x = np.linspace(*separation_range, 100)
tck = splrep(separation, tight_tol, s=s)
y = splev(x, tck, der=0)

cubic_spline = Graph("cubic_spline")
cubic_spline.plot(separation, tight_tol, "x", x, y)
cubic_spline.legend([r"Data", r"Cubic Spline $s=" + str(s) + "$"])
cubic_spline.save()

s = 4
tck = splrep(separation, tight_tol, s=s, k=4)
y = splev(x, tck, der=0)

quartic_spline = Graph("quartic_spline")
quartic_spline.plot(separation, tight_tol, "x", x, y)
quartic_spline.legend([r"Data", r"Quartic Spline $s=" + str(s) + "$"])
quartic_spline.save()

y = np.zeros_like(x)
poly_fit = np.polyfit(separation, tight_tol, deg=10)
for n, coef in enumerate(poly_fit[::-1]):
    y += x**n * coef

tenth_order_poly = Graph("tenth_order_poly")
tenth_order_poly.plot(separation, tight_tol, "x", x, y)
tenth_order_poly.legend([r"Data", r"10th order polynomial"])
tenth_order_poly.save()
