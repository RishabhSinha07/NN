class DescentOptimizer:
    def __init__(self):
        self.x = 0 # initial guess for the variable x
        self.alpha = 0.1  # learning rate

    def f(self, x):
        return x**2 - 3*x + 4

    def df(self, x):
        return 2*x - 3

    def optimize(self, iterations=100):
        for _ in range(iterations):
            print("Iteration:", _, "x:", self.x, "f(x):", self.f(self.x))
            self.x = self.x - self.alpha * self.slope(self.f, self.x)

    def slope(self, f, x, h=1e-5):
        return (f(x + h) - f(x - h)) / (2 * h)
    
if __name__ == "__main__":
    optimizer = DescentOptimizer()
    optimizer.optimize()
    print("Optimized x:", optimizer.x)