import numpy as np
import matplotlib.pyplot as plt


class GlobalRandomSearch:
    def __init__(self, max_it,points):
        self.max_it  = max_it
        self.points = points
        self.qtd_points = points.shape[0]
        self.x_opt = np.random.permutation(self.qtd_points-1)+1
        self.x_opt = np.concatenate(([0],self.x_opt))
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]
        #atributos do plot
        self.fig = plt.figure(1)
        self.ax = self.fig.subplots()
        
        self.ax.scatter(points[:,0],points[:,1])
        self.lines = []
        self.update_plot()
    def clear_lines(self):
        for line in self.lines:
            line.remove()
        self.lines = []
    
    def update_plot(self):
        self.ax.set_title(f"Global Random Search {self.f_opt:.4f}")
        for i in range(self.qtd_points):
            p1 = self.points[self.x_opt[i]]
            p2 = self.points[self.x_opt[(i+1)%self.qtd_points]]
            
            if i == 0:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='r')
            elif i == self.qtd_points-1:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='g')
            else:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
            self.lines.append(line[0])
    def f(self, x):
        d = 0
        for i in range(self.qtd_points):
            p1 = self.points[x[i]]
            p2 = self.points[x[(i+1)%self.qtd_points]]
            d += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return d
    
    def perturb(self):
        x_cand = np.random.permutation(self.qtd_points-1)+1
        x_cand = np.concatenate(([0],x_cand))
        return x_cand
    
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            self.historico.append(self.f_opt)
            if f_cand < self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                plt.pause(.5)
                self.clear_lines()
                self.update_plot()
            it+=1
        plt.figure(2)
        plt.plot(self.historico)
        plt.grid()
        plt.title("GRS histórico")
       
        
        
class LocalRandomSearch:
    def __init__(self, max_it,points, sigma):
        self.sigma = sigma
        self.max_it  = max_it
        self.points = points
        self.qtd_points = points.shape[0]
        self.x_opt = np.random.permutation(self.qtd_points-1)+1
        self.x_opt = np.concatenate(([0],self.x_opt))
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]
        #atributos do plot
        self.fig = plt.figure(3)
        self.ax = self.fig.subplots()
        
        self.ax.scatter(points[:,0],points[:,1])
        self.lines = []
        self.update_plot()
    def clear_lines(self):
        for line in self.lines:
            line.remove()
        self.lines = []
    
    def update_plot(self):
        self.ax.set_title(f"Local Random Search {self.f_opt:.4f}")
        for i in range(self.qtd_points):
            p1 = self.points[self.x_opt[i]]
            p2 = self.points[self.x_opt[(i+1)%self.qtd_points]]
            
            if i == 0:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='r')
            elif i == self.qtd_points-1:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='g')
            else:
                line = self.ax.plot([p1[0],p2[0]],[p1[1],p2[1]],c='k')
            self.lines.append(line[0])
    def f(self, x):
        d = 0
        for i in range(self.qtd_points):
            p1 = self.points[x[i]]
            p2 = self.points[x[(i+1)%self.qtd_points]]
            d += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        return d
    
    def perturb(self):
        x_cand = np.copy(self.x_opt)
        indexes1 = (np.random.permutation(self.qtd_points-1)+1)[:self.sigma]        
        indexes2 = np.random.permutation(indexes1)
        x_cand[indexes1] = x_cand[indexes2]
        
        
        
        return x_cand
    
    def search(self):
        it = 0
        while it < self.max_it:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            self.historico.append(self.f_opt)
            if f_cand < self.f_opt:
                self.x_opt = x_cand
                self.f_opt = f_cand
                plt.pause(.5)
                self.clear_lines()
                self.update_plot()
            it+=1
        plt.figure(4)
        plt.plot(self.historico)
        plt.grid()
        plt.title("LRS histórico")
        plt.show()



bp=1