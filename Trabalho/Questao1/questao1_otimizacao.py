import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Callable, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def f1(x: np.ndarray) -> float:
    """
    Função 1: f(x1, x2) = x1² + x2²
    Domínio: x1, x2 ∈ [-100, 100]
    Objetivo: MINIMIZAR
    """
    return x[0]**2 + x[1]**2


def f2(x: np.ndarray) -> float:
    """
    Função 2: f(x1, x2) = e^(-(x1²+x2²)) + 2·e^(-((x1-1.7)²+(x2-1.7)²))
    Domínio: x1 ∈ [-2, 4], x2 ∈ [-2, 5]
    Objetivo: MAXIMIZAR
    """
    termo1 = np.exp(-(x[0]**2 + x[1]**2))
    termo2 = 2 * np.exp(-((x[0] - 1.7)**2 + (x[1] - 1.7)**2))
    return termo1 + termo2


def f3(x: np.ndarray) -> float:
    """
    Função 3 (Ackley): f(x1, x2) = -20·e^(-0.2·√(0.5·(x1²+x2²))) - e^(0.5·(cos(2πx1)+cos(2πx2))) + 20 + e
    Domínio: x1, x2 ∈ [-8, 8]
    Objetivo: MINIMIZAR
    """
    termo1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2)))
    termo2 = -np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return termo1 + termo2 + 20 + np.e


def f4(x: np.ndarray) -> float:
    """
    Função 4 (Rastrigin): f(x1, x2) = (x1² - 10·cos(2πx1) + 10) + (x2² - 10·cos(2πx2) + 10)
    Domínio: x1, x2 ∈ [-5.12, 5.12]
    Objetivo: MINIMIZAR
    """
    termo1 = x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10
    termo2 = x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10
    return termo1 + termo2


def f5(x: np.ndarray) -> float:
    """
    Função 5: f(x1, x2) = (x1·cos(x1))/20 + 2·e^(-(x1)²-(x2-1)²) + 0.01·x1·x2
    Domínio: x1 ∈ [-10, 10], x2 ∈ [-10, 10]
    Objetivo: MAXIMIZAR
    """
    termo1 = (x[0] * np.cos(x[0])) / 20
    termo2 = 2 * np.exp(-(x[0]**2) - (x[1] - 1)**2)
    termo3 = 0.01 * x[0] * x[1]
    return termo1 + termo2 + termo3


def f6(x: np.ndarray) -> float:
    """
    Função 6: f(x1, x2) = x1·sin(4πx1) - x2·sin(4πx2 + π) + 1
    Domínio: x1 ∈ [-1, 3], x2 ∈ [-1, 3]
    Objetivo: MAXIMIZAR
    """
    termo1 = x[0] * np.sin(4 * np.pi * x[0])
    termo2 = x[1] * np.sin(4 * np.pi * x[1] + np.pi)
    return termo1 - termo2 + 1



PROBLEMAS = {
    1: {
        'funcao': f1,
        'limites': np.array([[-100, 100], [-100, 100]]),
        'objetivo': 'minimizar',
        'nome': 'Esfera',
        'otimo_conhecido': 0.0
    },
    2: {
        'funcao': f2,
        'limites': np.array([[-2, 4], [-2, 5]]),
        'objetivo': 'maximizar',
        'nome': 'Gaussiana Dupla',
        'otimo_conhecido': 2.0  # máximo em (1.7, 1.7)
    },
    3: {
        'funcao': f3,
        'limites': np.array([[-8, 8], [-8, 8]]),
        'objetivo': 'minimizar',
        'nome': 'Ackley',
        'otimo_conhecido': 0.0
    },
    4: {
        'funcao': f4,
        'limites': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'objetivo': 'minimizar',
        'nome': 'Rastrigin',
        'otimo_conhecido': 0.0
    },
    5: {
        'funcao': f5,
        'limites': np.array([[-10, 10], [-10, 10]]),
        'objetivo': 'maximizar',
        'nome': 'Custom 5',
        'otimo_conhecido': 2.0  # aproximado
    },
    6: {
        'funcao': f6,
        'limites': np.array([[-1, 3], [-1, 3]]),
        'objetivo': 'maximizar',
        'nome': 'Seno Modulado',
        'otimo_conhecido': 6.0  # aproximado
    }
}


# =============================================================================
# ALGORITMO HILL CLIMBING
# =============================================================================

class HillClimbing:
    
    def __init__(self, funcao: Callable, limites: np.ndarray, 
                 objetivo: str = 'minimizar', epsilon: float = 0.1,
                 max_it: int = 1000, max_vizinhos: int = 100,
                 parada_sem_melhoria: int = 100):

        self.funcao = funcao
        self.limites = limites
        self.objetivo = objetivo
        self.epsilon = epsilon
        self.max_it = max_it
        self.max_vizinhos = max_vizinhos
        self.parada_sem_melhoria = parada_sem_melhoria
        self.p = limites.shape[0]
        
    def _gerar_candidato_vizinho(self, x_best: np.ndarray) -> np.ndarray:
        y = np.random.uniform(
            low=x_best - self.epsilon,
            high=x_best + self.epsilon
        )
        y = np.clip(y, self.limites[:, 0], self.limites[:, 1])
        return y
    
    def _melhor(self, f_cand: float, f_best: float) -> bool:
        if self.objetivo == 'minimizar':
            return f_cand < f_best
        else:
            return f_cand > f_best
    
    def search(self) -> Tuple[np.ndarray, float, List[float]]:
        x_best = self.limites[:, 0].copy()
        f_best = self.funcao(x_best)
        historico = [f_best]
        it = 0
        iteracoes_sem_melhoria = 0
        
        while it < self.max_it and iteracoes_sem_melhoria < self.parada_sem_melhoria:
            melhoria = False
            
            for _ in range(self.max_vizinhos):
                y = self._gerar_candidato_vizinho(x_best)
                f_cand = self.funcao(y)
                
                if self._melhor(f_cand, f_best):
                    x_best = y.copy()
                    f_best = f_cand
                    melhoria = True
                    break
            
            historico.append(f_best)
            
            if melhoria:
                iteracoes_sem_melhoria = 0
            else:
                iteracoes_sem_melhoria += 1
            
            it += 1
        
        return x_best, f_best, historico


# =============================================================================
# ALGORITMO LOCAL RANDOM SEARCH (LRS)
# =============================================================================

class LocalRandomSearch:    
    def __init__(self, funcao: Callable, limites: np.ndarray,
                 objetivo: str = 'minimizar', sigma: float = 0.5,
                 max_it: int = 1000, parada_sem_melhoria: int = 100):

        self.funcao = funcao
        self.limites = limites
        self.objetivo = objetivo
        self.sigma = sigma
        self.max_it = max_it
        self.parada_sem_melhoria = parada_sem_melhoria
        self.p = limites.shape[0]
        
    def _verificar_restricao_caixa(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.limites[:, 0], self.limites[:, 1])
    
    def _melhor(self, f_cand: float, f_best: float) -> bool:
        if self.objetivo == 'minimizar':
            return f_cand < f_best
        else:
            return f_cand > f_best
    
    def search(self) -> Tuple[np.ndarray, float, List[float]]:
        x_best = np.random.uniform(
            self.limites[:, 0],
            self.limites[:, 1]
        )
        f_best = self.funcao(x_best)
        
        historico = [f_best]
        it = 0
        iteracoes_sem_melhoria = 0
        
        while it < self.max_it and iteracoes_sem_melhoria < self.parada_sem_melhoria:
            n = np.random.normal(0, self.sigma, size=self.p)
            x_cand = x_best + n
            x_cand = self._verificar_restricao_caixa(x_cand)
            f_cand = self.funcao(x_cand)
            if self._melhor(f_cand, f_best):
                x_best = x_cand.copy()
                f_best = f_cand
                iteracoes_sem_melhoria = 0
            else:
                iteracoes_sem_melhoria += 1
            
            historico.append(f_best)
            it += 1
        
        return x_best, f_best, historico


# =============================================================================
# ALGORITMO GLOBAL RANDOM SEARCH (GRS)
# =============================================================================

class GlobalRandomSearch:    
    def __init__(self, funcao: Callable, limites: np.ndarray,
                 objetivo: str = 'minimizar', max_it: int = 1000,
                 parada_sem_melhoria: int = 100):

        self.funcao = funcao
        self.limites = limites
        self.objetivo = objetivo
        self.max_it = max_it
        self.parada_sem_melhoria = parada_sem_melhoria
        self.p = limites.shape[0]
        
    def _melhor(self, f_cand: float, f_best: float) -> bool:
        if self.objetivo == 'minimizar':
            return f_cand < f_best
        else:
            return f_cand > f_best
    
    def search(self) -> Tuple[np.ndarray, float, List[float]]:
        x_best = np.random.uniform(
            self.limites[:, 0],
            self.limites[:, 1]
        )
        f_best = self.funcao(x_best)
        historico = [f_best]
        it = 0
        iteracoes_sem_melhoria = 0
        
        while it < self.max_it and iteracoes_sem_melhoria < self.parada_sem_melhoria:
            x_cand = np.random.uniform(
                self.limites[:, 0],
                self.limites[:, 1]
            )
            
            f_cand = self.funcao(x_cand)
            
            if self._melhor(f_cand, f_best):
                x_best = x_cand.copy()
                f_best = f_cand
                iteracoes_sem_melhoria = 0
            else:
                iteracoes_sem_melhoria += 1
            
            historico.append(f_best)
            it += 1
        
        return x_best, f_best, historico


# =============================================================================
# EXPERIMENTO PRINCIPAL
# - 100 rodadas por algoritmo/problema
# - Resultado: moda das soluções
# - Máximo de 1000 iterações por execução
# =============================================================================

def executar_experimento(num_problema: int, num_rodadas: int = 100,
                         epsilon_hc: float = 0.1, sigma_lrs: float = 0.5,
                         max_it: int = 1000, verbose: bool = True):

    problema = PROBLEMAS[num_problema]
    funcao = problema['funcao']
    limites = problema['limites']
    objetivo = problema['objetivo']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PROBLEMA {num_problema}: {problema['nome']}")
        print(f"Objetivo: {objetivo.upper()}")
        print(f"Domínio: x1 ∈ [{limites[0,0]}, {limites[0,1]}], x2 ∈ [{limites[1,0]}, {limites[1,1]}]")
        print(f"{'='*60}")
    
    resultados = {
        'Hill Climbing': {'solucoes': [], 'valores': [], 'x_best': []},
        'LRS': {'solucoes': [], 'valores': [], 'x_best': []},
        'GRS': {'solucoes': [], 'valores': [], 'x_best': []}
    }
    
    for rodada in range(num_rodadas):
        if verbose and (rodada + 1) % 20 == 0:
            print(f"  Rodada {rodada + 1}/{num_rodadas}...")
        
        # Hill Climbing
        hc = HillClimbing(funcao, limites, objetivo, epsilon=epsilon_hc, max_it=max_it)
        x_hc, f_hc, _ = hc.search()
        resultados['Hill Climbing']['solucoes'].append(round(f_hc, 4))
        resultados['Hill Climbing']['valores'].append(f_hc)
        resultados['Hill Climbing']['x_best'].append(x_hc)
        
        # Local Random Search
        lrs = LocalRandomSearch(funcao, limites, objetivo, sigma=sigma_lrs, max_it=max_it)
        x_lrs, f_lrs, _ = lrs.search()
        resultados['LRS']['solucoes'].append(round(f_lrs, 4))
        resultados['LRS']['valores'].append(f_lrs)
        resultados['LRS']['x_best'].append(x_lrs)
        
        # Global Random Search
        grs = GlobalRandomSearch(funcao, limites, objetivo, max_it=max_it)
        x_grs, f_grs, _ = grs.search()
        resultados['GRS']['solucoes'].append(round(f_grs, 4))
        resultados['GRS']['valores'].append(f_grs)
        resultados['GRS']['x_best'].append(x_grs)
    
    # Calcular estatísticas
    for alg in resultados:
        valores = resultados[alg]['valores']
        solucoes = resultados[alg]['solucoes']
        
        # Moda das soluções
        try:
            moda = stats.mode(solucoes, keepdims=True)[0][0]
        except:
            moda = stats.mode(solucoes)[0]
        
        resultados[alg]['moda'] = moda
        resultados[alg]['media'] = np.mean(valores)
        resultados[alg]['desvio'] = np.std(valores)
        resultados[alg]['melhor'] = np.min(valores) if objetivo == 'minimizar' else np.max(valores)
        resultados[alg]['pior'] = np.max(valores) if objetivo == 'minimizar' else np.min(valores)
    
    return resultados


def exibir_tabela_resultados(resultados: dict, num_problema: int):
    """Exibe tabela formatada com resultados"""
    problema = PROBLEMAS[num_problema]
    
    print(f"\n{'='*80}")
    print(f"TABELA DE RESULTADOS - PROBLEMA {num_problema} ({problema['nome']})")
    print(f"{'='*80}")
    print(f"{'Algoritmo':<20} {'Moda':<12} {'Média':<12} {'Desvio':<12} {'Melhor':<12} {'Pior':<12}")
    print(f"{'-'*80}")
    
    for alg, dados in resultados.items():
        print(f"{alg:<20} {dados['moda']:<12.4f} {dados['media']:<12.4f} "
              f"{dados['desvio']:<12.4f} {dados['melhor']:<12.4f} {dados['pior']:<12.4f}")
    
    print(f"{'='*80}")


def plotar_convergencia(num_problema: int, epsilon_hc: float = 0.1, 
                        sigma_lrs: float = 0.5, max_it: int = 1000):
    """Plota gráfico de convergência dos três algoritmos"""
    problema = PROBLEMAS[num_problema]
    funcao = problema['funcao']
    limites = problema['limites']
    objetivo = problema['objetivo']
    
    # Executar uma vez cada algoritmo e guardar histórico
    hc = HillClimbing(funcao, limites, objetivo, epsilon=epsilon_hc, max_it=max_it)
    _, _, hist_hc = hc.search()
    
    lrs = LocalRandomSearch(funcao, limites, objetivo, sigma=sigma_lrs, max_it=max_it)
    _, _, hist_lrs = lrs.search()
    
    grs = GlobalRandomSearch(funcao, limites, objetivo, max_it=max_it)
    _, _, hist_grs = grs.search()
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_hc, label='Hill Climbing', alpha=0.8)
    plt.plot(hist_lrs, label='LRS', alpha=0.8)
    plt.plot(hist_grs, label='GRS', alpha=0.8)
    plt.xlabel('Iterações')
    plt.ylabel('f(x)')
    plt.title(f'Convergência - Problema {num_problema} ({problema["nome"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plotar_superficie_3d(num_problema: int, resolucao: int = 100):
    """Plota superfície 3D da função objetivo"""
    problema = PROBLEMAS[num_problema]
    funcao = problema['funcao']
    limites = problema['limites']
    
    x1 = np.linspace(limites[0, 0], limites[0, 1], resolucao)
    x2 = np.linspace(limites[1, 0], limites[1, 1], resolucao)
    X1, X2 = np.meshgrid(x1, x2)
    
    Z = np.zeros_like(X1)
    for i in range(resolucao):
        for j in range(resolucao):
            Z[i, j] = funcao(np.array([X1[i, j], X2[i, j]]))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8, 
                    rstride=5, cstride=5, edgecolors='k', lw=0.1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title(f'Problema {num_problema}: {problema["nome"]}')
    
    return fig


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():

    print("\n" + "="*80)
    print("TRABALHO AV3 - BUSCA/OTIMIZAÇÃO META-HEURÍSTICA")
    print("QUESTÃO 1: Problema de Minimização/Maximização de função Custo/Objetivo")
    print("="*80)
    print("\nAlgoritmos implementados:")
    print("  1. Hill Climbing (Subida de Encosta)")
    print("  2. Local Random Search (LRS)")
    print("  3. Global Random Search (GRS)")
    print("\nParâmetros:")
    print("  - Máximo de iterações: 1000")
    print("  - Número de rodadas: 100")
    print("  - Epsilon (HC): 0.1")
    print("  - Sigma (LRS): 0.5")
    
    # Armazenar todos os resultados
    todos_resultados = {}
    
    # Executar para cada problema
    for num_problema in range(1, 7):
        resultados = executar_experimento(
            num_problema=num_problema,
            num_rodadas=100,
            epsilon_hc=0.1,
            sigma_lrs=0.5,
            max_it=1000,
            verbose=True
        )
        todos_resultados[num_problema] = resultados
        exibir_tabela_resultados(resultados, num_problema)
    
    # Resumo final
    print("\n" + "="*80)
    print("RESUMO FINAL - MODA DAS SOLUÇÕES")
    print("="*80)
    print(f"{'Problema':<12} {'Hill Climbing':<15} {'LRS':<15} {'GRS':<15} {'Objetivo':<12}")
    print("-"*80)
    
    for num_problema in range(1, 7):
        res = todos_resultados[num_problema]
        obj = PROBLEMAS[num_problema]['objetivo']
        print(f"{num_problema:<12} {res['Hill Climbing']['moda']:<15.4f} "
              f"{res['LRS']['moda']:<15.4f} {res['GRS']['moda']:<15.4f} {obj:<12}")
    
    print("="*80)
    
    return todos_resultados


if __name__ == '__main__':
    resultados = main()
    
    for i in range(1, 7):
        plotar_convergencia(i)
        plotar_superficie_3d(i)
    plt.show()
