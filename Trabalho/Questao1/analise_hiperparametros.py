import numpy as np
import matplotlib.pyplot as plt
from questao1_otimizacao import (
    HillClimbing, LocalRandomSearch,
    PROBLEMAS, f1, f2, f3, f4, f5, f6
)
import warnings
warnings.filterwarnings('ignore')


def analisar_epsilon_hill_climbing(num_problema: int, 
                                    epsilons: list = None,
                                    num_rodadas: int = 30,
                                    max_it: int = 1000):
    """
    Analisa diferentes valores de epsilon para Hill Climbing
    e identifica o menor valor que encontra a solução ótima.
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    problema = PROBLEMAS[num_problema]
    funcao = problema['funcao']
    limites = problema['limites']
    objetivo = problema['objetivo']
    otimo = problema['otimo_conhecido']
    
    print(f"\n{'='*60}")
    print(f"ANÁLISE DE EPSILON - HILL CLIMBING")
    print(f"Problema {num_problema}: {problema['nome']}")
    print(f"Ótimo conhecido: {otimo}")
    print(f"{'='*60}")
    
    resultados = {}
    
    for eps in epsilons:
        valores = []
        for _ in range(num_rodadas):
            hc = HillClimbing(funcao, limites, objetivo, epsilon=eps, max_it=max_it)
            _, f_best, _ = hc.search()
            valores.append(f_best)
        
        media = np.mean(valores)
        desvio = np.std(valores)
        melhor = np.min(valores) if objetivo == 'minimizar' else np.max(valores)
        
        resultados[eps] = {
            'media': media,
            'desvio': desvio,
            'melhor': melhor,
            'valores': valores
        }
        
        print(f"  ε = {eps:<6.2f} | Média: {media:>12.4f} | Desvio: {desvio:>10.4f} | Melhor: {melhor:>12.4f}")
    
    return resultados


def analisar_sigma_lrs(num_problema: int,
                       sigmas: list = None,
                       num_rodadas: int = 30,
                       max_it: int = 1000):
    """
    Analisa diferentes valores de sigma para Local Random Search
    e identifica o menor valor que encontra a solução ótima.
    """
    if sigmas is None:
        sigmas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    problema = PROBLEMAS[num_problema]
    funcao = problema['funcao']
    limites = problema['limites']
    objetivo = problema['objetivo']
    otimo = problema['otimo_conhecido']
    
    print(f"\n{'='*60}")
    print(f"ANÁLISE DE SIGMA - LOCAL RANDOM SEARCH (LRS)")
    print(f"Problema {num_problema}: {problema['nome']}")
    print(f"Ótimo conhecido: {otimo}")
    print(f"{'='*60}")
    
    resultados = {}
    
    for sigma in sigmas:
        valores = []
        for _ in range(num_rodadas):
            lrs = LocalRandomSearch(funcao, limites, objetivo, sigma=sigma, max_it=max_it)
            _, f_best, _ = lrs.search()
            valores.append(f_best)
        
        media = np.mean(valores)
        desvio = np.std(valores)
        melhor = np.min(valores) if objetivo == 'minimizar' else np.max(valores)
        
        resultados[sigma] = {
            'media': media,
            'desvio': desvio,
            'melhor': melhor,
            'valores': valores
        }
        
        print(f"  σ = {sigma:<6.2f} | Média: {media:>12.4f} | Desvio: {desvio:>10.4f} | Melhor: {melhor:>12.4f}")
    
    return resultados


def analisar_todos_problemas():
    """
    Executa análise de hiperparâmetros para todos os 6 problemas.
    """
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA DE HIPERPARÂMETROS")
    print("="*80)
    
    # Epsilons para Hill Climbing (valores pequenos)
    epsilons_hc = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    
    # Sigmas para LRS (0 < σ < 1)
    sigmas_lrs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    resultados_completos = {}
    
    for num_problema in range(1, 7):
        print(f"\n\n{'#'*80}")
        print(f"# PROBLEMA {num_problema}: {PROBLEMAS[num_problema]['nome']}")
        print(f"{'#'*80}")
        
        # Análise Hill Climbing
        res_hc = analisar_epsilon_hill_climbing(
            num_problema, 
            epsilons=epsilons_hc,
            num_rodadas=30,
            max_it=1000
        )
        
        # Análise LRS
        res_lrs = analisar_sigma_lrs(
            num_problema,
            sigmas=sigmas_lrs,
            num_rodadas=30,
            max_it=1000
        )
        
        resultados_completos[num_problema] = {
            'Hill Climbing': res_hc,
            'LRS': res_lrs
        }
    
    return resultados_completos


def identificar_melhor_hiperparametro(resultados: dict, objetivo: str):
    """
    Identifica o menor valor de hiperparâmetro que encontra 
    a solução próxima ao ótimo.
    """
    if objetivo == 'minimizar':
        melhor_param = min(resultados.keys(), key=lambda k: resultados[k]['media'])
    else:
        melhor_param = max(resultados.keys(), key=lambda k: resultados[k]['media'])
    
    return melhor_param, resultados[melhor_param]


def gerar_relatorio_hiperparametros():

    print("\n" + "="*80)
    print("RELATÓRIO: MELHORES HIPERPARÂMETROS POR PROBLEMA")
    print("="*80)
    
    resultados = analisar_todos_problemas()
    
    print("\n\n" + "="*80)
    print("RESUMO - MELHORES HIPERPARÂMETROS IDENTIFICADOS")
    print("="*80)
    print(f"\n{'Problema':<12} {'Algoritmo':<20} {'Parâmetro':<15} {'Valor':<10} {'Média f(x)':<15}")
    print("-"*80)
    
    for num_problema in range(1, 7):
        objetivo = PROBLEMAS[num_problema]['objetivo']
        
        # Hill Climbing
        melhor_eps, res_eps = identificar_melhor_hiperparametro(
            resultados[num_problema]['Hill Climbing'], objetivo
        )
        print(f"{num_problema:<12} {'Hill Climbing':<20} {'epsilon':<15} {melhor_eps:<10.2f} {res_eps['media']:<15.4f}")
        
        # LRS
        melhor_sigma, res_sigma = identificar_melhor_hiperparametro(
            resultados[num_problema]['LRS'], objetivo
        )
        print(f"{'':<12} {'LRS':<20} {'sigma':<15} {melhor_sigma:<10.2f} {res_sigma['media']:<15.4f}")
        print("-"*80)
    
    return resultados


if __name__ == '__main__':
    resultados = gerar_relatorio_hiperparametros()
