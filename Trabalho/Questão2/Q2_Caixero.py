import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
from collections import Counter

class TSPGeneticAlgorithm:
    def __init__(self, pontos, grupos, N=100, max_gen=200, nst=3, pr=0.88, pm=0.01, elitismo=0):
        self.pontos = pontos
        self.grupos = grupos
        self.n_pontos = len(pontos)
        self.N = N
        self.max_gen = max_gen
        self.nst = nst
        self.pr = pr
        self.pm = pm
        self.elitismo = elitismo
        
        self.population = [self.random_route() for _ in range(N)]
        self.fitness = np.array([self.f(ind) for ind in self.population])
        
        self.hist_best = []
        self.hist_worst = []
        self.hist_avg = []
        
    def random_route(self):
        route = np.concatenate(([0], np.random.permutation(range(1, self.n_pontos))))
        return route
    
    def f(self, route):
        dist = 0.0
        for i in range(self.n_pontos):
            p1 = self.pontos[route[i]]
            p2 = self.pontos[route[(i+1) % self.n_pontos]]
            dist += np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        return dist
    
    def selection(self):
        selected = []
        for _ in range(self.N):
            indices = np.random.choice(self.N, size=self.nst, replace=False)
            winner_idx = indices[np.argmin(self.fitness[indices])]
            selected.append(self.population[winner_idx].copy())
        return selected
    
    def crossover_2pontos(self, pai1, pai2):
        n = len(pai1)
        pontos = sorted(np.random.choice(range(1, n), size=2, replace=False))
        p1, p2 = pontos[0], pontos[1]
        
        filho1 = np.full(n, -1, dtype=int)
        filho2 = np.full(n, -1, dtype=int)
        
        filho1[0] = 0
        filho2[0] = 0
        
        filho1[p1:p2] = pai1[p1:p2]
        filho2[p1:p2] = pai2[p1:p2]
        
        pos = p2
        for gene in np.concatenate([pai2[p2:], pai2[1:p2]]):
            if gene not in filho1:
                filho1[pos % n] = gene
                pos += 1
                if pos % n == 0:
                    pos = 1
        
        pos = p2
        for gene in np.concatenate([pai1[p2:], pai1[1:p2]]):
            if gene not in filho2:
                filho2[pos % n] = gene
                pos += 1
                if pos % n == 0:
                    pos = 1
        
        return filho1, filho2
    
    def mutation_swap(self, individual):
        if np.random.uniform() < self.pm:
            idx = np.random.choice(range(1, self.n_pontos), size=2, replace=False)
            individual[idx[0]], individual[idx[1]] = individual[idx[1]], individual[idx[0]]
        return individual
    
    def evolve(self):
        gen = 0
        geracoes_sem_melhora = 0
        melhor_anterior = self.fitness.min()
        
        while gen < self.max_gen:
            selected = self.selection()
            
            offspring = []
            for i in range(0, self.N, 2):
                if i+1 < self.N:
                    if np.random.uniform() < self.pr:
                        f1, f2 = self.crossover_2pontos(selected[i], selected[i+1])
                        offspring.append(f1)
                        offspring.append(f2)
                    else:
                        offspring.append(selected[i].copy())
                        offspring.append(selected[i+1].copy())
            
            offspring = [self.mutation_swap(ind) for ind in offspring]
            
            if self.elitismo > 0:
                elite_idx = np.argsort(self.fitness)[:self.elitismo]
                elite = [self.population[i].copy() for i in elite_idx]
                offspring[:self.elitismo] = elite
            
            self.population = offspring[:self.N]
            self.fitness = np.array([self.f(ind) for ind in self.population])
            
            self.hist_best.append(self.fitness.min())
            self.hist_worst.append(self.fitness.max())
            self.hist_avg.append(self.fitness.mean())
            
            melhor_atual = self.fitness.min()
            if melhor_atual < melhor_anterior - 0.01:
                geracoes_sem_melhora = 0
                melhor_anterior = melhor_atual
            else:
                geracoes_sem_melhora += 1
            
            if geracoes_sem_melhora >= 50:
                break
            
            gen += 1
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx], gen


def carregar_pontos(filename):
    data = np.loadtxt(filename, delimiter=',')
    pontos = data[:, :3]  # x, y, z
    grupos = data[:, 3].astype(int)  # grupo
    
    origem_idx = np.where((pontos[:, 0] == 0) & (pontos[:, 1] == 0) & (pontos[:, 2] == 0))[0][0]
    
    pontos_reorganizados = np.vstack([pontos[origem_idx], 
                                       np.delete(pontos, origem_idx, axis=0)])
    grupos_reorganizados = np.concatenate([[grupos[origem_idx]], 
                                           np.delete(grupos, origem_idx)])
    
    return pontos_reorganizados, grupos_reorganizados


def plotar_rota_3d(pontos, grupos, rota, titulo, output_path):
    """Plota rota 3D com pontos coloridos por grupo"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Cores por grupo
    cores_grupos = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}
    nomes_grupos = {0: 'Origem', 1: 'Grupo 1', 2: 'Grupo 2', 3: 'Grupo 3', 4: 'Grupo 4'}
    
    # Plotar pontos por grupo
    for grupo in np.unique(grupos):
        mask = grupos == grupo
        ax.scatter(pontos[mask, 0], pontos[mask, 1], pontos[mask, 2],
                  c=cores_grupos[grupo], s=50, alpha=0.6, label=nomes_grupos[grupo])
    
    # Plotar rota
    rota_pontos = pontos[rota]
    ax.plot(rota_pontos[:, 0], rota_pontos[:, 1], rota_pontos[:, 2],
           'k-', alpha=0.3, linewidth=1)
    
    # Destacar origem
    ax.scatter(pontos[0, 0], pontos[0, 1], pontos[0, 2],
              c='black', s=200, marker='*', edgecolors='yellow', linewidths=2,
              label='Origem', zorder=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analisar_grupos_na_rota(grupos, rota):
    """Analisa como a rota passa pelos grupos"""
    sequencia_grupos = [grupos[i] for i in rota]
    
    # Contar transições entre grupos
    transicoes = {}
    for i in range(len(sequencia_grupos)-1):
        g1, g2 = sequencia_grupos[i], sequencia_grupos[i+1]
        if g1 != g2:
            par = (g1, g2)
            transicoes[par] = transicoes.get(par, 0) + 1
    
    return sequencia_grupos, transicoes


def analisar_convergencia(pontos, grupos, num_runs=30, com_elitismo=False, output_dir='.'):
    geracoes_convergencia = []
    fitness_final = []
    tempos = []
    
    print(f"\n{'='*70}")
    print(f"ANÁLISE: {num_runs} EXECUÇÕES {'COM' if com_elitismo else 'SEM'} ELITISMO")
    print(f"{'='*70}\n")
    
    for run in range(1, num_runs + 1):
        tempo_inicio = time.time()
        
        elite = 5 if com_elitismo else 0
        ga = TSPGeneticAlgorithm(pontos, grupos, N=100, max_gen=300, nst=3, pr=0.88, pm=0.01, elitismo=elite)
        melhor_rota, melhor_dist, gens = ga.evolve()
        
        tempo_exec = time.time() - tempo_inicio
        
        geracoes_convergencia.append(gens)
        fitness_final.append(melhor_dist)
        tempos.append(tempo_exec)
        
        print(f"Run {run:2d}/{num_runs} | Gerações: {gens:3d} | "
              f"Distância: {melhor_dist:8.2f} | Tempo: {tempo_exec:.2f}s")
    
    counter = Counter(geracoes_convergencia)
    moda = counter.most_common(1)[0][0]
    freq_moda = counter.most_common(1)[0][1]
    
    print(f"\n{'='*70}")
    print("ESTATÍSTICAS:")
    print(f"{'='*70}")
    print(f"Gerações até convergir:")
    print(f"  Mínimo: {min(geracoes_convergencia)}")
    print(f"  Máximo: {max(geracoes_convergencia)}")
    print(f"  Média: {np.mean(geracoes_convergencia):.1f}")
    print(f"  Moda: {moda} (ocorreu {freq_moda}x)")
    print(f"\nDistância final:")
    print(f"  Melhor: {min(fitness_final):.2f}")
    print(f"  Pior: {max(fitness_final):.2f}")
    print(f"  Média: {np.mean(fitness_final):.2f}")
    print(f"\nTempo médio: {np.mean(tempos):.2f}s")
    
    tipo = "COM_ELITISMO" if com_elitismo else "SEM_ELITISMO"
    with open(os.path.join(output_dir, f'ga_analise_{tipo}.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"ANÁLISE: {num_runs} EXECUÇÕES {tipo.replace('_', ' ')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"GERAÇÕES ATÉ CONVERGIR:\n")
        f.write(f"  Mínimo: {min(geracoes_convergencia)}\n")
        f.write(f"  Máximo: {max(geracoes_convergencia)}\n")
        f.write(f"  Média: {np.mean(geracoes_convergencia):.1f}\n")
        f.write(f"  Moda: {moda} (ocorreu {freq_moda} vezes)\n\n")
        f.write(f"DISTÂNCIA FINAL:\n")
        f.write(f"  Melhor: {min(fitness_final):.2f}\n")
        f.write(f"  Pior: {max(fitness_final):.2f}\n")
        f.write(f"  Média: {np.mean(fitness_final):.2f}\n")
        f.write(f"  Desvio padrão: {np.std(fitness_final):.2f}\n\n")
        f.write(f"TEMPO:\n")
        f.write(f"  Total: {sum(tempos):.2f}s\n")
        f.write(f"  Médio por run: {np.mean(tempos):.2f}s\n\n")
        f.write("DETALHAMENTO POR RUN:\n")
        f.write("-"*70 + "\n")
        for i, (g, d, t) in enumerate(zip(geracoes_convergencia, fitness_final, tempos), 1):
            f.write(f"Run {i:2d}: Ger={g:3d}, Dist={d:8.2f}, Tempo={t:.2f}s\n")
    
    print(f"✓ Relatório salvo: {os.path.join(output_dir, f'ga_analise_{tipo}.txt')}")
    
    return geracoes_convergencia, fitness_final


if __name__ == '__main__':
    print("="*70)
    print("PROBLEMA DO CAIXEIRO VIAJANTE 3D - ALGORITMO GENÉTICO")
    print("COM ANÁLISE DE GRUPOS/REGIÕES")
    print("="*70)
    
    csv_path = r'C:\Users\Calil\Documents\VIDA\Faculdade\2025.2\IA\AV3\Trabalho AV3\-Busca-Otimizacao-IA\Aula\CaixeiroGruposGA.csv'
    output_dir = r'C:\Users\Calil\Documents\VIDA\Faculdade\2025.2\IA\AV3\Trabalho AV3\-Busca-Otimizacao-IA\Trabalho\Questão2\ResultadosQ2'
    
    pontos, grupos = carregar_pontos(csv_path)
    print(f"\nPontos carregados: {len(pontos)}")
    print(f"Grupos identificados: {np.unique(grupos)}")
    print(f"Distribuição por grupo:")
    for g in np.unique(grupos):
        count = np.sum(grupos == g)
        print(f"  Grupo {g}: {count} pontos")
    
    # TESTE RÁPIDO
    print(f"\n{'='*70}")
    print("TESTE: 1 execução")
    print(f"{'='*70}\n")
    
    ga = TSPGeneticAlgorithm(pontos, grupos, N=100, max_gen=200, nst=3, pr=0.88, pm=0.01, elitismo=0)
    melhor_rota, melhor_dist, gens = ga.evolve()
    
    print(f"\nMelhor rota encontrada: {melhor_rota[:10]}... (primeiros 10)")
    print(f"Distância total: {melhor_dist:.2f}")
    print(f"Gerações: {gens}")
    
    # Analisar grupos na rota
    sequencia, transicoes = analisar_grupos_na_rota(grupos, melhor_rota)
    print(f"\nTransições entre grupos: {len(transicoes)}")
    print("Transições mais frequentes:")
    for (g1, g2), count in sorted(transicoes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Grupo {g1} → Grupo {g2}: {count}x")
    
    # Plot evolução fitness
    plt.figure(figsize=(10, 6))
    plt.plot(ga.hist_best, label='Melhor', linewidth=2)
    plt.plot(ga.hist_worst, label='Pior', alpha=0.5)
    plt.plot(ga.hist_avg, label='Média', linestyle='--')
    plt.xlabel('Geração')
    plt.ylabel('Distância')
    plt.title('Evolução do Fitness - Teste Rápido')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'ga_historico.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: {os.path.join(output_dir, 'ga_historico.png')}")
    
    # Plot rota 3D
    plotar_rota_3d(pontos, grupos, melhor_rota, 
                   f'Rota 3D - Distância: {melhor_dist:.2f}',
                   os.path.join(output_dir, 'ga_rota_3d.png'))
    print(f"✓ Rota 3D salva: {os.path.join(output_dir, 'ga_rota_3d.png')}")
    
    # Relatório teste rápido
    with open(os.path.join(output_dir, 'ga_teste_rapido.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TESTE RÁPIDO - ALGORITMO GENÉTICO TSP 3D\n")
        f.write("="*70 + "\n\n")
        f.write(f"Pontos: {len(pontos)}\n")
        f.write(f"Grupos: {len(np.unique(grupos))}\n")
        f.write(f"População: {ga.N}\n")
        f.write(f"Gerações máximas: {ga.max_gen}\n")
        f.write(f"Prob. recombinação: {ga.pr}\n")
        f.write(f"Prob. mutação: {ga.pm}\n\n")
        f.write(f"DISTRIBUIÇÃO POR GRUPO:\n")
        for g in np.unique(grupos):
            count = np.sum(grupos == g)
            f.write(f"  Grupo {g}: {count} pontos\n")
        f.write(f"\nRESULTADO:\n")
        f.write(f"Distância inicial: {ga.hist_best[0]:.2f}\n")
        f.write(f"Distância final: {melhor_dist:.2f}\n")
        f.write(f"Melhoria: {ga.hist_best[0] - melhor_dist:.2f} ({100*(ga.hist_best[0]-melhor_dist)/ga.hist_best[0]:.1f}%)\n")
        f.write(f"Gerações executadas: {gens}\n\n")
        f.write(f"ANÁLISE DE GRUPOS:\n")
        f.write(f"Transições entre grupos: {len(transicoes)}\n")
        f.write(f"Top 5 transições mais frequentes:\n")
        for (g1, g2), count in sorted(transicoes.items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"  Grupo {g1} → Grupo {g2}: {count} vezes\n")
        f.write(f"\nMelhor rota (primeiros 30): {melhor_rota[:30]}\n")
    
    print(f"✓ Relatório salvo: {os.path.join(output_dir, 'ga_teste_rapido.txt')}")
    
    # ANÁLISE SEM ELITISMO
    gens_sem, fit_sem = analisar_convergencia(pontos, grupos, num_runs=30, com_elitismo=False, output_dir=output_dir)
    
    # ANÁLISE COM ELITISMO
    gens_com, fit_com = analisar_convergencia(pontos, grupos, num_runs=30, com_elitismo=True, output_dir=output_dir)
    
    # COMPARAÇÃO
    print(f"\n{'='*70}")
    print("COMPARAÇÃO: COM vs SEM ELITISMO")
    print(f"{'='*70}")
    print(f"{'':20} | SEM Elitismo | COM Elitismo")
    print(f"{'-'*70}")
    print(f"Gerações médias:     | {np.mean(gens_sem):>12.1f} | {np.mean(gens_com):>12.1f}")
    print(f"Distância média:     | {np.mean(fit_sem):>12.2f} | {np.mean(fit_com):>12.2f}")
    print(f"Melhor distância:    | {min(fit_sem):>12.2f} | {min(fit_com):>12.2f}")
    
    # Histograma
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(gens_sem, bins=15, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(gens_sem), color='red', linestyle='--', label=f'Média: {np.mean(gens_sem):.1f}')
    plt.xlabel('Gerações até convergir')
    plt.ylabel('Frequência')
    plt.title('SEM Elitismo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(gens_com, bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(np.mean(gens_com), color='red', linestyle='--', label=f'Média: {np.mean(gens_com):.1f}')
    plt.xlabel('Gerações até convergir')
    plt.ylabel('Frequência')
    plt.title('COM Elitismo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ga_comparacao_elitismo.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo: {os.path.join(output_dir, 'ga_comparacao_elitismo.png')}")
    
    # Relatório comparação
    with open(os.path.join(output_dir, 'ga_comparacao_final.txt'), 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("COMPARAÇÃO FINAL: COM vs SEM ELITISMO\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'':25} | SEM Elitismo | COM Elitismo\n")
        f.write("-"*70 + "\n")
        f.write(f"Gerações médias:          | {np.mean(gens_sem):>12.1f} | {np.mean(gens_com):>12.1f}\n")
        f.write(f"Gerações moda:            | {Counter(gens_sem).most_common(1)[0][0]:>12d} | {Counter(gens_com).most_common(1)[0][0]:>12d}\n")
        f.write(f"Distância média:          | {np.mean(fit_sem):>12.2f} | {np.mean(fit_com):>12.2f}\n")
        f.write(f"Melhor distância:         | {min(fit_sem):>12.2f} | {min(fit_com):>12.2f}\n")
        f.write(f"Pior distância:           | {max(fit_sem):>12.2f} | {max(fit_com):>12.2f}\n")
        f.write(f"Desvio padrão distância:  | {np.std(fit_sem):>12.2f} | {np.std(fit_com):>12.2f}\n\n")
        
        f.write("CONCLUSÃO:\n")
        if np.mean(fit_com) < np.mean(fit_sem):
            dif = np.mean(fit_sem) - np.mean(fit_com)
            perc = 100 * dif / np.mean(fit_sem)
            f.write(f"✓ Elitismo MELHORA os resultados\n")
            f.write(f"  Redução média de distância: {dif:.2f} ({perc:.1f}%)\n")
        else:
            dif = np.mean(fit_com) - np.mean(fit_sem)
            perc = 100 * dif / np.mean(fit_sem)
            f.write(f"⚠ Elitismo NÃO melhora os resultados\n")
            f.write(f"  Aumento médio de distância: {dif:.2f} (+{perc:.1f}%)\n")
            f.write(f"  Possível causa: convergência prematura / perda de diversidade\n")
    
    print(f"✓ Relatório de comparação salvo: {os.path.join(output_dir, 'ga_comparacao_final.txt')}")
    
    print(f"\n{'='*70}")
    print("CONCLUSÃO:")
    if np.mean(fit_com) < np.mean(fit_sem):
        print("✓ Elitismo MELHORA os resultados")
    else:
        print("⚠ Elitismo NÃO melhora (possível convergência prematura)")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("ANÁLISE COMPLETA FINALIZADA!")
    print(f"{'='*70}")
    print(f"\nArquivos salvos em: {output_dir}")
    print("- ga_teste_rapido.txt (com análise de grupos)")
    print("- ga_analise_SEM_ELITISMO.txt")
    print("- ga_analise_COM_ELITISMO.txt")
    print("- ga_comparacao_final.txt")
    print("- ga_historico.png")
    print("- ga_rota_3d.png (NOVO! Visualização 3D por grupos)")
    print("- ga_comparacao_elitismo.png")
    print(f"{'='*70}")