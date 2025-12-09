import numpy as np
import time

class SimulatedAnnealing:
    def __init__(self, T0=200, alpha=0.96, max_it=20000):
        self.T0 = T0
        self.alpha = alpha
        self.max_it = max_it
        self.x_opt = np.random.randint(1, 9, size=8)
        self.f_opt = self.f(self.x_opt)
        self.historico = [self.f_opt]
        
    def h(self, x):
        conflicts = 0
        n = len(x)
        for i in range(n):
            for j in range(i+1, n):
                if x[i] == x[j]:
                    conflicts += 1
                elif abs(i - j) == abs(x[i] - x[j]):
                    conflicts += 1
        return conflicts
    
    def f(self, x):
        return 28 - self.h(x)
    
    def perturb(self):
        x_cand = np.copy(self.x_opt)
        strategy = np.random.randint(0, 2)
        
        if strategy == 0:
            col = np.random.randint(0, 8)
            direction = np.random.choice([-1, 1])
            new_row = x_cand[col] + direction
            if 1 <= new_row <= 8:
                x_cand[col] = new_row
            else:
                x_cand[col] = np.random.randint(1, 9)
        else:
            cols = np.random.choice(8, size=2, replace=False)
            x_cand[cols[0]], x_cand[cols[1]] = x_cand[cols[1]], x_cand[cols[0]]
        
        return x_cand
    
    def search(self):
        T = self.T0
        it = 0
        
        while it < self.max_it and self.f_opt < 28:
            x_cand = self.perturb()
            f_cand = self.f(x_cand)
            delta = f_cand - self.f_opt
            
            if delta > 0 or np.random.uniform(0, 1) < np.exp(delta / T):
                self.x_opt = x_cand
                self.f_opt = f_cand
            
            self.historico.append(self.f_opt)
            T = T * self.alpha
            it += 1
        
        return self.x_opt, self.f_opt, it


def buscar_92_solucoes():
    print("="*70)
    print("BUSCANDO AS 92 SOLUÇÕES ÚNICAS DO PROBLEMA DAS 8 RAINHAS")
    print("="*70)
    
    solucoes_unicas = set()
    tentativas = 0
    iteracoes_totais = 0
    tempo_inicio = time.time()
    historico_tentativas = []
    
    while len(solucoes_unicas) < 92:
        tentativas += 1
        sa = SimulatedAnnealing(T0=200, alpha=0.96, max_it=20000)
        x_final, f_final, iterations = sa.search()
        iteracoes_totais += iterations
        
        if f_final == 28:
            sol_tuple = tuple(x_final)
            era_nova = sol_tuple not in solucoes_unicas
            solucoes_unicas.add(sol_tuple)
            
            historico_tentativas.append({
                'tentativa': tentativas,
                'solucao': x_final.copy(),
                'iteracoes': iterations,
                'nova': era_nova
            })
            
            if era_nova:
                print(f"Tentativa {tentativas:4d} | Soluções: {len(solucoes_unicas):2d}/92 | "
                      f"Iter: {iterations:5d} | ✓ NOVA")
            else:
                print(f"Tentativa {tentativas:4d} | Soluções: {len(solucoes_unicas):2d}/92 | "
                      f"Iter: {iterations:5d} | ✗ repetida")
        else:
            if tentativas % 10 == 0:
                print(f"Tentativa {tentativas:4d} | Soluções: {len(solucoes_unicas):2d}/92 | "
                      f"Iter: {iterations:5d} | ⚠ f={f_final}")
        
        if tentativas % 100 == 0:
            tempo_parcial = time.time() - tempo_inicio
            print(f"\n>>> Checkpoint: {tentativas} tentativas, {tempo_parcial:.1f}s, "
                  f"{len(solucoes_unicas)} soluções\n")
    
    tempo_total = time.time() - tempo_inicio
    
    print("\n" + "="*70)
    print("CONCLUÍDO!")
    print("="*70)
    
    solucoes_novas = [h for h in historico_tentativas if h['nova']]
    solucoes_repetidas = [h for h in historico_tentativas if not h['nova']]
    
    print(f"\nCUSTO COMPUTACIONAL:")
    print(f"  Total de tentativas: {tentativas}")
    print(f"  Soluções únicas: {len(solucoes_unicas)}")
    print(f"  Soluções repetidas: {len(solucoes_repetidas)}")
    print(f"  Taxa de sucesso: {100*len(historico_tentativas)/tentativas:.1f}%")
    print(f"  Iterações totais: {iteracoes_totais:,}")
    print(f"  Iterações médias: {iteracoes_totais/tentativas:.0f}")
    print(f"  Tempo total: {tempo_total:.2f}s")
    print(f"  Tempo médio/tentativa: {tempo_total/tentativas:.3f}s")
    
    if solucoes_novas:
        iter_novas = [s['iteracoes'] for s in solucoes_novas]
        print(f"\nSOLUÇÕES NOVAS:")
        print(f"  Iter mínima: {min(iter_novas)}")
        print(f"  Iter máxima: {max(iter_novas)}")
        print(f"  Iter média: {np.mean(iter_novas):.0f}")
    
    # Salvar resultados
    with open('92_solucoes.txt', 'w') as f:
        f.write("92 SOLUÇÕES ÚNICAS DO PROBLEMA DAS 8 RAINHAS\n")
        f.write("="*70 + "\n\n")
        for i, sol in enumerate(sorted(solucoes_unicas), 1):
            f.write(f"Solução {i:2d}: {list(sol)}\n")
        f.write(f"\n\nESTATÍSTICAS:\n")
        f.write(f"Tentativas: {tentativas}\n")
        f.write(f"Iterações totais: {iteracoes_totais:,}\n")
        f.write(f"Tempo total: {tempo_total:.2f}s\n")
        f.write(f"Taxa de sucesso: {100*len(historico_tentativas)/tentativas:.1f}%\n")
    
    print("\n✓ Arquivo salvo: 92_solucoes.txt")
    

if __name__ == '__main__':
    # TESTE RÁPIDO: encontrar 1 solução
    print("TESTE: Buscando 1 solução ótima...")
    sa = SimulatedAnnealing(T0=150, alpha=0.97, max_it=15000)
    x, f, it = sa.search()
    print(f"Solução: {x}, f={f}, h={sa.h(x)}, iter={it}")
    
    if f == 28:
        print("✓ Teste OK! Iniciando busca das 92 soluções...\n")
        buscar_92_solucoes()
    else:
        print("⚠ Não convergiu no teste. Rodando mesmo assim...\n")
        buscar_92_solucoes()