# dispersionTM_test1

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# 1. Definizione Simbolica (Analisi di Von Neumann)
dt, h, c, k, w = sp.symbols('dt h c k omega')
theta = 0.5  # Il tuo valore
I = sp.I

# Rappresentazione del Theta-method per l'equazione delle onde
# u^{n+1} - 2u^n + u^{n-1} / dt^2 = c^2 * L_h(theta*u^{n+1} + (1-2theta)*u^n + theta*u^{n-1})
# Inserendo l'armonica e semplificando, otteniamo il fattore di amplificazione G
# Per semplicità analizziamo la relazione di fase relativa:
def numerical_dispersion(kh_values, theta_val, courant):
    """
    Calcola il rapporto tra velocità di fase numerica e fisica.
    kh = k * h (numero d'onda normalizzato)
    """
    v_ratio = []
    for kh in kh_values:
        # Relazione di dispersione per il Theta-method (approssimazione al secondo ordine)
        # Derivata dalla sostituzione u = exp(i(jkh - nwt))
        cos_kh = np.cos(kh)
        arg = 1 - (courant**2 * (1 - cos_kh)) / (1 + theta_val * courant**2 * (1 - cos_kh))

        # Estraiamo la frequenza numerica wh
        # wh * dt = arccos(arg)
        if abs(arg) <= 1:
            wh_dt = np.arccos(arg)
            v_num = wh_dt / (courant * kh) # Velocità di fase numerica normalizzata
            v_ratio.append(v_num)
        else:
            v_ratio.append(np.nan) # Instabilità
    return v_ratio

# 2. Parametri del tuo caso
L = 2.0
n_refine = 5
nodes = 2**n_refine
h_val = L / nodes
dt_val = 0.01
c_phys = 1.0 # Velocità nell'equazione delle onde 2D normalizzata
C = c_phys * dt_val / h_val # Numero di Courant

kh_space = np.linspace(0.01, np.pi, 100) # Spettro da onde lunghe a onde corte (limite di Nyquist)

# 3. Grafico
plt.figure(figsize=(10, 6))
plt.plot(kh_space, numerical_dispersion(kh_space, 0.5, C), label=f'Theta=0.5 (C={C:.2f})')
plt.axhline(1.0, color='red', linestyle='--', label='Soluzione Esatta (No dispersione)')

# Evidenziamo la tua soluzione specifica
# u = sin(pi*(x+1)/2) -> k = pi/2
k_target = np.pi / 2
plt.scatter([k_target * h_val], numerical_dispersion([k_target * h_val], 0.5, C),
            color='black', zorder=5, label='Tua Soluzione (k=π/2)')

plt.title("Analisi della Dispersione Numerica (Fase)")
plt.xlabel("Numero d'onda normalizzato (kh)")
plt.ylabel("Velocità di Fase Relativa (v_num / c)")
plt.grid(True)
plt.legend()
plt.savefig('analisi_dispersione.png')
print("Grafico salvato come 'analisi_dispersione.png' nella cartella del progetto.")
