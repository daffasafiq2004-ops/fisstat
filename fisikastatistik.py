import streamlit as st
import numpy as np
import math
from math import lgamma as gammaln
import pandas as pd

# =================================================================
# FUNGSI PERHITUNGAN MIKROSTATE (OMEGA)
# =================================================================

# Semua fungsi perhitungan menggunakan Logaritma (ln) untuk stabilitas numerik
@st.cache_data
def hitung_ln_omega_mb(N, ni, gi):
    """Menghitung ln(Omega) untuk Maxwell-Boltzmann (MB)."""
    ln_N_factorial = N * math.log(N) - N
    ln_sum_ni_factorial = sum(n * math.log(n) - n for n in ni if n > 0)
    return ln_N_factorial - ln_sum_ni_factorial

@st.cache_data
def hitung_ln_omega_be(ni, gi):
    """Menghitung ln(Omega) untuk Bose-Einstein (BE)."""
    ln_omega = 0
    for n, g in zip(ni, gi):
        if g <= 0: continue
        # Rumus BE: ln(C(n + g - 1, n))
        ln_omega += gammaln(n + g) - gammaln(n + 1) - gammaln(g)
    return ln_omega

@st.cache_data
def hitung_ln_omega_fd(ni, gi):
    """Menghitung ln(Omega) untuk Fermi-Dirac (FD)."""
    ln_omega = 0
    for n, g in zip(ni, gi):
        if g <= 0: continue
        if n > g: continue 
        # Rumus FD: ln(C(g, n))
        ln_omega += gammaln(g + 1) - gammaln(n + 1) - gammaln(g - n + 1)
    return ln_omega

# =================================================================
# APLIKASI STREAMLIT (TAMPILAN V7)
# =================================================================

st.set_page_config(page_title="FisStat Custom Simulator", layout="wide")
st.title('🔬 Simulator Statistik Kuantum Kustom')
st.caption('Menghitung Keadaan Mikro ($\Omega$), Keadaan Makro ($E_{total}$), Entropi ($S$), dan menganalisis Distribusi (MB, BE, FD).')

# Inisialisasi list
ni_list, gi_list, E_list = [], [], []
N_tingkat = 0

# --- SIDEBAR KIRI: KONFIGURASI SISTEM & MODE ANALISIS ---
with st.sidebar:
    st.header('1. Konfigurasi Sistem')
    
    statistik_pilih = st.selectbox(
        'Mode Statistik:',
        ('Maxwell-Boltzmann', 'Bose-Einstein', 'Fermi-Dirac'), index=0
    )
    
    st.markdown("---")
    st.header('2. Mode Analisis')
    mode_dos = st.checkbox('Aktifkan Plot Kerapatan Keadaan (DOS)', value=False)
    mode_otomatis = st.checkbox('Aktifkan Pencarian Makrostate Otomatis', value=False)
    
    st.markdown("---")
    st.subheader('Input Dasar')
    
    tipe_degenerasi = st.radio(
        'Tipe Degenerasi ($g$):',
        ('Sama', 'Per-level')
    )
    
    level_energi_input = st.text_input(
        'Level Energi Unik (pisah koma)', 
        value='0,1,2'
    )
    N_total_input = st.number_input(
        'Jumlah Partikel Total $N$', 
        min_value=1, value=5, step=1
    )

# Parsing Level Energi
try:
    E_list = [float(e.strip()) for e in level_energi_input.split(',') if e.strip()]
    N_tingkat = len(E_list)
except ValueError:
    st.sidebar.error("❌ Format Level Energi salah.")
    E_list = [0.0]
    N_tingkat = 1

# Inisialisasi Degenerasi Sama
g_sama = 1
if tipe_degenerasi == 'Sama':
    g_sama = st.sidebar.number_input('Degenerasi $g$ (nilai tunggal)', min_value=1, value=1)
    gi_list = [g_sama] * N_tingkat
else:
    gi_list = []


# --- KOLOM UTAMA (Input Detail, Output Utama, Output Analisis) ---
col_detail, col_hasil, col_analisis = st.columns([1.5, 2, 2])

# =================================================================
# 1. KOLOM DETAIL INPUT (Pusat Kiri)
# =================================================================
with col_detail:
    st.header('3. Konfigurasi Makro State')
    st.caption(f"Mengisi {N_tingkat} Tingkat Energi.")
    st.divider()

    input_data_dict = {}
    
    # Loop Input Partikel dan Degenerasi
    for i in range(N_tingkat):
        E_val = E_list[i]
        
        st.markdown(f'**Level {i}: $E={E_val}$**')
        c1, c2 = st.columns(2)
        
        # Input Degenerasi Per-Level
        if tipe_degenerasi == 'Per-level':
            with c1:
                g = st.number_input(f'$g_{i}$ (Degenerasi)', key=f'g{i}', min_value=1, value=1)
                gi_list.append(g)
            g_val = gi_list[i]
        else:
            g_val = gi_list[i]
            c1.write(f"Degenerasi $g_{i} = {g_val}$")
        
        # Input Partikel n_i
        with c2:
            n = st.number_input(f'$n_{i}$ (Partikel)', key=f'n{i}', min_value=0, value=1)
            ni_list.append(n)

        input_data_dict[f'Level {i}'] = [E_val, ni_list[i], g_val]

    st.markdown("---")
    hitung_btn = st.button('Hitung $\Omega$, $S$, & $P$')

# =================================================================
# 2. KOLOM HASIL UTAMA (Pusat Tengah)
# =================================================================
with col_hasil:
    st.header('4. Rumus & Hasil $\Omega$')
    st.divider()

    # Tampilkan Rumus Sesuai Mode Statistik
    st.subheader(f'Rumus {statistik_pilih} ($\ln\Omega$):')
    if 'Boltzmann' in statistik_pilih:
        st.latex(r'''\Omega_{MB} = N! \prod_i \frac{g_i^{n_i}}{n_i!} \quad \rightarrow \quad \Omega_{total} = \Omega_{MB}''')
    elif 'Bose-Einstein' in statistik_pilih:
        st.latex(r'''\Omega_{BE} = \prod_i \frac{(n_i + g_i - 1)!}{n_i! (g_i - 1)!}''')
    elif 'Fermi-Dirac' in statistik_pilih:
        st.latex(r'''\Omega_{FD} = \prod_i \frac{g_i!}{n_i! (g_i - n_i)!}''')
    
    st.markdown("---")
    
    # OUTPUT HASIL
    if hitung_btn:
        # --- Validasi & Perhitungan ---
        if sum(ni_list) != N_total_input:
            st.error(f"❌ **Validasi Gagal:** $N$ ({N_total_input}) $\neq \sum n_i$ ({sum(ni_list)}).")
        elif 'Fermi-Dirac' in statistik_pilih and any(n > g for n, g in zip(ni_list, gi_list)):
             st.error("❌ **FD Error:** $n_i > g_i$ melanggar Prinsip Pauli.")
        else:
            ln_Omega = 0.0
            if 'Boltzmann' in statistik_pilih:
                ln_Omega = hitung_ln_omega_mb(N_total_input, ni_list, gi_list)
            elif 'Bose-Einstein' in statistik_pilih:
                ln_Omega = hitung_ln_omega_be(ni_list, gi_list)
            elif 'Fermi-Dirac' in statistik_pilih:
                ln_Omega = hitung_ln_omega_fd(ni_list, gi_list)

            # Output Metrik
            Omega_macro = math.exp(ln_Omega) if ln_Omega < 700 else float('inf')
            
            st.metric(label="Logaritma Mikro State ($\ln\Omega$)", value=f"{ln_Omega:.6f}")
            st.metric(label="Entropi ($S/k_B$)", value=f"{ln_Omega:.6f}")
            
            if Omega_macro != float('inf'):
                 st.write(f"$\Omega_{{macro}}$ Absolut: **{Omega_macro:,.0f}**")
            else:
                 st.write(f"$\Omega_{{macro}}$ Absolut: **Terlalu Besar** ($e^{{ {ln_Omega:.2f} }}$)")


# =================================================================
# 3. KOLOM ANALISIS (Pusat Kanan)
# =================================================================
with col_analisis:
    st.header('5. Analisis Termodinamika')
    st.divider()
    
    # Tampilkan Tabel Input
    st.subheader('Ringkasan Makro State')
    df_input = pd.DataFrame.from_dict(input_data_dict, orient='index', columns=['Energi $E_i$', 'Partikel $n_i$', 'Degenerasi $g_i$'])
    st.dataframe(df_input)

    # Output Metrik Termodinamika
    if hitung_btn and sum(ni_list) == N_total_input:
        Energi_Total = sum(E * n for E, n in zip(E_list, ni_list))
        
        st.metric(label="Total Partikel ($N$)", value=N_total_input)
        st.metric(label="Energi Total ($E_{total}$)", value=f"{Energi_Total:.2f} unit")
        
        st.markdown("---")
        st.subheader("Probabilitas ($P$)")
        st.latex(r'''P = \frac{\Omega_{macro}}{\Omega_{total}}''')
        
        st.warning("Probabilitas $P$ di Mode Umum adalah **Probabilitas Relatif** ($\Omega_{macro}$)")
        st.info("Nilai $\Omega_{total}$ (Denominator) membutuhkan Mode Kerapatan Keadaan.")