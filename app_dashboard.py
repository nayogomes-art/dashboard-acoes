# app_dashboard.py
# Dashboard interativo - claro por defeito, opção entre "Minimalista" e "Bloomberg" styles
# Suporta dividendos via yfinance, upload CSV e entrada manual com indicação da moeda da compra (USD/EUR)
# Requer: pip install streamlit pandas yfinance plotly openpyxl numpy

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(page_title='Dashboard Ações - Interativo', layout='wide', initial_sidebar_state='expanded')

OUTPUT_DIR = 'output_dashboard'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Helpers ------------------
def sanitize_ticker(t):
    return str(t).strip().replace('.', '-').upper()

@st.cache_data(ttl=3600)
def get_fx_rate_usd_to_eur():
    try:
        fx = yf.Ticker('EURUSD=X')
        hist = fx.history(period='1d')
        if not hist.empty:
            rate = hist['Close'].iloc[-1]  # 1 EUR = rate USD
            return 1.0 / float(rate)  # USD -> EUR
    except Exception:
        pass
    return 1.0

@st.cache_data(ttl=3600)
def get_price_batch(tickers):
    if not tickers:
            return {}
    try:
        df = yf.download(tickers, period='1d', threads=True, progress=False)
    except Exception:
        return {}
    prices = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                last = df[t]['Close'].dropna().iloc[-1]
                prices[t] = float(last)
            except Exception:
                prices[t] = np.nan
    else:
        try:
            last = df['Close'].dropna().iloc[-1]
            prices[tickers[0]] = float(last)
        except Exception:
            prices = {t: np.nan for t in tickers}
    return prices

@st.cache_data(ttl=3600)
def fetch_dividends_series(ticker):
    try:
        tk = yf.Ticker(ticker)
        div = tk.dividends  # series indexed by date
        if div is None:
            return pd.Series(dtype=float)
        return div
    except Exception:
        return pd.Series(dtype=float)

def load_transactions(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception:
        return pd.DataFrame()
    # Accept files with or without header; prefer columns if present
    expected = set(['Data','Ticker','Tipo','Quantidade','Preco','Moeda'])
    if not expected.issubset(set(df.columns)):
        # try mapping first 6 columns
        if df.shape[1] >= 6:
            df = df.iloc[:, :6]
            df.columns = ['Data','Ticker','Tipo','Quantidade','Preco','Moeda']
        else:
            # fallback: if 5 cols assume default USD
            if df.shape[1] == 5:
                df.columns = ['Data','Ticker','Tipo','Quantidade','Preco']
                df['Moeda'] = 'USD'
            else:
                return pd.DataFrame()
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df['Ticker'] = df['Ticker'].apply(sanitize_ticker)
    df['Tipo'] = df['Tipo'].str.upper().str.strip()
    df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce').fillna(0)
    df['Preco'] = pd.to_numeric(df['Preco'], errors='coerce').fillna(0.0)
    df['Moeda'] = df['Moeda'].str.upper().str.strip().replace({'EURO':'EUR','DOLAR':'USD','$':'USD','€':'EUR'})
    df = df.dropna(subset=['Data','Ticker'])
    df = df[df['Quantidade'] > 0]
    df = df[df['Tipo'].isin(['COMPRA','VENDA'])]
    df = df.sort_values('Data').reset_index(drop=True)
    df['Valor_Total'] = df['Quantidade'] * df['Preco']
    return df

def holdings_at_date(df_trans, ticker, as_of_date):
    sub = df_trans[df_trans['Ticker'] == ticker].copy().sort_values('Data')
    qty = 0.0
    for _, row in sub.iterrows():
        if row['Data'] <= as_of_date:
            if row['Tipo'] == 'COMPRA':
                qty += row['Quantidade']
            else:
                qty -= row['Quantidade']
        else:
            break
    return max(qty, 0.0)

def compute_dividends_received(df_trans, display_currency, fx_rate):
    tickers = df_trans['Ticker'].unique()
    div_received = {}
    for t in tickers:
        series = fetch_dividends_series(t)
        if series.empty:
            div_received[t] = 0.0
            continue
        total = 0.0
        for pay_date, div in series.items():
            held = holdings_at_date(df_trans, t, pay_date)
            if held > 0:
                total += float(div) * held  # dividend amounts come in USD for US stocks typically
        # Convert dividend amounts from USD to display_currency if needed
        if display_currency == 'EUR':
            total *= fx_rate
        div_received[t] = round(total, 4)
    return div_received

def calculate_positions_fifo(df_trans, display_currency, fx_rate):
    tickers = df_trans['Ticker'].unique()
    rows = []
    for t in tickers:
        sub = df_trans[df_trans['Ticker'] == t].copy().sort_values('Data')
        buys = []
        for _, r in sub.iterrows():
            qty = r['Quantidade']
            price = r['Preco']
            moeda = r['Moeda'] if 'Moeda' in r.index else 'USD'
            # convert price to display currency unit price
            if moeda == 'USD' and display_currency == 'EUR':
                price_conv = price * fx_rate
            elif moeda == 'EUR' and display_currency == 'USD':
                price_conv = price / fx_rate if fx_rate != 0 else price
            else:
                price_conv = price
            if r['Tipo'] == 'COMPRA':
                buys.append([qty, price_conv])
            else:
                tosell = qty
                while tosell > 0 and buys:
                    if buys[0][0] > tosell:
                        buys[0][0] -= tosell
                        tosell = 0
                    else:
                        tosell -= buys[0][0]
                        buys.pop(0)
        qty_total = sum([b[0] for b in buys])
        custo_total = sum([b[0]*b[1] for b in buys])
        if qty_total > 0:
            rows.append({'Ticker': t, 'Quantidade_Total': qty_total, 'Custo_Total': round(custo_total,2), 'Preco_Medio': round(custo_total/qty_total,4)})
    return pd.DataFrame(rows)

def calculate_performance(positions, current_prices, divs, display_currency, fx_rate):
    rows = []
    total_investido = positions['Custo_Total'].sum() if not positions.empty else 0.0
    total_atual = 0.0
    total_ganhos = 0.0
    for _, r in positions.iterrows():
        t = r['Ticker']
        qty = r['Quantidade_Total']
        custo = r['Custo_Total']
        price_usd = current_prices.get(t, np.nan)
        if np.isnan(price_usd):
            preco_atual = np.nan
        else:
            preco_atual = price_usd * fx_rate if display_currency == 'EUR' else price_usd
        valor_atual = round(qty * preco_atual, 2) if not np.isnan(preco_atual) else np.nan
        divs_t = divs.get(t, 0.0)
        ganho = (valor_atual - custo) + divs_t if not np.isnan(valor_atual) else np.nan
        rent = (ganho / custo * 100) if custo > 0 and not np.isnan(ganho) else np.nan
        if not np.isnan(valor_atual):
            total_atual += valor_atual
            total_ganhos += (valor_atual - custo) + divs_t
        rows.append({'Ticker': t, 'Quantidade': qty, 'Preco_Medio': r['Preco_Medio'], 'Preco_Atual': round(preco_atual,4) if not np.isnan(preco_atual) else np.nan, 'Valor_Atual': valor_atual, 'Custo_Total': round(custo,2), 'Dividendos_Recebidos': round(divs_t,2), 'Ganho/Perda_Total': round(ganho,2) if not np.isnan(ganho) else np.nan, 'Rentabilidade (%)': round(rent,2) if not np.isnan(rent) else np.nan})
    resumo = {'Total_Investido': round(total_investido,2), 'Valor_Atual_Portfolio': round(total_atual,2), 'Ganhos_Perdas_Totais': round(total_ganhos,2), 'Rentabilidade_Total (%)': round((total_ganhos/total_investido*100) if total_investido>0 else 0.0,2)}
    return pd.DataFrame(rows), resumo

# ------------------ UI ------------------
st.title("📊 Dashboard Interativo de Ações — Claro (alternar design)")
with st.sidebar:
    st.header("Configuração")
    display_currency = st.radio("Mostrar valores em:", ("USD", "EUR"), index=0)
    style_choice = st.selectbox("Escolher design:", ("Minimalista", "Bloomberg"))
    st.markdown("---")
    uploaded = st.file_uploader("Carrega um CSV com transacções (Data,Ticker,Tipo,Quantidade,Preco,Moeda)", type=['csv'])
    example_btn = st.button("Carregar ficheiro de exemplo")
    st.markdown("---")
    if st.button("Guardar relatórios (output_dashboard)"):
        st.session_state['save_reports'] = True

# Example file path (created alongside app)
EXAMPLE_CSV = 'transacoes_exemplo_with_currency.csv'

# Load data
if uploaded is not None:
    df_trans = load_transactions(uploaded)
elif example_btn:
    if os.path.exists(EXAMPLE_CSV):
        df_trans = load_transactions(EXAMPLE_CSV)
    else:
        df_trans = pd.DataFrame()
else:
    df_trans = pd.DataFrame()

# Manual add form (with currency selection)
with st.expander("➕ Adicionar transacção manualmente"):
    c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,1,1,1])
    date_in = c1.date_input("Data", value=datetime.today())
    tk = c2.text_input("Ticker", value="AAPL")
    typ = c3.selectbox("Tipo", ("COMPRA","VENDA"))
    qtd = c4.number_input("Quantidade", min_value=0.0, value=1.0)
    preco = c5.number_input("Preço por ação", min_value=0.0, value=100.0)
    moeda = c6.selectbox("Moeda da compra", ("USD","EUR"))
    if st.button("Adicionar transacção"):
        new = pd.DataFrame([{'Data': pd.to_datetime(date_in), 'Ticker': sanitize_ticker(tk), 'Tipo': typ, 'Quantidade': qtd, 'Preco': preco, 'Moeda': moeda, 'Valor_Total': qtd*preco}])
        if df_trans.empty:
            df_trans = new
        else:
            df_trans = pd.concat([df_trans, new], ignore_index=True)
        st.success("Transacção adicionada na sessão (não persistida no CSV automaticamente).")

# Display transactions
if not df_trans.empty:
    st.subheader("📑 Histórico de Transacções")
    st.dataframe(df_trans)

    # FX rate and prices
    fx_rate = get_fx_rate_usd_to_eur()  # USD -> EUR
    prices = get_price_batch(list(df_trans['Ticker'].unique()))
    divs = compute_dividends_received(df_trans, display_currency, fx_rate)

    # Positions (convert per-transaction currency to display currency)
    positions = calculate_positions_fifo(df_trans, display_currency, fx_rate)
    if positions.empty:
        st.warning("Sem posições abertas.")
    else:
        perf_df, resumo = calculate_performance(positions, prices, divs, display_currency, fx_rate)

        # Top metrics cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Investido ("+display_currency+")", f"{resumo['Total_Investido']:,.2f}")
        col2.metric("Valor Atual ("+display_currency+")", f"{resumo['Valor_Atual_Portfolio']:,.2f}")
        col3.metric("Ganhos/Perdas Totais ("+display_currency+")", f"{resumo['Ganhos_Perdas_Totais']:,.2f}")
        col4.metric("Rentabilidade Total (%)", f"{resumo['Rentabilidade_Total (%)']:.2f} %")

        # Interactive plots (Plotly)
        st.subheader("📈 Alocação & Valor por Ação")
        fig_alloc = px.pie(perf_df, names='Ticker', values='Valor_Atual', title='Alocação do Portfólio')
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.subheader("📊 Rentabilidade por Ação")
        fig_bar = px.bar(perf_df.sort_values('Rentabilidade (%)', ascending=False), x='Ticker', y='Rentabilidade (%)', color='Rentabilidade (%)', title='Rentabilidade (%) por Ação', text='Rentabilidade (%)')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("📉 Detalhes por Ticker")
        st.dataframe(perf_df.style.format({"Preco_Atual": "{:.2f}", "Valor_Atual": "{:,.2f}", "Custo_Total":"{:,.2f}", "Dividendos_Recebidos":"{:,.2f}", "Ganho/Perda_Total":"{:,.2f}"}))

        # Export options
        csv_perf = perf_df.to_csv(index=False).encode('utf-8')
        csv_pos = positions.to_csv(index=False).encode('utf-8')
        st.download_button("Descarregar performance.csv", csv_perf, file_name="performance.csv")
        st.download_button("Descarregar posicoes.csv", csv_pos, file_name="posicoes.csv")

        if st.session_state.get('save_reports', False):
            perf_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_app.csv'), index=False)
            positions.to_csv(os.path.join(OUTPUT_DIR, 'posicoes_app.csv'), index=False)
            st.success(f"Relatórios guardados em {OUTPUT_DIR}")
            st.session_state['save_reports'] = False

        # Styling switch (minimal vs bloomberg) - simple CSS tweaks
        if style_choice == "Minimalista":
            st.markdown("""
                <style>
                .main { background-color: #ffffff; color: #0b0b0b; }
                .stMetric { border: none; }
                </style>
                """, unsafe_allow_html=True)
        else:
            # Bloomberg-like: light gray panels and more compact
            st.markdown("""
                <style>
                .main { background-color: #f4f6f8; color: #0b0b0b; }
                .stDataFrame { background-color: #ffffff; border-radius: 8px; padding: 8px; }
                </style>
                """, unsafe_allow_html=True)

else:
    st.info("Carrega um CSV de transacções ou adiciona manualmente para começar.")

st.markdown("---")
st.caption("App criado conforme pedido — tema claro por defeito. Podes alternar o design via sidebar.") 
