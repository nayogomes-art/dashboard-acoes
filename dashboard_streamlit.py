# dashboard_streamlit.py
# Streamlit app para Dashboard S&P500 com suporte a dividendos
# Requer: pip install yfinance pandas matplotlib numpy streamlit openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.set_page_config(page_title='Dashboard Ações (S&P500)', layout='wide')

OUTPUT_DIR = 'output_dashboard'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CURRENCY = st.sidebar.selectbox('Moeda', ['USD', 'EUR'], index=0)
CURRENCY_SYMBOL = '$' if CURRENCY == 'USD' else '€'

@st.cache_data(ttl=3600)
def get_exchange_rate():
    if CURRENCY == 'USD':
        return 1.0
    try:
        fx = yf.Ticker('EURUSD=X')
        hist = fx.history(period='1d')
        if not hist.empty:
            rate = hist['Close'].iloc[-1]
            return 1.0 / float(rate)
    except Exception:
        pass
    return 1.0

def sanitize_ticker(t):
    return str(t).strip().replace('.', '-').upper()

@st.cache_data(ttl=3600)
def fetch_dividends(ticker):
    try:
        tk = yf.Ticker(ticker)
        div = tk.dividends
        if div is None:
            return pd.Series(dtype=float)
        return div
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(ttl=3600)
def fetch_latest_price(tickers, exchange_rate):
    if not tickers:
        return {}
    try:
        data = yf.download(tickers, period='1d', threads=True, progress=False)
    except Exception:
        return {}
    prices = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                last = data[t]['Close'].dropna().iloc[-1]
                prices[t] = float(last) * (exchange_rate if CURRENCY=='EUR' else 1.0)
            except Exception:
                continue
    else:
        try:
            last = data['Close'].dropna().iloc[-1]
            prices[tickers[0]] = float(last) * (exchange_rate if CURRENCY=='EUR' else 1.0)
        except Exception:
            pass
    return prices

# ------------------
# Leitura/edição de transacções
# ------------------
def load_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        return df
    # Se cabeçalho não tiver os nomes, forçar mapeamento
    if not set(['Data','Ticker','Tipo','Quantidade','Preco']).issubset(df.columns):
        df.columns = ['Data','Ticker','Tipo','Quantidade','Preco']
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df['Ticker'] = df['Ticker'].apply(sanitize_ticker)
    df['Tipo'] = df['Tipo'].str.upper().str.strip()
    df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce').fillna(0)
    df['Preco'] = pd.to_numeric(df['Preco'], errors='coerce').fillna(0.0)
    df = df.dropna(subset=['Data','Ticker'])
    df = df[df['Quantidade'] > 0]
    df = df[df['Tipo'].isin(['COMPRA','VENDA'])]
    df = df.sort_values('Data').reset_index(drop=True)
    df['Valor_Total'] = df['Quantidade'] * df['Preco']
    return df

# FIFO calculation to determine holdings at arbitrary date (for dividend allocation)
def holdings_at_date(df_trans, ticker, as_of_date):
    sub = df_trans[df_trans['Ticker'] == ticker].copy()
    sub = sub.sort_values('Data')
    qty = 0.0
    # process chronologically
    for _, row in sub.iterrows():
        if row['Data'] <= as_of_date:
            if row['Tipo'] == 'COMPRA':
                qty += row['Quantidade']
            else:
                qty -= row['Quantidade']
        else:
            break
    return max(qty, 0.0)

# Compute dividends received per ticker using historical dividend series multiplied by holdings at ex-date
def compute_dividends_received(df_trans, exchange_rate):
    tickers = df_trans['Ticker'].unique()
    div_received = {}
    for t in tickers:
        div_series = fetch_dividends(t)
        if div_series.empty:
            div_received[t] = 0.0
            continue
        total = 0.0
        for pay_date, div_amount in div_series.items():
            # pay_date is Timestamp
            held_qty = holdings_at_date(df_trans, t, pay_date)
            if held_qty > 0:
                total += float(div_amount) * held_qty
        # convert currency if needed
        if CURRENCY == 'EUR':
            total *= exchange_rate
        div_received[t] = round(total, 4)
    return div_received

# Calculate positions via FIFO left-over (similar to earlier script)
def calculate_positions_fifo(df_trans):
    tickers = df_trans['Ticker'].unique()
    rows = []
    for t in tickers:
        sub = df_trans[df_trans['Ticker'] == t].copy()
        buys = []
        for _, r in sub.iterrows():
            if r['Tipo'] == 'COMPRA':
                buys.append([r['Quantidade'], r['Preco']])
            else:
                qty = r['Quantidade']
                while qty > 0 and buys:
                    if buys[0][0] > qty:
                        buys[0][0] -= qty
                        qty = 0
                    else:
                        qty -= buys[0][0]
                        buys.pop(0)
        qty_total = sum([b[0] for b in buys])
        custo_total = sum([b[0]*b[1] for b in buys])
        if qty_total > 0:
            rows.append({'Ticker': t, 'Quantidade_Total': qty_total, 'Custo_Total': round(custo_total,2), 'Preco_Medio': round(custo_total/qty_total,4)})
    return pd.DataFrame(rows)

# Performance calc including dividends
def calculate_performance(positions, current_prices, div_received):
    rows = []
    total_investido = positions['Custo_Total'].sum() if not positions.empty else 0.0
    total_atual = 0.0
    total_ganhos = 0.0
    for _, r in positions.iterrows():
        t = r['Ticker']
        qty = r['Quantidade_Total']
        custo = r['Custo_Total']
        preco_atual = current_prices.get(t, np.nan)
        valor_atual = qty * preco_atual if not np.isnan(preco_atual) else np.nan
        divs = div_received.get(t, 0.0)
        ganho = (valor_atual - custo) + divs if not np.isnan(valor_atual) else np.nan
        rent = (ganho / custo * 100) if custo > 0 and not np.isnan(ganho) else np.nan
        if not np.isnan(valor_atual):
            total_atual += valor_atual
            total_ganhos += (valor_atual - custo)
            total_ganhos += divs
        rows.append({'Ticker': t, 'Quantidade': qty, 'Preco_Medio': r['Preco_Medio'], 'Preco_Atual': round(preco_atual,4) if not np.isnan(preco_atual) else np.nan, 'Valor_Atual': round(valor_atual,2) if not np.isnan(valor_atual) else np.nan, 'Custo_Total': round(custo,2), 'Dividendos_Recebidos': round(divs,2), 'Ganho/Perda_Total': round(ganho,2) if not np.isnan(ganho) else np.nan, 'Rentabilidade (%)': round(rent,2) if not np.isnan(rent) else np.nan})
    resumo = {'Total_Investido': round(total_investido,2), 'Valor_Atual_Portfolio': round(total_atual,2), 'Ganhos_Perdas_Totais': round(total_ganhos,2), 'Rentabilidade_Total (%)': round((total_ganhos/total_investido*100) if total_investido>0 else 0.0,2)}
    return pd.DataFrame(rows), resumo

# Plotting helpers
def plot_positions(perf_df):
    fig, ax = plt.subplots(figsize=(10,6))
    df = perf_df.dropna(subset=['Valor_Atual'])
    ax.bar(df['Ticker'], df['Valor_Atual'])
    ax.set_xlabel('Ticker')
    ax.set_ylabel(f'Valor Atual ({CURRENCY_SYMBOL})')
    ax.set_title('Valor Atual das Posições')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_rentabilidade(perf_df):
    fig, ax = plt.subplots(figsize=(10,6))
    df = perf_df.dropna(subset=['Rentabilidade (%)'])
    ax.bar(df['Ticker'], df['Rentabilidade (%)'])
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Rentabilidade (%)')
    ax.set_title('Rentabilidade por Ação')
    plt.axhline(0, linestyle='--', linewidth=0.8)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ------------------
# UI
# ------------------

st.title('Dashboard Interactivo de Ações (S&P 500)')

with st.sidebar.expander('Carregar / Exemplo'):
    uploaded = st.file_uploader('Carrega um ficheiro CSV com transacções (Data,Ticker,Tipo,Quantidade,Preco)', type=['csv'])
    use_example = st.button('Usar ficheiro de exemplo incluído')

if uploaded is not None:
    df_trans = load_csv(uploaded)
elif use_example:
    example_path = 'transacoes_exemplo.csv'
    if os.path.exists(example_path):
        df_trans = load_csv(example_path)
    else:
        st.error('Ficheiro de exemplo não encontrado no diretório da app.')
        df_trans = pd.DataFrame()
else:
    st.info('Carrega um CSV ou carrega o exemplo para começar.')
    df_trans = pd.DataFrame()

# Manual add form
with st.expander('Adicionar transacção manualmente'):
    col1, col2, col3, col4, col5 = st.columns(5)
    d = col1.date_input('Data', value=datetime.today())
    tk = sanitize_ticker(col2.text_input('Ticker', value='AAPL'))
    tp = col3.selectbox('Tipo', ['COMPRA','VENDA'])
    q = col4.number_input('Quantidade', min_value=0.0, value=1.0)
    p = col5.number_input('Preço por ação', min_value=0.0, value=1.0)
    if st.button('Adicionar'):
        new = pd.DataFrame([{'Data': pd.to_datetime(d), 'Ticker': tk, 'Tipo': tp, 'Quantidade': q, 'Preco': p, 'Valor_Total': q*p}])
        if df_trans.empty:
            df_trans = new
        else:
            df_trans = pd.concat([df_trans, new], ignore_index=True)
        st.success('Transacção adicionada (temporariamente na sessão).')

if not df_trans.empty:
    st.subheader('Histórico de Transacções')
    st.dataframe(df_trans)

    # Cálculos
    exchange_rate = get_exchange_rate()
    positions = calculate_positions_fifo(df_trans)
    if positions.empty:
        st.warning('Nenhuma posição aberta encontrada.')
    else:
        st.subheader('Posições (FIFO)')
        st.dataframe(positions)

        tickers = positions['Ticker'].tolist()
        with st.spinner('A obter preços atuais e dividendos...'):
            current_prices = fetch_latest_price(tickers, exchange_rate)
            divs = compute_dividends_received(df_trans, exchange_rate)
            perf_df, resumo = calculate_performance(positions, current_prices, divs)

        st.subheader('Performance Detalhada (inclui dividendos recebidos)')
        st.dataframe(perf_df)

        st.subheader('Resumo Global')
        st.metric('Total Investido', f"{CURRENCY_SYMBOL}{resumo['Total_Investido']:,.2f}")
        st.metric('Valor Atual do Portfolio', f"{CURRENCY_SYMBOL}{resumo['Valor_Atual_Portfolio']:,.2f}")
        st.metric('Ganhos/Perdas Totais', f"{CURRENCY_SYMBOL}{resumo['Ganhos_Perdas_Totais']:,.2f}")
        st.metric('Rentabilidade Total (%)', f"{resumo['Rentabilidade_Total (%)']:.2f}%")

        st.markdown('**Gráficos**')
        plot_positions(perf_df)
        plot_rentabilidade(perf_df)

        # Export
        if st.button('Guardar relatórios para pasta output_dashboard'):
            perf_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_streamlit.csv'), index=False)
            positions.to_csv(os.path.join(OUTPUT_DIR, 'posicoes_streamlit.csv'), index=False)
            st.success(f'Relatórios guardados em {OUTPUT_DIR}')

        # Download buttons
        st.download_button('Descarregar performance.csv', perf_df.to_csv(index=False).encode('utf-8'), file_name='performance.csv')
        st.download_button('Descarregar posicoes.csv', positions.to_csv(index=False).encode('utf-8'), file_name='posicoes.csv')

else:
    st.info('Sem transacções carregadas.')

st.markdown('---')
st.caption('App gerado automaticamente — reporta bugs ou pede melhorias.')
