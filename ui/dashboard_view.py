# ui/dashboard_view.py

import streamlit as st
import plotly.graph_objects as go

def display():
    """
    顯示「本月顧客分析」儀表板的所有 UI 元素。
    此函式直接從 1013_test.py 中的 show_monthly_analysis() 函式修改而來。
    """
    st.subheader("📈 本月營運儀表板")

    # --- 模擬數據 ---
    # 在實際應用中，這些數據應該從資料庫或 API 獲取
    new_customers = 1247
    satisfaction_rate = 87.5
    items = ["套餐A", "套餐B", "套餐C", "飲料", "甜點"]
    revenue = [45000, 38000, 52000, 28000, 15000]
    quantity = [150, 120, 180, 280, 95]

    # --- 頂部關鍵指標 (KPIs) ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 16px;'>本月新來客數</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 52px; font-weight: bold;'>{new_customers:,}</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0;'>較上月 +12.3%</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 30px; border-radius: 10px; text-align: center;'>
            <h3 style='color: white; margin: 0; font-size: 16px;'>本月顧客滿意度</h3>
            <h1 style='color: white; margin: 10px 0; font-size: 52px; font-weight: bold;'>{satisfaction_rate}%</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0;'>較上月 +3.2%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 互動式圖表 (Plotly) ---
    fig = go.Figure()
    # 長條圖 (銷售數量)，使用次要 Y 軸
    fig.add_trace(go.Bar(
        x=items, y=quantity, name='銷售數量',
        marker_color='rgba(99, 110, 250, 0.7)', yaxis='y2'
    ))
    # 折線圖 (營收)，使用主要 Y 軸
    fig.add_trace(go.Scatter(
        x=items, y=revenue, name='營收 (元)',
        mode='lines+markers', line=dict(width=3, color='rgba(239, 85, 59, 0.9)'), 
        marker=dict(size=10), yaxis='y'
    ))
    
    fig.update_layout(
        title={'text': '本月品項營收 vs. 銷售數量分析', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        xaxis=dict(title='品項'),
        yaxis=dict(title=dict(text='營收 (元)'), side='left'),
        yaxis2=dict(title=dict(text='銷售數量'), overlaying='y', side='right'),
        hovermode='x unified', height=500, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 底部匯總指標 ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總營收", f"NT$ {sum(revenue):,}", "+8.5%")
    with col2:
        st.metric("總銷售數", f"{sum(quantity):,}", "+15.2%")
    with col3:
        # 避免除以零的錯誤
        avg_ticket_price = sum(revenue) // new_customers if new_customers > 0 else 0
        st.metric("平均客單價", f"NT$ {avg_ticket_price}", "+2.1%")
    with col4:
        st.metric("回購率", "42.3%", "+5.7%")