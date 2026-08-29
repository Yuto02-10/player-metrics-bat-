import streamlit as st
import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px  # ヒートマップ描画用ライブラリ

st.set_page_config(page_title="配球アシスタントAI", layout="wide")
st.title("配球アシスタントAI（期間比較モード）")

@st.cache_data
def load_all_data():
    file_paths = glob.glob('試合データ/*.csv')
    if not file_paths:
        return None
        
    df_list = []
    for path in file_paths:
        df_each = pd.read_csv(path).dropna(subset=['PitchType', 'PitchLocation'])
        df_each['Date'] = pd.to_datetime(df_each['Date'], errors='coerce')
        df_each = df_each.dropna(subset=['Date'])
        
        df_each['PitchLocation'] = pd.to_numeric(df_each['PitchLocation'], errors='coerce')

        if 'Batter' in df_each.columns:
            df_each['Batter'] = df_each['Batter'].str.replace(r'\s+', '', regex=True)
            
        if 'Pitcher' in df_each.columns:
            df_each['Pitcher'] = df_each['Pitcher'].str.replace(r'\s+', '', regex=True)
            
        df_list.append(df_each)
        
    return pd.concat(df_list, ignore_index=True)

df_raw = load_all_data()

if df_raw is None:
    st.warning("「試合データ/」フォルダにCSVファイルを追加してください。")
else:
    # ------------------------------------
    # サイドバー：期間を2つ設定
    # ------------------------------------
    st.sidebar.header("📅 比較する期間の設定")
    
    min_date = df_raw['Date'].min().date()
    max_date = df_raw['Date'].max().date()
    
    st.sidebar.subheader("期間 1")
    date_range_1 = st.sidebar.date_input(
        "期間1を選択", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, key="dr1"
    )
    
    st.sidebar.subheader("期間 2")
    date_range_2 = st.sidebar.date_input(
        "期間2を選択", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, key="dr2"
    )
    
    # データの絞り込み関数
    def filter_by_date(df, date_range):
        if len(date_range) == 2:
            return df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
        return df

    df_filtered_1 = filter_by_date(df_raw, date_range_1)
    df_filtered_2 = filter_by_date(df_raw, date_range_2)
    
    st.sidebar.write(f"球数 - 期間1: {len(df_filtered_1)}球 / 期間2: {len(df_filtered_2)}球")
    
    def assign_weight_advanced(row):
        if row['KorBB'] == '空振り三振': return 2.5
        if row['KorBB'] == '見逃し三振': return 2.5
        if row['KorBB'] == '四球': return -4.0
        if row['PitchResult'] == '死球': return -3.0

        if row['PitchResult'] == '空振り': return 1.5
        if row['PitchResult'] == '見逃し': return 1.3
        if row['PitchResult'] == 'ファウル': return 0.7
        if row['PitchResult'] == 'ボール': return -0.4

        if row['PitchResult'] == 'インプレー':
            hit_type = str(row['HitType'])
            hit_result = str(row['HitResult'])
            catch_position = str(row['Catch'])
            infielders = ['投手', '捕手', '一塁手', '二塁手', '三塁手', '遊撃手']
            
            if hit_result == 'アウト' or hit_result == 'nan':
                if hit_type == 'フライ': return 2.5 if catch_position in infielders else 1.8
                elif hit_type == 'ゴロ': return 2.0
                elif hit_type == 'ライナー': return 1.5
                else: return 0.5
            
            weight_matrix = {
                ('ゴロ', '単打'): -2.8, ('ライナー', '単打'): -5.5, ('フライ', '単打'): -4.7,
                ('ライナー', '二塁打'): -8.0, ('フライ', '二塁打'): -9.5, ('ゴロ', '二塁打'): -6.3,
                ('フライ', '三塁打'): -11.0, ('ライナー', '三塁打'): -9.5, ('ゴロ', '三塁打'): -8.0,
                ('フライ', '本塁打'): -16.0, ('ゴロ', '本塁打'): -4.5, ('ライナー', '本塁打'): -20.0,
                ('ゴロ', 'エラー'): 2.5, ('フライ', 'エラー'): 1.0, ('ライナー', 'エラー'): 1.5,             
            }
            return weight_matrix.get((hit_type, hit_result), 0.0)

        return 0.0

    @st.cache_resource(show_spinner="AIモデルを学習中...（初回のみ時間がかかります）")
    def train_model(df_input):
        if len(df_input) < 10:
            return None # データが少なすぎる場合は学習しない
            
        df_work = df_input.copy()
        df_work['PitchScore'] = df_work.apply(assign_weight_advanced, axis=1)
        
        adjacent_map = {
            1.0: [2.0, 4.0, 5.0, 11.0], 2.0: [1.0, 3.0, 5.0, 11.0, 12.0],
            3.0: [2.0, 5.0, 6.0, 10.0, 12.0], 4.0: [1.0, 2.0, 5.0, 7.0, 8.0, 11.0],
            5.0: [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0], 6.0: [2.0, 3.0, 5.0, 8.0, 9.0, 12.0],
            7.0: [4.0, 5.0, 8.0, 11.0, 13.0], 8.0: [4.0, 5.0, 6.0, 7.0, 9.0, 13.0],
            9.0: [5.0, 6.0, 8.0, 12.0, 13.0], 11.0: [1.0, 2.0, 3.0],
            12.0: [1.0, 4.0, 7.0], 13.0: [3.0, 6.0, 9.0], 14.0: [7.0, 8.0, 9.0]
        }
        discount_rate = 0.3
        
        records = df_work.to_dict('records')
        augmented_rows = []
        
        for row in records:
            augmented_rows.append(row)
            loc = row.get('PitchLocation')
            if pd.notna(loc) and loc in adjacent_map:
                for adj_loc in adjacent_map[loc]:
                    new_row = row.copy()
                    new_row['PitchLocation'] = adj_loc
                    new_row['PitchScore'] = row['PitchScore'] * discount_rate
                    augmented_rows.append(new_row)

        df_train = pd.DataFrame(augmented_rows)
        features = ['Ball', 'Strike', 'PitcherLR', 'Batter', 'PitchType', 'PitchLocation']
        X = df_train[features].copy()
        y = df_train['PitchScore']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['PitcherLR', 'Batter', 'PitchType'])
            ], remainder='passthrough'
        )
        
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
        ])
        
        model.fit(X, y)
        return model 

    # 期間1、期間2のモデルをそれぞれ学習
    model_1 = train_model(df_filtered_1)
    model_2 = train_model(df_filtered_2)
    
    features = ['Ball', 'Strike', 'PitcherLR', 'Batter', 'PitchType', 'PitchLocation']
    
    # 左右でヒートマップの形を揃えるために、データ全体から球種・コースのマスターリストを作成
    global_pitch_types = df_raw['PitchType'].dropna().unique()
    global_pitch_locations = sorted(df_raw['PitchLocation'].dropna().unique())
    
    # --- 予測UI ---
    st.sidebar.header("🎯 配球シミュレーション設定")
    
    # 選択可能な打者を期間1・期間2のデータから統合
    batter_list = pd.concat([df_filtered_1['Batter'], df_filtered_2['Batter']]).dropna().unique()
    target_batters = st.sidebar.multiselect("対象打者を選択（複数可）", batter_list)
    
    if not target_batters:
        st.warning("打者を1人以上選択してください。")
        st.stop()
    
    col_sb1, col_sb2 = st.sidebar.columns(2)
    with col_sb1:
        c_ball = st.slider("ボール", 0, 3, 0)
    with col_sb2:
        c_strike = st.slider("ストライク", 0, 2, 0)
        
    p_lr = st.sidebar.radio("投手の左右", ["右", "左"])
    
    if st.sidebar.button("AI配球予測を開始", use_container_width=True):
        
        # ペナルティ設定
        breaking_balls = ['スライダー', 'フォーク', 'カーブ', 'チェンジアップ', 'スプリット', 'シンカー', 'カットボール']
        penalty_map = {1.0: 2.0, 2.0: 3.5, 3.0: 2.0, 5.0: 1.5, 11.0: 1.0}
        
        # 各期間ごとに予測と描画を行う関数
        def predict_and_display(model, period_name):
            if model is None:
                st.error(f"{period_name} のデータ量が不足しているため予測できません。")
                return

            situation = {
                'Ball': c_ball, 'Strike': c_strike,
                'PitcherLR': p_lr, 'Batter': target_batter
            }
            
            candidates = []
            for pt in global_pitch_types:
                for pl in global_pitch_locations:
                    row = situation.copy()
                    row['PitchType'] = pt
                    row['PitchLocation'] = pl
                    candidates.append(row)
                    
            X_test = pd.DataFrame(candidates)[features]
            expected_scores = model.predict(X_test)
            
            results = pd.DataFrame({
                '球種': X_test['PitchType'], 
                'コース': X_test['PitchLocation'],
                'AI推奨度': expected_scores
            })

            def apply_risk_penalty(row):
                score = row['AI推奨度']
                if row['球種'] in breaking_balls and row['コース'] in penalty_map:
                    score -= penalty_map[row['コース']]
                return score

            results['AI推奨度'] = results.apply(apply_risk_penalty, axis=1)
            results_sorted = results.sort_values(by='AI推奨度', ascending=False)
            
            # --- 結果の表示 ---
            st.markdown(f"#### 📅 {period_name} のAI推奨")
            st.dataframe(results_sorted.head(5).reset_index(drop=True), use_container_width=True)
            
            st.markdown("##### 📊 球種×コース ヒートマップ")
            pivot_recommend = results.pivot(index='球種', columns='コース', values='AI推奨度')
            
            fig = px.imshow(
                pivot_recommend,
                labels=dict(x="コース番号", y="球種", color="推奨度"),
                color_continuous_scale="RdBu_r", # 赤が高推奨、青が低推奨
                aspect="auto"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        # 打者ごとにUIを構築
        for target_batter in target_batters:
            st.subheader(f"🎯 {target_batter} 選手への推奨配球 期間比較")
            
            # 画面を左右に分割
            col1, col2 = st.columns(2)
            
            with col1:
                predict_and_display(model_1, "期間 1")
                
            with col2:
                predict_and_display(model_2, "期間 2")
                
            st.markdown("---")
