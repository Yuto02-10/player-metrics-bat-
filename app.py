import streamlit as st
import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="配球アシスタントAI", layout="wide")
st.title("配球アシスタントAI")

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
    st.sidebar.header("📅 データの期間絞り込み")
    
    min_date = df_raw['Date'].min().date()
    max_date = df_raw['Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "分析対象にする期間を選択",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_raw[
            (df_raw['Date'].dt.date >= start_date) & 
            (df_raw['Date'].dt.date <= end_date)
        ]
    else:
        df_filtered = df_raw
        
    st.sidebar.write(f"現在の対象球数: {len(df_filtered)} 球")
    
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
                if hit_type == 'フライ':
                    return 2.5 if catch_position in infielders else 1.8
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

    if len(df_filtered) < 10:
        st.error("選択された期間のデータが少なすぎます。期間を広げてください。")
    else:
        @st.cache_resource(show_spinner="AIモデルを学習中...（初回のみ時間がかかります）")
        def train_model(df_input):
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
            
            categorical_cols = ['PitcherLR', 'Batter', 'PitchType']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
                ],
                remainder='passthrough'
            )
            
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(random_state=42, n_estimators=100))
            ])
            
            model.fit(X, y)
            
            return model 

        model = train_model(df_filtered)
        features = ['Ball', 'Strike', 'PitcherLR', 'Batter', 'PitchType', 'PitchLocation']
        
        # --- 予測UI ---
        st.sidebar.header("🎯 配球シミュレーション設定")
        
        batter_list = df_filtered['Batter'].dropna().unique()
        target_batters = st.sidebar.multiselect("対象打者を選択（複数可）", batter_list)
        
        if not target_batters:
            st.warning("打者を1人以上選択してください。")
            st.stop()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            c_ball = st.slider("ボール", 0, 3, 0)
        with col2:
            c_strike = st.slider("ストライク", 0, 2, 0)
            
        p_lr = st.sidebar.radio("投手の左右", ["右", "左"])
        
        if st.sidebar.button("AI配球予測を開始", use_container_width=True):
            pitch_types = df_filtered['PitchType'].dropna().unique()
            pitch_locations = df_filtered['PitchLocation'].dropna().unique()
            
            # ==========================================
            # 【追加・修正】ソースコード内でペナルティを設定
            # ==========================================
            # 1. ペナルティの対象となる変化球を指定
            breaking_balls = ['スライダー', 'フォーク', 'カーブ', 'チェンジアップ', 'スプリット', 'シンカー', 'カットボール']
            
            # 2. コースごとのペナルティ値（減点値）を指定
            # 左側の数字がコース番号、右側の数字がマイナスするスコアの大きさ
            penalty_map = {
                1.0: 1.0,   # イン高めストライク（右投手vs右打者の場合など）
                2.0: 1.1,   # ど真ん中高めストライク（最も危険なのでペナルティ大）
                3.0: 0.8,   # アウト高めストライク
                5.0: 1.3,   # ど真ん中ストライク
                4.0: 0.2,
                6.0: 0.15,
                11.0: 0.1,  # 高めのボールゾーン（すっぽ抜け）
            }
            
            for target_batter in target_batters:
                situation = {
                    'Ball': c_ball, 'Strike': c_strike,
                    'PitcherLR': p_lr, 'Batter': target_batter
                }
                
                candidates = []
                for pt in pitch_types:
                    for pl in pitch_locations:
                        row = situation.copy()
                        row['PitchType'] = pt
                        row['PitchLocation'] = pl
                        candidates.append(row)
                        
                X_test = pd.DataFrame(candidates)[features]
                
                expected_scores = model.predict(X_test)
                
                results = pd.DataFrame({
                    '球種': X_test['PitchType'], 
                    'コース': X_test['PitchLocation'],
                    'AI推奨度(期待値)': expected_scores
                })

                # ------------------------------------
                # 【追加】算出された期待値にペナルティを適用する関数
                # ------------------------------------
                def apply_risk_penalty(row):
                    score = row['AI推奨度(期待値)']
                    p_type = row['球種']
                    loc = row['コース']
                    
                    # 対象の変化球であり、かつpenalty_mapに登録されているコースなら減点
                    if p_type in breaking_balls and loc in penalty_map:
                        score -= penalty_map[loc]
                        
                    return score

                # 関数を適用してスコアを更新し、ソートし直す
                results['AI推奨度(期待値)'] = results.apply(apply_risk_penalty, axis=1)
                results = results.sort_values(by='AI推奨度(期待値)', ascending=False)
                
                st.subheader(f"🎯 {target_batter} 選手への推奨配球 Top 5")
                st.dataframe(results.head(5).reset_index(drop=True), use_container_width=True)
                st.markdown("---")
