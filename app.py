import streamlit as st
import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px

st.set_page_config(page_title="配球アシスタントAI (ハイブリッド予測)", layout="wide")
st.title("配球アシスタントAI（直近×長期 ハイブリッドモード）")

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
    # サイドバー：期間と重視度の設定
    # ------------------------------------
    st.sidebar.header("📅 データの期間設定")
    
    min_date = df_raw['Date'].min().date()
    max_date = df_raw['Date'].max().date()
    
    st.sidebar.subheader("① 直近データ（前回のリーグ戦など）")
    date_range_recent = st.sidebar.date_input(
        "直近の期間を選択", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, key="dr_recent"
    )
    
    st.sidebar.subheader("② 長期データ（通算・シーズン全体など）")
    date_range_long = st.sidebar.date_input(
        "長期の期間を選択", value=(min_date, max_date),
        min_value=min_date, max_value=max_date, key="dr_long"
    )
    
    def filter_by_date(df, date_range):
        if len(date_range) == 2:
            return df[(df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])]
        return df

    df_recent = filter_by_date(df_raw, date_range_recent)
    df_long = filter_by_date(df_raw, date_range_long)
    
    st.sidebar.write(f"対象球数 - 直近: {len(df_recent)}球 / 長期: {len(df_long)}球")
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚖️ 直近データの重視度")
    weight_percent = st.sidebar.slider(
        "直近の調子をどの程度重視しますか？", 
        min_value=0, max_value=100, value=30, step=5,
        help="例: 30%にすると、「直近の予測30% ＋ 長期の予測70%」で最終推奨度を計算します。"
    )
    weight_recent = weight_percent / 100.0
    weight_long = 1.0 - weight_recent

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
            return None 
            
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

    model_recent = train_model(df_recent)
    model_long = train_model(df_long)
    
    features = ['Ball', 'Strike', 'PitcherLR', 'Batter', 'PitchType', 'PitchLocation']
    global_pitch_types = df_raw['PitchType'].dropna().unique()
    global_pitch_locations = sorted(df_raw['PitchLocation'].dropna().unique())
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 配球シミュレーション設定")
    
    batter_list = df_raw['Batter'].dropna().unique()
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
    
    if st.sidebar.button("AIハイブリッド配球予測を開始", use_container_width=True):
        
        breaking_balls = ['スライダー', 'フォーク', 'カーブ', 'チェンジアップ', 'スプリット', 'シンカー', 'カットボール']
        penalty_map = {1.0: 2.0, 2.0: 3.5, 3.0: 2.0, 5.0: 1.5, 11.0: 1.0}
        
        for target_batter in target_batters:
            st.subheader(f"🎯 {target_batter} 選手へのハイブリッド推奨配球")
            st.write(f"算出比率: 直近データ {weight_percent}% ＋ 長期データ {100 - weight_percent}%")
            
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
            
            # 直近と長期のモデルそれぞれで予測を出す
            pred_recent = model_recent.predict(X_test) if model_recent is not None else 0
            pred_long = model_long.predict(X_test) if model_long is not None else 0
            
            # データの不足によるフォールバック処理
            if model_recent is None:
                final_scores = pred_long
                st.warning("直近データが少なすぎるため、長期データ100%で算出しました。")
            elif model_long is None:
                final_scores = pred_recent
                st.warning("長期データが少なすぎるため、直近データ100%で算出しました。")
            else:
                # 指定された重視度（ウェイト）で期待値をブレンド
                final_scores = (pred_recent * weight_recent) + (pred_long * weight_long)
            
            results = pd.DataFrame({
                '球種': X_test['PitchType'], 
                'コース': X_test['PitchLocation'],
                '直近スコア': pred_recent if model_recent is not None else 0,
                '長期スコア': pred_long if model_long is not None else 0,
                '総合推奨度': final_scores
            })

            # ペナルティ処理（統合後のスコアから減点する）
            def apply_risk_penalty(row):
                score = row['総合推奨度']
                if row['球種'] in breaking_balls and row['コース'] in penalty_map:
                    score -= penalty_map[row['コース']]
                return score

            results['総合推奨度'] = results.apply(apply_risk_penalty, axis=1)
            results_sorted = results.sort_values(by='総合推奨度', ascending=False)
            
            # Top5の表示（直近と長期の内訳も表示して根拠をわかりやすく）
            st.dataframe(
                results_sorted[['球種', 'コース', '総合推奨度', '直近スコア', '長期スコア']].head(5).reset_index(drop=True),
                use_container_width=True
            )
            
            # ヒートマップ描画
            st.markdown("##### 📊 球種×コース 総合推奨度ヒートマップ")
            pivot_recommend = results.pivot(index='球種', columns='コース', values='総合推奨度')
            
            fig = px.imshow(
                pivot_recommend,
                labels=dict(x="コース番号", y="球種", color="推奨度"),
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
