import streamlit as st
import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("配球アシスタントAI")

# 1. データの自動一括読み込みと日付の型変換
@st.cache_data
def load_all_data():
    file_paths = glob.glob('試合データ/*.csv')
    if not file_paths:
        return None
        
    df_list = []
    for path in file_paths:
        # Shift-JISなど文字化けする場合は encoding='shift_jis' を追加してください
        df_each = pd.read_csv(path).dropna(subset=['PitchType', 'PitchLocation'])
        # Date列を日付型に変換 (フォーマットを自動解析)
        # errors='coerce' を追加して、日付に変換できない文字は NaT にする
        df_each['Date'] = pd.to_datetime(df_each['Date'], errors='coerce')
        
        # Date列が NaT (無効な日付) になってしまった行をデータから除外する
        df_each = df_each.dropna(subset=['Date'])
        df_list.append(df_each)
        df_each['PitchLocation'] = pd.to_numeric(df_each['PitchLocation'], errors='coerce')
        
    return pd.concat(df_list, ignore_index=True)

df_raw = load_all_data()

if df_raw is None:
    st.warning("GitHubの data/ フォルダにCSVファイルを追加してください。")
else:
    # ------------------------------------
    # サイドバー：日付期間の絞り込みUI
    # ------------------------------------
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
    
    # ------------------------------------
    # 重み付けルールの定義
    # ------------------------------------
    def assign_weight_advanced(row):
        # 0. 三振や四死球など、結果が確定するイベントを最優先で評価
        if row['KorBB'] == '空振り三振': return 1.8
        if row['KorBB'] == '見逃し三振': return 1.8
        if row['KorBB'] == '四球': return -4.0
        if row['PitchResult'] == '死球': return -5.0

        # 1. インプレー（打球が前に飛んだ）以外の処理
        if row['PitchResult'] == '空振り': return 0.3
        if row['PitchResult'] == '見逃し': return 0.2
        if row['PitchResult'] == 'ファウル': return 0.2
        if row['PitchResult'] == 'ボール': return -0.1

        # 2. インプレーの場合（打球性質 × 結果 の組み合わせ）
        if row['PitchResult'] == 'インプレー':
            hit_type = str(row['HitType'])
            hit_result = str(row['HitResult'])
            catch_position = str(row['Catch'])
            
            infielders = ['投手', '捕手', '一塁手', '二塁手', '三塁手', '遊撃手']
            
            # --- アウトの評価ロジック ---
            if hit_result == 'アウト' or hit_result == 'nan':
                if hit_type == 'フライ':
                    if catch_position in infielders:
                        return 2.0  # 内野フライ
                    else:
                        return 1.0  # 外野フライ
                elif hit_type == 'ゴロ':
                    return 2.5      # ゴロアウト
                elif hit_type == 'ライナー':
                    return 1.5      # ライナーアウト
                else:
                    return 0.5      # その他のアウト
            
            # --- ヒット・エラーなどの評価ロジック（マトリクス） ---
            weight_matrix = {
                ('ゴロ', '単打'): -5,
                ('ライナー', '単打'): -20,
                ('フライ', '単打'): -15,
                ('ライナー', '二塁打'): -40,
                ('フライ', '二塁打'): -50, 
                ('ゴロ', '二塁打'): -10,
                ('フライ', '三塁打'): -75,
                ('ライナー', '三塁打'): -60,
                ('ゴロ', '三塁打'): -15,
                ('フライ', '本塁打'): -100,
                ('ゴロ', '本塁打'): -20,
                ('ライナー', '本塁打'): -80,
                ('ゴロ', 'エラー'): 2.5,
                ('フライ', 'エラー'): 1.0,         
                ('ライナー', 'エラー'): 1.5,             
            }
            
            return weight_matrix.get((hit_type, hit_result), 0.0)

        return 0.0

    # ------------------------------------
    # AIの学習と予測
    # ------------------------------------
    if len(df_filtered) < 10:
        st.error("選択された期間のデータが少なすぎます。期間を広げてください。")
    else:
        df_filtered = df_filtered.copy()
        # 修正: 関数名を assign_weight_advanced に変更
        df_filtered['PitchScore'] = df_filtered.apply(assign_weight_advanced, axis=1)
        # ------------------------------------
        # 隣接コースへの重み付け伝播（ターゲットスムージング）
        # ------------------------------------
        # コース1〜9の隣接マップ（必要に応じてボールゾーンも追加可能です）
        adjacent_map = {
            1.0: [2.0, 4.0, 5.0],
            2.0: [1.0, 3.0, 4.0, 5.0, 6.0],
            3.0: [2.0, 5.0, 6.0],
            4.0: [1.0, 2.0, 5.0, 7.0, 8.0],
            5.0: [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0],
            6.0: [2.0, 3.0, 5.0, 8.0, 9.0],
            7.0: [4.0, 5.0, 8.0],
            8.0: [4.0, 5.0, 6.0, 7.0, 9.0],
            9.0: [5.0, 6.0, 8.0]
        }

        # 一律の減衰率を設定（例: 0.5 なら 50% のスコアを伝播）
        discount_rate = 0.3
        augmented_rows = []

        for index, row in df_filtered.iterrows():
            # 1. 実際に投球されたオリジナルデータを追加
            augmented_rows.append(row)
            
            # 2. 投球コースを取得（小数点表記に統一済みの前提）
            loc = row['PitchLocation']
            
            # loc が NaN(欠損値) ではなく、かつ隣接マップに存在する場合のみ処理
            if pd.notna(loc) and loc in adjacent_map:
                for adj_loc in adjacent_map[loc]:
                    new_row = row.copy()
                    new_row['PitchLocation'] = adj_loc
                    # 隣接コースには減衰させたスコアを付与
                    new_row['PitchScore'] = row['PitchScore'] * discount_rate
                    augmented_rows.append(new_row)

        # 拡張したデータをAIの学習用データ(df_train)としてデータフレーム化
        df_train = pd.DataFrame(augmented_rows)

        # 以降のAI学習（RandomForestRegressorのfitなど）は df_train を使用する
        # 例: X = df_train[features], y = df_train['PitchScore']
        
        features = ['Ball', 'Strike', 'Out', 'PitcherLR', 'Batter', 'PitchType', 'PitchLocation']
        X = df_filtered[features].copy()
        y = df_filtered['PitchScore']
        
        le_dict = {}
        for col in ['PitcherLR', 'Batter', 'PitchType']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
            
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X, y)
        
        # --- 予測UI ---
        st.sidebar.header("🎯 配球シミュレーション設定")
        all_batters = sorted(df_filtered['Batter'].dropna().unique())
        
        
        target_batters = st.sidebar.multiselect("対象打者を選択（複数可）", batter_list)
        
        
　　　　 if not target_batters:
    　　st.warning("打者を1人以上選択してください。")
    　　st.stop()
        
        c_ball = st.sidebar.slider("ボール", 0, 3, 0)
        c_strike = st.sidebar.slider("ストライク", 0, 2, 0)
        c_out = st.sidebar.slider("アウト", 0, 2, 0)
        p_lr = st.sidebar.radio("投手の左右", ["右", "左"])
        
        # ------------------------------------
        # 予測UIと実行
        # ------------------------------------
        if st.sidebar.button("AI配球予測を開始"):
            pitch_types = df_filtered['PitchType'].unique()
            pitch_locations = df_filtered['PitchLocation'].unique()
            
            # 選ばれた複数の打者それぞれに対して予測を実行するループ
            for target_batter in target_batters:
                
                # ------------------------------------
                # ここから下は1人の打者に対する予測処理
                # ------------------------------------
                situation = {
                    'Ball': c_ball, 'Strike': c_strike, 'Out': c_out,
                    'PitcherLR': le_dict['PitcherLR'].transform([p_lr])[0],
                    'Batter': le_dict['Batter'].transform([target_batter])[0]
                }
                
                candidates = []
                for pt in pitch_types:
                    for pl in pitch_locations:
                        row = situation.copy()
                        row['PitchType'] = le_dict['PitchType'].transform([pt])[0]
                        row['PitchLocation'] = pl
                        candidates.append(row)
                        
                X_test = pd.DataFrame(candidates)[features]
                expected_scores = model.predict(X_test)
                
                results = pd.DataFrame({
                    '球種': X_test['PitchType'].apply(lambda x: le_dict['PitchType'].inverse_transform([x])[0]),
                    'コース': X_test['PitchLocation'],
                    'AI推奨度(期待値)': expected_scores
                }).sort_values(by='AI推奨度(期待値)', ascending=False)
                
                # 打者ごとに見出しと表を出力
                st.subheader(f"🎯 {target_batter} 選手への推奨配球 Top 5")
                st.dataframe(results.head(5))
                
                # 複数の表が連続して表示されるため、区切り線を入れると見やすくなります
                st.markdown("---")
