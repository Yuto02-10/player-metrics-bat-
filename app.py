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
        
        # コースの数値化
        df_each['PitchLocation'] = pd.to_numeric(df_each['PitchLocation'], errors='coerce')

        # 名前の空白削除処理
        if 'Batter' in df_each.columns:
            df_each['Batter'] = df_each['Batter'].str.replace(r'\s+', '', regex=True)
            
        if 'Pitcher' in df_each.columns:
            df_each['Pitcher'] = df_each['Pitcher'].str.replace(r'\s+', '', regex=True)
            
        df_list.append(df_each)
        
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
        if row['KorBB'] == '空振り三振': return 2.5
        if row['KorBB'] == '見逃し三振': return 2.5
        if row['KorBB'] == '四球': return -4.0
        if row['PitchResult'] == '死球': return -3.0

        # 1. インプレー（打球が前に飛んだ）以外の処理
        if row['PitchResult'] == '空振り': return 0.5
        if row['PitchResult'] == '見逃し': return 0.3
        if row['PitchResult'] == 'ファウル': return 0.3
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
    # ------------------------------------
    # AIの学習と予測
    # ------------------------------------
    if len(df_filtered) < 10:
        st.error("選択された期間のデータが少なすぎます。期間を広げてください。")
    else:
        # ------------------------------------
        # AIモデルの学習（キャッシュ化）
        # ------------------------------------
        # show_spinnerで学習中であることを画面にお知らせします
        @st.cache_resource(show_spinner="AIモデルを学習中...（初回のみ時間がかかります）")
        def train_model(df_input):
            df_work = df_input.copy()
            df_work['PitchScore'] = df_work.apply(assign_weight_advanced, axis=1)
            
            # 隣接コースへの重み付け伝播（13分割対応版）
            adjacent_map = {
                1.0: [2.0, 4.0, 5.0, 10.0, 11.0],
                2.0: [1.0, 3.0, 4.0, 5.0, 6.0, 10.0],
                3.0: [2.0, 5.0, 6.0, 10.0, 12.0],
                4.0: [1.0, 2.0, 5.0, 7.0, 8.0, 11.0],
                5.0: [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0],
                6.0: [2.0, 3.0, 5.0, 8.0, 9.0, 12.0],
                7.0: [4.0, 5.0, 8.0, 11.0, 13.0],
                8.0: [4.0, 5.0, 6.0, 7.0, 9.0, 13.0],
                9.0: [5.0, 6.0, 8.0, 12.0, 13.0],
                10.0: [1.0, 2.0, 3.0],
                11.0: [1.0, 4.0, 7.0],
                12.0: [3.0, 6.0, 9.0],
                13.0: [7.0, 8.0, 9.0]
            }
            discount_rate = 0.3
            augmented_rows = []

            for index, row in df_work.iterrows():
                augmented_rows.append(row)
                loc = row['PitchLocation']
                
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
            
            # One-Hot Encoding
            X_encoded = pd.get_dummies(X, columns=['PitcherLR', 'Batter', 'PitchType'])
            training_columns = X_encoded.columns
            
            model = RandomForestRegressor(random_state=42, n_estimators=100)
            model.fit(X_encoded, y)
            
            # 学習済みモデルと、列の構成を返す
            return model, training_columns

        # --- 関数の実行 ---
        # データ(df_filtered)が変わらない限り、2回目以降は一瞬で結果が返ってきます
        model, training_columns = train_model(df_filtered)
        
        # （予測時に使うため、特徴量リストをここで再定義しておきます）
        features = ['Ball', 'Strike', 'PitcherLR', 'Batter', 'PitchType', 'PitchLocation']
        
        # --- 予測UI ---
        # （この下からは既存の st.sidebar.header("🎯 配球シミュレーション設定") が続きます）
        
        # --- 予測UI ---
        st.sidebar.header("🎯 配球シミュレーション設定")
        
        batter_list = df_filtered['Batter'].dropna().unique()
        target_batters = st.sidebar.multiselect("対象打者を選択（複数可）", batter_list)
        
        if not target_batters:
            st.warning("打者を1人以上選択してください。")
            st.stop()
        
        c_ball = st.sidebar.slider("ボール", 0, 3, 0)
        c_strike = st.sidebar.slider("ストライク", 0, 2, 0)
        # 変更点: アウトの入力スライダー（c_out）を削除
        p_lr = st.sidebar.radio("投手の左右", ["右", "左"])
        
        # ------------------------------------
        # 予測UIと実行
        # ------------------------------------
        if st.sidebar.button("AI配球予測を開始"):
            pitch_types = df_filtered['PitchType'].unique()
            pitch_locations = df_filtered['PitchLocation'].unique()
            
            for target_batter in target_batters:
             # ------------------------------------
                # ここから下は1人の打者に対する予測処理
                # ------------------------------------
                # 変更点：変換(transform)をやめて、直接文字列を入れる
                situation = {
                    'Ball': c_ball, 'Strike': c_strike,
                    'PitcherLR': p_lr,
                    'Batter': target_batter
                }
                
                candidates = []
                for pt in pitch_types:
                    for pl in pitch_locations:
                        row = situation.copy()
                        row['PitchType'] = pt  # ここもそのまま文字列
                        row['PitchLocation'] = pl
                        candidates.append(row)
                        
                X_test = pd.DataFrame(candidates)[features]
                
                # ------------------------------------
                # 予測データにもOne-Hot Encodingを適用
                # ------------------------------------
                X_test_encoded = pd.get_dummies(X_test, columns=['PitcherLR', 'Batter', 'PitchType'])
                
                # 学習時と列の構成を完全に一致させる（データに存在しない球種などの列は0で埋める）
                X_test_encoded = X_test_encoded.reindex(columns=training_columns, fill_value=0)
                
                expected_scores = model.predict(X_test_encoded)
                
                # 変更点：文字列に戻す処理(inverse_transform)が不要になったため、スッキリしました
                results = pd.DataFrame({
                    '球種': X_test['PitchType'], 
                    'コース': X_test['PitchLocation'],
                    'AI推奨度(期待値)': expected_scores
                }).sort_values(by='AI推奨度(期待値)', ascending=False)
                
                st.subheader(f"🎯 {target_batter} 選手への推奨配球 Top 5")
                # 番号(インデックス)を非表示にしてスッキリ表示
                st.dataframe(results.head(5).reset_index(drop=True))
                st.markdown("---")
