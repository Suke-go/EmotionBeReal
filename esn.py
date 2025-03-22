import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import matplotlib
matplotlib.use('Agg')  # 表示問題を回避

# ------------------ 1. 基本機能 ------------------

def preprocess_emg(data, fs=1000):
    """EMGデータの前処理"""
    # 時間と電圧を取得
    time = data['Time[s]'].values
    emg = data['Voltage[V]'].values
    
    # バンドパスフィルタ（20-450Hz）
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [20/nyquist, 450/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, emg)
    
    # 整流化
    rectified = np.abs(filtered)
    
    # 平滑化
    window_size = int(0.05 * fs)
    smoothed = np.convolve(rectified, np.ones(window_size)/window_size, mode='same')
    
    return time, smoothed

def extract_features(signal_segment):
    """信号からの特徴量抽出"""
    # 平均絶対値 (MAV)
    mav = np.mean(np.abs(signal_segment))
    
    # 二乗平均平方根 (RMS)
    rms = np.sqrt(np.mean(np.square(signal_segment)))
    
    # 波形長 (WL)
    wl = np.sum(np.abs(np.diff(signal_segment)))
    
    # ゼロ交差率 (ZCR)
    zcr = np.sum(np.abs(np.diff(np.sign(signal_segment)))) / (2 * len(signal_segment))
    
    return [mav, rms, wl, zcr]

def segment_signal(signal, window_size=200, overlap=100):
    """信号をオーバーラップありでセグメント化"""
    segments = []
    step = window_size - overlap
    
    for i in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[i:i + window_size])
    
    return np.array(segments)

# ------------------ 2. ESNモデル ------------------

class SimpleESN:
    def __init__(self, input_size, output_size, reservoir_size=100, 
                 spectral_radius=0.8, leaking_rate=0.3, random_seed=42):
        """シンプルなESN実装（数値安定性改善版）"""
        np.random.seed(random_seed)
        
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        
        # 重みの初期化
        self.W_in = np.random.uniform(-0.5, 0.5, (reservoir_size, input_size))
        W = np.random.rand(reservoir_size, reservoir_size) - 0.5
        
        # スペクトル半径の調整（安定性向上）
        try:
            # 最大固有値を計算
            eigvals = np.linalg.eigvals(W)
            radius = np.max(np.abs(eigvals))
            if radius > 0:
                self.W = W * (spectral_radius / radius)
            else:
                # 固有値が0の場合は単純にスケール
                self.W = W * spectral_radius
        except np.linalg.LinAlgError:
            # 固有値計算に失敗した場合は単純にスケール
            print("Warning: Eigenvalue computation failed, using direct scaling")
            self.W = W * spectral_radius * 0.1
        
        # 出力重み（訓練で調整）
        self.W_out = None
    
    def _update(self, state, input_pattern):
        """リザバー状態の更新（NaNチェック付き）"""
        pre_activation = np.dot(self.W, state) + np.dot(self.W_in, input_pattern)
        # NaNを防止
        pre_activation = np.clip(pre_activation, -100, 100)
        
        new_state = (1 - self.leaking_rate) * state
        new_state += self.leaking_rate * np.tanh(pre_activation)
        
        # NaNチェック
        if np.any(np.isnan(new_state)):
            print("Warning: NaN detected in state update, resetting to zeros")
            new_state = np.zeros_like(new_state)
        
        return new_state
    
    def train(self, inputs, targets, reg_param=1e-4):
        """モデル訓練（安定性強化版）"""
        n_samples = len(inputs)
        
        # 入力のNaNチェック
        if np.any(np.isnan(inputs)):
            print("Warning: NaN detected in inputs, replacing with zeros")
            inputs = np.nan_to_num(inputs)
        
        # 目標値のNaNチェック
        if np.any(np.isnan(targets)):
            print("Warning: NaN detected in targets, replacing with zeros")
            targets = np.nan_to_num(targets)
        
        # リザーバー状態の収集
        states = np.zeros((n_samples, self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        
        for i, current_input in enumerate(inputs):
            state = self._update(state, current_input)
            states[i] = state
        
        # 訓練データ行列の準備
        X = np.hstack([np.ones((n_samples, 1)), states])
        
        # NaNチェック
        if np.any(np.isnan(X)):
            print("Warning: NaN detected in collected states, replacing with zeros")
            X = np.nan_to_num(X)
        
        try:
            # リッジ回帰でW_outを計算（SVDベース）
            ridge_term = reg_param * np.eye(X.shape[1])
            XTX = X.T @ X + ridge_term
            XTY = X.T @ targets
            
            # より安定した疑似逆行列計算
            self.W_out = np.linalg.lstsq(XTX, XTY, rcond=1e-10)[0]
            
            # NaNチェック
            if np.any(np.isnan(self.W_out)):
                raise ValueError("NaN detected in output weights")
                
        except Exception as e:
            print(f"Warning: Matrix inversion failed ({str(e)}), using pseudoinverse")
            # 疑似逆行列で計算
            X_pinv = np.linalg.pinv(X)
            self.W_out = X_pinv @ targets
        
        # トレーニング誤差
        predictions = X @ self.W_out
        error = np.mean((predictions - targets) ** 2)
        
        # NaNチェック
        if np.isnan(error):
            print("Warning: Training error is NaN, using infinity")
            error = float('inf')
            
        return error
    
    def predict(self, inputs):
        """予測を実行（NaNチェック付き）"""
        if np.any(np.isnan(inputs)):
            print("Warning: NaN detected in prediction inputs, replacing with zeros")
            inputs = np.nan_to_num(inputs)
            
        n_samples = len(inputs)
        states = np.zeros((n_samples, self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        
        for i, current_input in enumerate(inputs):
            state = self._update(state, current_input)
            states[i] = state
        
        X = np.hstack([np.ones((n_samples, 1)), states])
        
        # NaNチェック
        if np.any(np.isnan(X)):
            print("Warning: NaN detected in states during prediction, replacing with zeros")
            X = np.nan_to_num(X)
            
        predictions = X @ self.W_out
        
        # NaNチェック
        if np.any(np.isnan(predictions)):
            print("Warning: NaN detected in predictions, replacing with zeros")
            predictions = np.nan_to_num(predictions)
            
        return predictions

# ------------------ 3. メイン処理 ------------------

def main():
    print("=== EMG Gesture Recognition ===")
    
    # 訓練データロード
    train_files = {
        'grasp': 'EMG_Hand/grasp.csv',   # Class 0: Grasp
        'pinch': 'EMG_Hand/pinch.csv',   # Class 1: Pinch
        'lookup': 'EMG_Hand/lookup.csv'  # Class 2: Lookup
    }
    
    # テストデータフォルダの確認
    test_folder = 'EMG_Test'
    test_files = []
    
    if os.path.exists(test_folder):
        print(f"Test folder detected: {test_folder}")
        for file in os.listdir(test_folder):
            if file.endswith('.csv'):
                test_files.append(os.path.join(test_folder, file))
                print(f"  Test file: {file}")
    else:
        print(f"Warning: Test folder {test_folder} not found")
    
    # 1. データの前処理と特徴抽出
    print("\nPreprocessing and feature extraction...")
    
    train_features = []
    train_labels = []
    
    for gesture, filepath in train_files.items():
        try:
            # データ読み込み
            data = pd.read_csv(filepath)
            
            # 前処理
            _, processed = preprocess_emg(data)
            
            # セグメント化
            segments = segment_signal(processed)
            
            # 特徴抽出
            for segment in segments:
                features = extract_features(segment)
                train_features.append(features)
                
                # ラベル付け
                if gesture == 'grasp':
                    train_labels.append(0)
                elif gesture == 'pinch':
                    train_labels.append(1)
                elif gesture == 'lookup':
                    train_labels.append(2)
        except Exception as e:
            print(f"Error ({filepath}): {str(e)}")
    
    # NumPy配列に変換
    X_train = np.array(train_features)
    y_train = np.array(train_labels)
    
    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ワンホットエンコーディング
    n_classes = 3  # grasp, pinch, lookup
    y_train_onehot = np.zeros((len(y_train), n_classes))
    for i, label in enumerate(y_train):
        y_train_onehot[i, int(label)] = 1
    
    # 2. ESNモデルの訓練
    print("Training ESN model...")
    
    # 安定性を高めたパラメータ
    esn = SimpleESN(
        input_size=X_train_scaled.shape[1],
        output_size=n_classes,
        reservoir_size=150,
        spectral_radius=0.7,  # 減少
        leaking_rate=0.2      # 減少
    )
    
    error = esn.train(X_train_scaled, y_train_onehot, reg_param=1e-3)  # 正則化増加
    print(f"Training completed (error: {error:.6f})")
    
    # 3. 訓練データでの性能評価
    print("\nEvaluating on training data...")
    
    y_train_pred = esn.predict(X_train_scaled)
    y_train_pred_class = np.argmax(y_train_pred, axis=1)
    
    train_acc = accuracy_score(y_train, y_train_pred_class)
    print(f"Training accuracy: {train_acc:.4f}")
    
    # 混同行列
    conf_mat = confusion_matrix(y_train, y_train_pred_class)
    print("\nConfusion matrix:")
    print(conf_mat)
    
    # 4. テストデータの評価
    print("\nEvaluating test data...")
    
    # 「どれでもない」判定の閾値
    threshold = 0.5
    
    test_results = []
    
    for test_file in test_files:
        filename = os.path.basename(test_file)
        print(f"\nFile: {filename}")
        
        try:
            # テストデータの前処理
            test_data = pd.read_csv(test_file)
            _, test_processed = preprocess_emg(test_data)
            test_segments = segment_signal(test_processed)
            
            # 特徴抽出
            test_features = []
            for segment in test_segments:
                features = extract_features(segment)
                test_features.append(features)
            
            # 予測
            X_test = np.array(test_features)
            X_test_scaled = scaler.transform(X_test)
            y_test_pred = esn.predict(X_test_scaled)
            
            # 結果集計
            avg_probs = np.mean(y_test_pred, axis=0)
            max_prob = np.max(avg_probs)
            pred_class = np.argmax(avg_probs)
            
            # しきい値以下なら「どれでもない」
            gesture_names = ['Grasp', 'Pinch', 'Lookup', 'Unknown']
            if max_prob < threshold:
                result = 3  # Unknown
            else:
                result = pred_class
                
            print(f"Prediction: {gesture_names[result]} (confidence: {max_prob:.4f})")
            
            # 結果を保存
            test_results.append({
                'file': filename,
                'prediction': result,
                'confidence': max_prob,
                'probabilities': avg_probs
            })
            
        except Exception as e:
            print(f"Error ({test_file}): {str(e)}")
    
    # 5. 結果の可視化と保存
    print("\nSaving results...")
    
    # 訓練データの評価グラフ
    plt.figure(figsize=(10, 8))
    
    # 混同行列
    plt.subplot(2, 2, 1)
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Training Confusion Matrix')
    plt.colorbar()
    
    gestures = ['Grasp', 'Pinch', 'Lookup']
    plt.xticks(np.arange(len(gestures)), gestures)
    plt.yticks(np.arange(len(gestures)), gestures)
    
    # 数字を表示
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, str(conf_mat[i, j]),
                    ha="center", va="center", color="white" if conf_mat[i, j] > conf_mat.max()/2 else "black")
    
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # テスト結果を表示
    if test_results:
        plt.subplot(2, 2, 2)
        
        files = [r['file'] for r in test_results]
        predictions = [r['prediction'] for r in test_results]
        
        # 棒グラフで予測結果を表示
        colors = ['blue', 'orange', 'green', 'red']
        for i, (file, pred) in enumerate(zip(files, predictions)):
            plt.bar(i, 1, color=colors[pred])
        
        plt.xticks(range(len(files)), files, rotation=45)
        plt.yticks([])
        plt.title('Test File Classification Results')
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Grasp'),
            Patch(facecolor='orange', label='Pinch'),
            Patch(facecolor='green', label='Lookup'),
            Patch(facecolor='red', label='Unknown')
        ]
        plt.legend(handles=legend_elements)
    
    # 保存
    plt.tight_layout()
    plt.savefig('results_simple.png')
    print("Graph saved to 'results_simple.png'")
    
    # CSVでも結果を保存
    if test_results:
        result_df = pd.DataFrame({
            'Filename': [r['file'] for r in test_results],
            'Classification': [gesture_names[r['prediction']] for r in test_results],
            'Confidence': [r['confidence'] for r in test_results],
        })
        
        result_df.to_csv('classification_results_simple.csv', index=False)
        print("Classification results saved to 'classification_results_simple.csv'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()