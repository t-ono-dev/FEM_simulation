# ring_pipeline (R(s) / R(Vtip) / fitting)

FM-AFMリング半径の **静電ポテンシャルFEM計算**（NGSolve）を共通コアで実行し、  
- `R(s)` 掃引
- `R(Vtip)` 掃引
- 実験データに対する tip radius フィット

を **100%同じ処理経路（同じメッシュ/弱形式/ホモトピー/後処理）**で回すためのスクリプト群です。

---

## 1. ファイル構成

同じフォルダに4ファイルを置いて実行します。

- `core_fem.py`：共通コア（FEM/後処理/ログ/入出力ユーティリティ）
- `run_Rs_sweep.py`：R(s) 掃引
- `run_RVtip_sweep.py`：R(Vtip) 掃引
- `fit_tip_radius.py`：フィッティング

---

## 2. 必要環境

- Python 3.x
- `ngsolve` / `netgen`
- `numpy`, `scipy`, `pandas`, `matplotlib`

例（conda環境）：
```bash
conda activate ngsolve_env
python -c "import ngsolve, netgen, numpy, scipy, pandas, matplotlib; print('ok')"
```

---

## 3. 実験データ（CSV）の置き場所と形式

### 置き場所
- 推奨：スクリプトを実行する **カレントディレクトリ**にCSVを置く  
- あるいは `--file_z`, `--file_v` で **相対/絶対パス**を指定

### 形式
#### R(s) 実験（`--file_z`）
- **1列目**：s [nm]
- **2列目**：R [nm]

例：`exp_data_z_ring1_Vtip-.csv`

#### R(Vtip) 実験（`--file_v`）
- **1列目**：Vtip [V]
- **2列目**：R [nm]

例：`exp_data_ring1_Vtip-.csv`

---

## 4. 実行方法

### 4.1 R(s) 掃引
```bash
python run_Rs_sweep.py ^
  --out_root Rs_out ^
  --s_list 3 14 30 ^
  --Vtip_fixed -2 ^
  --CPD 0.6 ^
  --tip_radius 46 ^
  --sampLe_z_nm -5 ^
  --Nd_cm3 1e16 --sigma_s 1e12 --T 300 --l_sio2 5 ^
  --model Feenstra ^
  --fermi_np aymerich_humet ^
  --levels_list 0.067 ^
  --levels_range -0.10 0.16 0.01
```

- 出力先：`Rs_out/<timestamp>/...`
- FEM生データ：`Rs_out/<timestamp>/sweep_s/...` に保存

---

### 4.2 R(Vtip) 掃引
```bash
python run_RVtip_sweep.py ^
  --out_root RV_out ^
  --Vtip_list -6 -1 30 ^
  --s_fixed 8 ^
  --CPD 0.6 ^
  --tip_radius 46 ^
  --sample_z_nm -5 ^
  --Nd_cm3 1e16 --sigma_s 1e12 --T 300 --l_sio2 5 ^
  --model Feenstra ^
  --fermi_np aymerich_humet ^
  --levels_list 0.067 ^
  --levels_range -0.10 0.16 0.01
```

- 出力先：`RV_out/<timestamp>/...`
- FEM生データ：`RV_out/<timestamp>/sweep_V/...` に保存

---

### 4.3 tip radius フィッティング
```bash
python fit_tip_radius.py ^
  --out_root fit_out ^
  --file_z exp_data_z_ring1_Vtip-.csv ^
  --file_v exp_data_ring1_Vtip-.csv ^
  --tip_radius_list 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 ^
  --levels -0.15 0.15 --n_levels 300 ^
  --Vtip_fixed -2 ^
  --s_grid 3 14 30 ^
  --s_fixed 8 ^
  --v_grid -6 -1 30 ^
  --CPD 0.6 ^
  --Nd_cm3 1e16 --sigma_s 1e12 --T 300 --l_sio2 5 ^
  --model Feenstra ^
  --fermi_np aymerich_humet ^
  --disable_log
```

- 出力先：`fit_out/<timestamp>/...`
- FEM生データ：`fit_out/<timestamp>/sweep_s/...` と `fit_out/<timestamp>/sweep_V/...`

---

## 5. オプション指定の考え方（よく使うもの）

### 共通（3スクリプトで揃えて使うと「同じ条件」になります）
- `--CPD`：V_app = Vtip - CPD に使用
- `--model`：`Feenstra` / `Boltzmann`
- `--fermi_np`：Ef計算で使う NumPy の F_{1/2} 近似  
  - `aymerich_humet`（推奨）
  - `legacy`
- `--Nd_cm3`, `--sigma_s`, `--T`, `--l_sio2`

### レベル指定（run_Rs_sweep / run_RVtip_sweep）
2系統を **同時に**指定できます（両方を結合してユニーク化）。
- `--levels_list 0.067 0.100 ...`（任意の値を列挙）
- `--levels_range START STOP STEP`（範囲指定）

例：
- 67 mV を必ず含める + 範囲スキャン
  - `--levels_list 0.067 --levels_range -0.10 0.16 0.01`

### ログ（全スクリプト共通）
- `--enable_log`：詳細ログON（`stdout.log` も保存）
- `--disable_log`：詳細ログOFF（高速・静か）

---

## 6. 出力物

各 `out_root/<timestamp>/` に以下が作られます：
- `args.json`：実行引数（再現性用）
- `dos_effective_density.json`：Nc/Nvなど（日本語キー付き）
- `stdout.log`：ログON時のみ
- `*_R_vs_s.csv`, `*_R_vs_Vtip.csv`：結果CSV
- `*_R_vs_*.png`：クイックプロット
- `sweep_s/` `sweep_V/`：FEMの中間結果（mesh, u_dimless/u_volts, metadata, vtk等）

---

## 7. よくある注意

- **同じ条件でも結果が揺れる**場合は、メッシュやホモトピーが違っていないかを疑ってください。  
  本パッケージは `core_fem.py` に統一したので、原理的に同一条件なら一致します。
- 実験CSVの列順が違うと読み間違えます（必ず 1列目=x, 2列目=R）。
