import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import parselmouth
from parselmouth.praat import call
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import seaborn as sns
import threading

# ─── THEME ────────────────────────────────────────────────────────────────────
BG         = "#0f172a"   # slate-900
PANEL      = "#1e293b"   # slate-800
CARD       = "#334155"   # slate-700
ACCENT     = "#38bdf8"   # sky-400
ACCENT2    = "#818cf8"   # indigo-400
SUCCESS    = "#4ade80"   # green-400
WARNING    = "#fb923c"   # orange-400
DANGER     = "#f87171"   # red-400
TEXT       = "#f1f5f9"   # slate-100
MUTED      = "#94a3b8"   # slate-400
FONT_MONO  = ("Consolas", 10)
FONT_UI    = ("Segoe UI", 10)
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_BTN   = ("Segoe UI", 10, "bold")

# ─── GLOBALS ──────────────────────────────────────────────────────────────────
filename   = None
dataset    = None
X = Y      = None
X_train = X_test = y_train = y_test = None
cnn        = None
sc         = MinMaxScaler(feature_range=(0, 1))
accuracy   = []
precision  = []
recall     = []
fscore     = []
algorithms = []

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def getLabel(name):
    return 1 if name == "PD" else 0

def measurePitch(voice_data, f0min, f0max, unit):
    sound = parselmouth.Sound(voice_data)
    call(sound, "To Pitch", 0.0, f0min, f0max)
    pp = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    feats = [
        call(pp,            "Get jitter (local)",           0, 0, 0.0001, 0.02, 1.3),
        call(pp,            "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3),
        call(pp,            "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3),
        call(pp,            "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3),
        call([sound, pp],   "Get shimmer (local)",          0, 0, 0.0001, 0.02, 1.3, 1.6),
        call([sound, pp],   "Get shimmer (local_dB)",       0, 0, 0.0001, 0.02, 1.3, 1.6),
        call([sound, pp],   "Get shimmer (apq3)",           0, 0, 0.0001, 0.02, 1.3, 1.6),
        call([sound, pp],   "Get shimmer (apq5)",           0, 0, 0.0001, 0.02, 1.3, 1.6),
        call([sound, pp],   "Get shimmer (apq11)",          0, 0, 0.0001, 0.02, 1.3, 1.6),
    ]
    for freq in [500, 1500, 2500, 3500, 3800]:
        h = call(sound, "To Harmonicity (cc)", 0.01, freq, 0.1, 1.0)
        feats.append(call(h, "Get mean", 0, 0))
    return feats

# ─── APP CLASS ────────────────────────────────────────────────────────────────
class ParkinsonsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Parkinson Disease Detection · Deep Neural Networks")
        self.geometry("1300x820")
        self.configure(bg=BG)
        self.resizable(True, True)

        # ttk style
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",          background=BG,    borderwidth=0)
        style.configure("TNotebook.Tab",      background=PANEL, foreground=MUTED,
                        font=FONT_UI, padding=[18, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)],
                  foreground=[("selected", ACCENT)])
        style.configure("TFrame",             background=BG)
        style.configure("Vertical.TScrollbar", background=CARD, troughcolor=PANEL)

        self._build_header()
        self._build_tabs()
        self._build_statusbar()

    # ── HEADER ────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self, bg=PANEL, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        dot = tk.Frame(hdr, bg=ACCENT, width=6, height=36)
        dot.pack(side="left", padx=(20, 12), pady=14)

        tk.Label(hdr, text="Parkinson Disease Detection",
                 font=("Segoe UI", 17, "bold"), bg=PANEL, fg=TEXT).pack(side="left", anchor="w")
        tk.Label(hdr, text="Deep Neural Network Classifier",
                 font=("Segoe UI", 10), bg=PANEL, fg=MUTED).pack(side="left", padx=(10,0), anchor="w")

    # ── TABS ──────────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.tab_data    = tk.Frame(self.nb, bg=BG)
        self.tab_train   = tk.Frame(self.nb, bg=BG)
        self.tab_predict = tk.Frame(self.nb, bg=BG)
        self.tab_compare = tk.Frame(self.nb, bg=BG)

        self.nb.add(self.tab_data,    text="  📂  Dataset  ")
        self.nb.add(self.tab_train,   text="  🧠  Train  ")
        self.nb.add(self.tab_predict, text="  🎙  Predict  ")
        self.nb.add(self.tab_compare, text="  📊  Compare  ")

        self._build_tab_data()
        self._build_tab_train()
        self._build_tab_predict()
        self._build_tab_compare()

    # ── STATUS BAR ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=PANEL, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self.status_var,
                 font=("Segoe UI", 9), bg=PANEL, fg=MUTED, anchor="w").pack(
                     side="left", padx=16, fill="y")
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=16, pady=6)

    def set_status(self, msg, busy=False):
        self.status_var.set(msg)
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()
        self.update_idletasks()

    # ── TAB: DATASET ──────────────────────────────────────────────────────────
    def _build_tab_data(self):
        t = self.tab_data
        # left panel: controls + log
        left = tk.Frame(t, bg=BG, width=400)
        left.pack(side="left", fill="y", padx=(20,10), pady=20)
        left.pack_propagate(False)

        tk.Label(left, text="1 · Load Dataset", font=FONT_TITLE,
                 bg=BG, fg=ACCENT).pack(anchor="w")
        tk.Label(left, text="Select the folder containing PD & Healthy speech files.",
                 font=FONT_UI, bg=BG, fg=MUTED, wraplength=360, justify="left").pack(
                     anchor="w", pady=(4,12))

        self._btn(left, "Upload Parkinson Speech Dataset",
                  self._upload_dataset).pack(fill="x")

        tk.Frame(left, bg=CARD, height=1).pack(fill="x", pady=14)

        tk.Label(left, text="2 · Preprocess", font=FONT_TITLE,
                 bg=BG, fg=ACCENT).pack(anchor="w")
        tk.Label(left, text="Normalize features and split 80 / 20 for train / test.",
                 font=FONT_UI, bg=BG, fg=MUTED, wraplength=360, justify="left").pack(
                     anchor="w", pady=(4,12))
        self._btn(left, "Preprocess Dataset", self._preprocess_dataset).pack(fill="x")

        tk.Frame(left, bg=CARD, height=1).pack(fill="x", pady=14)

        # log
        tk.Label(left, text="Log", font=FONT_TITLE, bg=BG, fg=MUTED).pack(anchor="w")
        self.data_log = self._log_box(left)

        # right panel: inline graph
        right = tk.Frame(t, bg=PANEL, bd=0)
        right.pack(side="left", fill="both", expand=True, padx=(0,20), pady=20)
        tk.Label(right, text="Dataset Distribution",
                 font=FONT_TITLE, bg=PANEL, fg=TEXT).pack(anchor="w", padx=16, pady=(12,0))
        self.fig_data = Figure(figsize=(5,4), facecolor=PANEL)
        self.ax_data  = self.fig_data.add_subplot(111)
        self._style_ax(self.ax_data)
        self.canvas_data = FigureCanvasTkAgg(self.fig_data, master=right)
        self.canvas_data.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=16)

    # ── TAB: TRAIN ────────────────────────────────────────────────────────────
    def _build_tab_train(self):
        t = self.tab_train
        # top: buttons row
        btn_row = tk.Frame(t, bg=BG)
        btn_row.pack(fill="x", padx=20, pady=(20,0))
        tk.Label(btn_row, text="Run Algorithms", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(
            side="left", anchor="w")

        for label, cmd in [
            ("SVM",     self._run_svm),
            ("XGBoost", self._run_xgboost),
            ("MLP",     self._run_mlp),
            ("CNN",     self._run_cnn),
        ]:
            self._btn(btn_row, label, cmd, width=12).pack(side="left", padx=(12,0))

        # split: left log, right graph
        pane = tk.Frame(t, bg=BG)
        pane.pack(fill="both", expand=True, padx=20, pady=14)

        left = tk.Frame(pane, bg=PANEL, width=380)
        left.pack(side="left", fill="y", padx=(0,12))
        left.pack_propagate(False)
        tk.Label(left, text="Metrics", font=FONT_TITLE, bg=PANEL, fg=TEXT).pack(
            anchor="w", padx=14, pady=(12,6))
        self.train_log = self._log_box(left, bg=PANEL)

        right = tk.Frame(pane, bg=PANEL)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="Confusion Matrix", font=FONT_TITLE, bg=PANEL, fg=TEXT).pack(
            anchor="w", padx=14, pady=(12,0))
        self.fig_cm  = Figure(figsize=(5,4), facecolor=PANEL)
        self.ax_cm   = self.fig_cm.add_subplot(111)
        self._style_ax(self.ax_cm)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=right)
        self.canvas_cm.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=14)

    # ── TAB: PREDICT ──────────────────────────────────────────────────────────
    def _build_tab_predict(self):
        t = self.tab_predict
        ctr = tk.Frame(t, bg=BG)
        ctr.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(ctr, text="Predict from Speech File",
                 font=("Segoe UI", 20, "bold"), bg=BG, fg=ACCENT).pack(pady=(0,6))
        tk.Label(ctr, text="Upload a .wav file to classify as Healthy or Parkinson Disease.",
                 font=FONT_UI, bg=BG, fg=MUTED).pack(pady=(0,20))

        self._btn(ctr, "🎙  Select Speech File & Predict",
                  self._predict, width=32).pack()

        self.pred_card = tk.Frame(ctr, bg=PANEL, pady=24, padx=40)
        self.pred_card.pack(pady=24, fill="x")

        self.pred_label = tk.Label(self.pred_card, text="—",
                                   font=("Segoe UI", 28, "bold"),
                                   bg=PANEL, fg=TEXT)
        self.pred_label.pack()

        self.pred_file = tk.Label(self.pred_card, text="No file selected.",
                                  font=FONT_UI, bg=PANEL, fg=MUTED)
        self.pred_file.pack(pady=(8,0))

        self.pred_feats = tk.Label(self.pred_card, text="",
                                   font=FONT_MONO, bg=PANEL, fg=MUTED,
                                   wraplength=560, justify="left")
        self.pred_feats.pack(pady=(8,0))

    # ── TAB: COMPARE ──────────────────────────────────────────────────────────
    def _build_tab_compare(self):
        t = self.tab_compare
        top = tk.Frame(t, bg=BG)
        top.pack(fill="x", padx=20, pady=(20,0))
        tk.Label(top, text="Algorithm Comparison", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(side="left")
        self._btn(top, "Refresh Graph", self._show_comparison, width=16).pack(side="left", padx=16)

        self.fig_cmp  = Figure(figsize=(9,5), facecolor=BG)
        self.ax_cmp   = self.fig_cmp.add_subplot(111)
        self._style_ax(self.ax_cmp)
        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=t)
        self.canvas_cmp.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=16)

    # ── WIDGET HELPERS ────────────────────────────────────────────────────────
    def _btn(self, parent, text, cmd, width=None):
        kw = dict(font=FONT_BTN, bg=ACCENT, fg=BG,
                  activebackground=ACCENT2, activeforeground=TEXT,
                  relief="flat", padx=14, pady=9, cursor="hand2",
                  command=cmd, bd=0)
        if width:
            kw["width"] = width
        return tk.Button(parent, text=text, **kw)

    def _log_box(self, parent, bg=BG):
        frame = tk.Frame(parent, bg=bg)
        frame.pack(fill="both", expand=True, pady=(6,0))
        sb = ttk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        box = tk.Text(frame, font=FONT_MONO, bg="#0d1a2e", fg=TEXT,
                      relief="flat", padx=10, pady=8,
                      insertbackground=ACCENT,
                      yscrollcommand=sb.set, wrap="word")
        box.pack(fill="both", expand=True)
        sb.config(command=box.yview)
        return box

    def _log(self, box, msg, tag=None):
        box.config(state="normal")
        box.insert("end", msg + "\n")
        box.see("end")
        box.config(state="disabled")
        self.update_idletasks()

    def _clear_log(self, box):
        box.config(state="normal")
        box.delete("1.0", "end")
        box.config(state="disabled")

    def _style_ax(self, ax):
        ax.set_facecolor("#0d1a2e")
        for spine in ax.spines.values():
            spine.set_color(CARD)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)

    # ── ACTIONS ───────────────────────────────────────────────────────────────
    def _upload_dataset(self):
        global dataset, X, Y, filename
        fname = filedialog.askdirectory(initialdir=".")
        if not fname:
            return
        filename = fname
        self._clear_log(self.data_log)
        self._log(self.data_log, f"Directory: {fname}")
        self.set_status("Loading dataset…", busy=True)

        def task():
            global dataset, X, Y
            try:
                if os.path.exists("ProcessedData/processed_results.csv"):
                    dataset = pd.read_csv("ProcessedData/processed_results.csv")
                    dataset.fillna(0, inplace=True)
                    self._log(self.data_log, "Loaded cached processed CSV.")
                else:
                    X, Y = [], []
                    for folder in ["ParkinsonDataset/ReadText",
                                   "ParkinsonDataset/SpontaneousDialogue"]:
                        for root, _, files in os.walk(folder):
                            for f in files:
                                name  = os.path.basename(root)
                                label = getLabel(name)
                                sound = parselmouth.Sound(os.path.join(root, f))
                                feats = measurePitch(sound, 75, 1000, "Hertz")
                                X.append(feats); Y.append(label)
                    cols = ["Jitter_rel","Jitter_abs","Jitter_RAP","Jitter_PPQ",
                            "Shim_loc","Shim_dB","Shim_APQ3","Shim_APQ5","Shi_APQ11",
                            "hnr05","hnr15","hnr25","hnr35","hnr38"]
                    dataset = pd.DataFrame(X, columns=cols)
                    for c in ["hnr25","hnr15","hnr35","hnr38"]:
                        dataset[c].fillna(dataset[c].mean(), inplace=True)
                    dataset["Label"] = Y
                    os.makedirs("ProcessedData", exist_ok=True)
                    dataset.to_csv("ProcessedData/processed_results.csv", index=False)
                    dataset = pd.read_csv("ProcessedData/processed_results.csv")
                    dataset.fillna(0, inplace=True)

                self._log(self.data_log,
                          f"Records: {len(dataset)}   Features: {dataset.shape[1]-1}")
                self._log(self.data_log, "\n" + dataset.head().to_string())

                # inline bar chart
                counts = dataset["Label"].value_counts().sort_index()
                self.ax_data.clear()
                self._style_ax(self.ax_data)
                bars = self.ax_data.bar(["Healthy (0)", "Parkinson (1)"],
                                        counts.values,
                                        color=[SUCCESS, DANGER], width=0.45,
                                        edgecolor="none")
                for bar, val in zip(bars, counts.values):
                    self.ax_data.text(bar.get_x() + bar.get_width()/2,
                                      bar.get_height() + 0.5,
                                      str(val), ha="center", color=TEXT,
                                      fontsize=11, fontweight="bold")
                self.ax_data.set_title("Class Distribution", color=TEXT)
                self.ax_data.set_ylabel("Count", color=MUTED)
                self.fig_data.tight_layout()
                self.canvas_data.draw()

                self.set_status(f"Dataset loaded — {len(dataset)} samples.")
            except Exception as e:
                self._log(self.data_log, f"ERROR: {e}")
                self.set_status("Error loading dataset.")

        threading.Thread(target=task, daemon=True).start()

    def _preprocess_dataset(self):
        global X_train, X_test, y_train, y_test, X, Y, sc, dataset
        if dataset is None:
            messagebox.showwarning("No dataset", "Load a dataset first.")
            return
        vals = dataset.values
        X    = vals[:, :-1].astype(float)
        Y    = vals[:, -1].astype(float)
        sc   = MinMaxScaler()
        sc.fit(X); X = sc.transform(X)
        idx  = np.arange(X.shape[0]); np.random.shuffle(idx)
        X    = X[idx]; Y = Y[idx]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        self._clear_log(self.data_log)
        self._log(self.data_log, "✔ Preprocessing complete.")
        self._log(self.data_log, f"  Total samples : {X.shape[0]}")
        self._log(self.data_log, f"  Training      : {X_train.shape[0]}")
        self._log(self.data_log, f"  Testing       : {X_test.shape[0]}")
        self.set_status("Preprocessing done — ready to train.")

    def _calc_metrics(self, algo, pred, y_true):
        a = accuracy_score(y_true, pred) * 100
        p = precision_score(y_true, pred, average="macro", zero_division=0) * 100
        r = recall_score(y_true, pred, average="macro", zero_division=0) * 100
        f = f1_score(y_true, pred, average="macro", zero_division=0) * 100
        accuracy.append(a); precision.append(p); recall.append(r); fscore.append(f)
        algorithms.append(algo)
        self._log(self.train_log, f"{'─'*30}")
        self._log(self.train_log, f" {algo}")
        self._log(self.train_log, f"  Accuracy  : {a:.2f}%")
        self._log(self.train_log, f"  Precision : {p:.2f}%")
        self._log(self.train_log, f"  Recall    : {r:.2f}%")
        self._log(self.train_log, f"  F1 Score  : {f:.2f}%")

        # inline confusion matrix
        cm     = confusion_matrix(y_true, pred)
        labels = ["Healthy", "Parkinson"]
        self.ax_cm.clear()
        self._style_ax(self.ax_cm)
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    ax=self.ax_cm, linewidths=0.5,
                    annot_kws={"size": 13, "color": TEXT})
        self.ax_cm.set_title(f"{algo} — Confusion Matrix", color=TEXT, pad=10)
        self.ax_cm.set_xlabel("Predicted", color=MUTED)
        self.ax_cm.set_ylabel("True", color=MUTED)
        self.fig_cm.tight_layout()
        self.canvas_cm.draw()

    def _guard(self):
        if X_train is None:
            messagebox.showwarning("Not ready", "Preprocess the dataset first.")
            return False
        return True

    def _run_svm(self):
        if not self._guard(): return
        global accuracy, precision, recall, fscore, algorithms
        accuracy=[]; precision=[]; recall=[]; fscore=[]; algorithms=[]
        self._clear_log(self.train_log)
        self.set_status("Training SVM…", busy=True)
        def task():
            cls = svm.SVC(); cls.fit(X_train, y_train)
            self._calc_metrics("SVM", cls.predict(X_test), y_test)
            self.set_status("SVM done.")
        threading.Thread(target=task, daemon=True).start()

    def _run_xgboost(self):
        if not self._guard(): return
        self.set_status("Training XGBoost…", busy=True)
        def task():
            cls = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            cls.fit(X_train, y_train)
            self._calc_metrics("XGBoost", cls.predict(X_test), y_test)
            self.set_status("XGBoost done.")
        threading.Thread(target=task, daemon=True).start()

    def _run_mlp(self):
        if not self._guard(): return
        self.set_status("Training MLP…", busy=True)
        def task():
            cls = MLPClassifier(max_iter=5000); cls.fit(X_train, y_train)
            self._calc_metrics("MLP", cls.predict(X_test), y_test)
            self.set_status("MLP done.")
        threading.Thread(target=task, daemon=True).start()

    def _run_cnn(self):
        if not self._guard(): return
        self.set_status("Training CNN…", busy=True)
        def task():
            global cnn
            X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
            Y1 = to_categorical(Y)
            Xtr, Xte, ytr, yte = train_test_split(X1, Y1, test_size=0.2)

            if os.path.exists("model/model.json"):
                with open("model/model.json") as jf:
                    cnn = model_from_json(jf.read())
                cnn.load_weights("model/model_weights.h5")
            else:
                cnn = Sequential([
                    Conv2D(128, (1,1), input_shape=Xtr.shape[1:], activation="relu"),
                    MaxPooling2D((1,1)),
                    Conv2D(256, (1,1), activation="relu"),
                    MaxPooling2D((1,1)),
                    Flatten(),
                    Dense(256, activation="relu"),
                    Dense(ytr.shape[1], activation="softmax"),
                ])
                cnn.compile(optimizer="adam",
                            loss="categorical_crossentropy",
                            metrics=["accuracy"])
                cnn.fit(Xtr, ytr, batch_size=4, epochs=30,
                        shuffle=True, verbose=0,
                        validation_data=(Xte, yte))
                os.makedirs("model", exist_ok=True)
                cnn.save_weights("model/model_weights.h5")
                with open("model/model.json", "w") as jf:
                    jf.write(cnn.to_json())

            pred = np.argmax(cnn.predict(Xte), axis=1)
            ytrue = np.argmax(yte, axis=1)
            self._calc_metrics("CNN", pred, ytrue)
            self.set_status("CNN done.")
        threading.Thread(target=task, daemon=True).start()

    def _predict(self):
        if cnn is None:
            messagebox.showwarning("No model", "Train the CNN first.")
            return
        fname = filedialog.askopenfilename(
            initialdir="testSpeechFiles",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not fname:
            return
        try:
            sound = parselmouth.Sound(fname)
            feats = measurePitch(sound, 75, 1000, "Hertz")
            test  = sc.transform(np.array([feats]))
            test1 = np.reshape(test, (1, test.shape[1], 1, 1))
            pred  = np.argmax(cnn.predict(test1), axis=1)[0]
            label = "Parkinson Disease" if pred == 1 else "Healthy"
            color = DANGER if pred == 1 else SUCCESS
            self.pred_label.config(text=label, fg=color)
            self.pred_file.config(text=f"File: {os.path.basename(fname)}")
            self.pred_feats.config(
                text="Features: " + "  ".join(f"{v:.5f}" for v in feats))
            self.nb.select(self.tab_predict)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_comparison(self):
        if not accuracy:
            messagebox.showinfo("No data", "Run at least one algorithm first.")
            return
        self.ax_cmp.clear()
        self._style_ax(self.ax_cmp)
        n   = len(algorithms)
        x   = np.arange(n)
        w   = 0.2
        colors = [ACCENT, ACCENT2, SUCCESS, WARNING]
        metrics_data = [accuracy, precision, recall, fscore]
        mlabels      = ["Accuracy", "Precision", "Recall", "F1 Score"]
        for i, (m, c, ml) in enumerate(zip(metrics_data, colors, mlabels)):
            if len(m) == n:
                self.ax_cmp.bar(x + i*w, m, w, label=ml, color=c,
                                edgecolor="none", alpha=0.9)
        self.ax_cmp.set_xticks(x + 1.5*w)
        self.ax_cmp.set_xticklabels(algorithms, color=TEXT, fontsize=10)
        self.ax_cmp.set_ylabel("Score (%)", color=MUTED)
        self.ax_cmp.set_ylim(0, 110)
        self.ax_cmp.set_title("Algorithm Performance Comparison", color=TEXT, pad=12)
        legend = self.ax_cmp.legend(facecolor=CARD, edgecolor=CARD,
                                     labelcolor=TEXT, fontsize=9)
        self.fig_cmp.tight_layout()
        self.canvas_cmp.draw()
        self.nb.select(self.tab_compare)

# ─── ENTRY ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ParkinsonsApp()
    app.mainloop()
