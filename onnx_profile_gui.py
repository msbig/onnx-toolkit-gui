#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import sys
import threading
from contextlib import redirect_stdout
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# 第三方依赖
try:
    import pandas as pd
except Exception as e:
    pd = None

try:
    import onnx_tool
except Exception as e:
    onnx_tool = None

APP_TITLE = "ONNX Model Profiler (onnx_tool GUI)"
DEFAULT_CSV_NAME = "model_profile.csv"

def resource_path(rel_path: str):
    """兼容 PyInstaller 的资源路径（开发环境 & 冻结后的 exe）"""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, rel_path)
def parse_table_from_text(text: str):
    """
    从 onnx_tool.model_profile 的 stdout 文本中提取表格，返回 (df, header_idx)
    若 pandas 不可用，抛异常（我们要求安装 pandas）
    """
    if pd is None:
        raise RuntimeError("需要 pandas 才能解析表格，请先安装：pip install pandas")

    start = text.find("Name")
    if start == -1:
        raise RuntimeError("未在输出中找到表头（'Name'），请检查 onnx_tool 输出格式。")
    table_text = text[start:]

    # 简单切断：遇到 3 个换行作为“表格结束”的分界（和你现有脚本一致）
    table_text = table_text.split("\n\n\n")[0]

    df = pd.read_fwf(io.StringIO(table_text))

    # 尝试把数值列转换为数值
    def to_number(s):
        s = str(s).replace(",", "").replace("%", "").strip()
        if s in ("", "_", "None"):
            return None
        try:
            return float(s)
        except Exception:
            return s

    for col in ["Forward_MACs", "FPercent", "Memory", "MPercent", "Params", "PPercent"]:
        if col in df.columns:
            df[col] = df[col].map(to_number)

    return df, start


class OnnxProfilerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x700")

        self.model_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.df = None
        self.raw_text = ""

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="ONNX 模型路径：").pack(side="left")
        self.entry = ttk.Entry(top, textvariable=self.model_path_var)
        self.entry.pack(side="left", fill="x", expand=True, padx=6)

        ttk.Button(top, text="选择文件", command=self.on_browse).pack(side="left", padx=4)
        ttk.Button(top, text="开始分析", command=self.on_profile).pack(side="left", padx=4)
        ttk.Button(top, text="保存为 CSV", command=self.on_save_csv).pack(side="left", padx=4)
        ttk.Button(top, text="复制文本", command=self.on_copy_text).pack(side="left", padx=4)
        ttk.Button(top, text="清空", command=self.on_clear).pack(side="left", padx=4)

        # 分隔面板：左侧文本、右侧表格
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 左：原始输出
        left_frame = ttk.Frame(paned)
        ttk.Label(left_frame, text="原始输出（stdout）").pack(anchor="w")
        self.text_area = ScrolledText(left_frame, wrap="none", height=20)
        self.text_area.pack(fill="both", expand=True, pady=(4, 0))
        paned.add(left_frame, weight=1)

        # 右：表格（Treeview）
        right_frame = ttk.Frame(paned)
        ttk.Label(right_frame, text="解析后的表格").pack(anchor="w")

        self.tree = ttk.Treeview(right_frame, show="headings")
        self.tree_scroll_x = ttk.Scrollbar(right_frame, orient="horizontal", command=self.tree.xview)
        self.tree_scroll_y = ttk.Scrollbar(right_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(xscrollcommand=self.tree_scroll_x.set, yscrollcommand=self.tree_scroll_y.set)

        self.tree.pack(fill="both", expand=True, pady=(4, 0))
        self.tree_scroll_x.pack(fill="x")
        self.tree_scroll_y.pack(side="right", fill="y")
        paned.add(right_frame, weight=1)

        # 底部状态栏
        status_bar = ttk.Frame(self, padding=(10, 0, 10, 10))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")

        # 提示依赖
        if onnx_tool is None:
            self._error(
                "未找到 onnx_tool。请安装：pip install onnx-tool\n"
                "安装后重新运行本程序。"
            )
        if pd is None:
            self._error(
                "未找到 pandas。请安装：pip install pandas\n"
                "安装后重新运行本程序。"
            )

    # --------------- Helpers ---------------
    def _info(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    def _error(self, msg):
        self.status_var.set("错误")
        messagebox.showerror("错误", msg)

    def _set_tree_from_df(self, df):
        # 清空旧列
        for c in self.tree.get_children():
            self.tree.delete(c)
        self.tree["columns"] = []
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")

        # 设置新列
        cols = list(df.columns)
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=max(80, min(260, int(len(col) * 10))))

        # 插入数据
        for _, row in df.iterrows():
            values = [row[c] for c in cols]
            self.tree.insert("", "end", values=values)

    def _run_profile_worker(self, model_path):
        try:
            self._info("分析中…（请稍候）")
            buf = io.StringIO()
            with redirect_stdout(buf):
                # 关键：直接复用你的 onnx_tool 调用
                onnx_tool.model_profile(model_path)
            text = buf.getvalue()
            self.raw_text = text

            # 展示原始输出
            self.text_area.delete("1.0", "end")
            self.text_area.insert("end", text)

            # 解析表格
            try:
                df, _ = parse_table_from_text(text)
                self.df = df
                self._set_tree_from_df(df)
                self._info("分析完成")
            except Exception as e:
                self.df = None
                self._info("分析完成（未能解析表格）")
                messagebox.showwarning("提示", f"已获取原始输出，但解析表格失败：\n{e}")
        except Exception as e:
            self._error(f"分析失败：\n{e}")

    # --------------- Callbacks ---------------
    def on_browse(self):
        path = filedialog.askopenfilename(
            title="选择 ONNX 模型文件",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")]
        )
        if path:
            self.model_path_var.set(path)

    def on_profile(self):
        if onnx_tool is None:
            self._error("未安装 onnx_tool：pip install onnx-tool")
            return
        if pd is None:
            self._error("未安装 pandas：pip install pandas")
            return
        model_path = self.model_path_var.get().strip()
        if not model_path or not os.path.isfile(model_path):
            self._error("请先选择有效的 .onnx 文件。")
            return

        # 开线程，避免阻塞 UI
        t = threading.Thread(target=self._run_profile_worker, args=(model_path,), daemon=True)
        t.start()

    def on_save_csv(self):
        if self.df is None or pd is None:
            self._error("没有可保存的数据表，或未安装 pandas。请先完成分析。")
            return
        # 默认保存到模型同目录
        init_dir = os.path.dirname(self.model_path_var.get().strip()) or "."
        init_name = DEFAULT_CSV_NAME
        save_path = filedialog.asksaveasfilename(
            title="保存为 CSV",
            defaultextension=".csv",
            initialdir=init_dir,
            initialfile=init_name,
            filetypes=[("CSV 文件", "*.csv")]
        )
        if not save_path:
            return
        try:
            self.df.to_csv(save_path, index=False, encoding="utf-8-sig")
            self._info(f"已保存：{save_path}")
            messagebox.showinfo("成功", f"已保存为 CSV：\n{save_path}")
        except Exception as e:
            self._error(f"保存失败：\n{e}")

    def on_copy_text(self):
        if not self.raw_text:
            self._error("当前没有可复制的文本，请先分析。")
            return
        self.clipboard_clear()
        self.clipboard_append(self.raw_text)
        self._info("原始文本已复制到剪贴板")

    def on_clear(self):
        self.text_area.delete("1.0", "end")
        for c in self.tree.get_children():
            self.tree.delete(c)
        self.tree["columns"] = []
        self.df = None
        self.raw_text = ""
        self._info("已清空")

def main():
    app = OnnxProfilerGUI()
    try:
        icon_file = resource_path("app.ico")
        if os.path.exists(icon_file):
            app.iconbitmap(icon_file)
    except Exception:
        pass
    app.mainloop()


if __name__ == "__main__":
    main()
