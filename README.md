# onnx-toolkit-gui
A lightweight GUI tool for analyzing ONNX models    It provides a simple Tkinter interface to run model profiling, view raw outputs, and export results as CSV.  

# ONNX Model Profiler (Tkinter GUI)

一个基于 **Tkinter** 的可视化工具，用来调用 `onnx_tool.model_profile` 对 **ONNX 模型**进行分析，展示并导出层级统计（如 **MACs / 参数量 / 内存开销 / 输入输出形状** 等）。
支持在界面中查看 **原始文本输出** 与 **表格视图**，并一键 **保存为 CSV**。

---

## ✨ 功能特性

- 选择 `.onnx` 文件，一键分析（后台捕获 `onnx_tool` 控制台输出）
- 左侧展示 **原始文本**，右侧展示 **解析后的表格**
- 支持 **导出 CSV**（UTF-8 带 BOM，Excel 可直接打开）
- 支持 **复制文本**、**清空**、**再次分析**
- 自动将常见数值列转换为数值类型（去逗号、百分号）

---

## 🧰 环境要求

- Python 3.8+
- 依赖：
  - `onnx-tool`
  - `pandas`

安装依赖：

```bash
pip install onnx-tool pandas

