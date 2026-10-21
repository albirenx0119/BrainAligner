"""Double-click launcher with a startup error dialog and local diagnostics."""
from pathlib import Path
import traceback

if __name__ == '__main__':
    try:
        from app import main
        main()
    except Exception:
        details = traceback.format_exc()
        target = Path(__file__).resolve().parent/'startup-error.log'
        target.write_text(details,encoding='utf-8')
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('BrainAligner 起動エラー',f'起動できませんでした。\n{target}\n\n{details[-2000:]}',parent=root)
        root.destroy()
