"""BrainAligner: an offline Python desktop workspace for serial sections."""
from __future__ import annotations

import json
import logging
import math
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from image_io import EXTENSIONS, load_sources, make_proxy, render
from project import Project, export_project
from registration import Cancelled, check_cancel, compose, identity, register_series, rigid

ROOT = Path(__file__).resolve().parent


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('BrainAligner — Desktop')
        self.geometry(f'{min(1420,self.winfo_screenwidth()-60)}x{min(940,self.winfo_screenheight()-90)}')
        self.minsize(1000,680)
        self.project = None
        self.current = 0
        self.busy = False
        self.dirty = False
        self.cancel_event = threading.Event()
        self.events = queue.Queue()
        self.recording = False
        self.recorded = None
        self.correction = [0.,0.,0.]
        self.range_start = None
        self.range_end = None
        self.cropping = False
        self.crop_box = None
        self.crop_anchor = None
        self.pan = [0.,0.]
        self.zoom = 1.
        self.display_scale = 1.
        self.display_origin = (0.,0.)
        self._render_job = None
        self._flicker_phase = False
        self._closing = False
        self._disabled = []
        self.status = tk.StringVar(value='画像ファイル、または画像フォルダを選択してください')
        self.meta = tk.StringVar(value='1切片 = 1ファイル · 単一ページTIFF / PNG / JPEG')
        self.slice_number = tk.IntVar(value=1)
        self.slice_caption = tk.StringVar(value='画像未読み込み')
        self.channel = tk.StringVar(value='Gray')
        self.proxy = tk.StringVar(value='768')
        self.reference = tk.StringVar(value='Middle')
        self.background = tk.StringVar(value='white')
        self.view = tk.StringVar(value='After')
        self.overlay = tk.StringVar(value='None')
        self.opacity = tk.DoubleVar(value=.5)
        self.flicker = tk.BooleanVar(value=False)
        self.format = tk.StringVar(value='png')
        self.record_status = tk.StringVar(value='補正記録なし')
        self.range_status = tk.StringVar(value='適用範囲: 未設定')
        self.crop_status = tk.StringVar(value='切り抜き範囲: 未設定')
        self.qc_status = tk.StringVar(value='')
        self.crop_fields = [tk.IntVar(value=v) for v in (0,0,1,1)]
        self._build()
        self.bind('<KeyPress>',self._key)
        self.protocol('WM_DELETE_WINDOW',self._close)
        self.report_callback_exception = self._callback_exception
        self.after(80,self._poll)
        self.after(450,self._flicker_tick)

    def _build(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame',background='#f1f4f7')
        style.configure('TLabel',background='#f1f4f7',font=('Yu Gothic UI',10))
        style.configure('TLabelframe',background='#f1f4f7')
        style.configure('TLabelframe.Label',background='#f1f4f7',font=('Yu Gothic UI',10,'bold'))
        style.configure('TButton',padding=(9,5),font=('Yu Gothic UI',10))
        style.configure('Accent.TButton',background='#256d85',foreground='white')
        top = ttk.Frame(self,padding=12)
        top.pack(fill='x')
        ttk.Label(top,text='BrainAligner',font=('Segoe UI',20,'bold')).pack(side='left')
        ttk.Label(top,text='LOCAL DESKTOP',foreground='#256d85').pack(side='left',padx=18)
        for text,command in [('画像を選択',self.choose_files),('フォルダを選択',self.choose_folder),('Undo',self.undo)]:
            ttk.Button(top,text=text,command=command).pack(side='left',padx=3)
        ttk.Button(top,text='操作ガイド',command=self.help).pack(side='right')
        ttk.Label(self,textvariable=self.meta,padding=(14,0,10,8)).pack(fill='x')
        body = ttk.Frame(self)
        body.pack(fill='both',expand=True)
        left_shell = ttk.Frame(body,width=370)
        left_shell.pack(side='left',fill='y',padx=(10,6))
        self.sidebar = tk.Canvas(left_shell,width=354,highlightthickness=0,bg='#f1f4f7')
        scroll = ttk.Scrollbar(left_shell,orient='vertical',command=self.sidebar.yview)
        scroll.pack(side='right',fill='y')
        self.sidebar.pack(side='left',fill='both',expand=True)
        self.sidebar.configure(yscrollcommand=scroll.set)
        side = ttk.Frame(self.sidebar,padding=(4,0,10,12))
        window = self.sidebar.create_window(0,0,window=side,anchor='nw')
        side.bind('<Configure>',lambda e:self.sidebar.configure(scrollregion=self.sidebar.bbox('all')))
        self.sidebar.bind('<Configure>',lambda e:self.sidebar.itemconfigure(window,width=e.width))
        self.sidebar.bind('<MouseWheel>',lambda e:self.sidebar.yview_scroll(-int(e.delta/120),'units'))
        self.bind_all('<MouseWheel>',self._sidebar_wheel,add='+')

        group = self._group(side,'01  自動位置合わせ')
        self._combo(group,'チャンネル',self.channel,['Gray','Red','Green','Blue'])
        self._combo(group,'計算用画像の最大辺',self.proxy,['512','768','1024','1536'])
        self._combo(group,'基準切片',self.reference,['Middle','Current'])
        self._combo(group,'キャンバス背景',self.background,['white','black'],self.change_background)
        ttk.Label(group,text='Middle: 中央 / Current: 表示中の切片\n全切片を隣接ペアから順に整列',wraplength=310).pack(anchor='w',pady=4)
        ttk.Button(group,text='位置合わせを実行',style='Accent.TButton',command=self.align).pack(fill='x',pady=4)

        group = self._group(side,'02  手動補正')
        row = ttk.Frame(group); row.pack(fill='x')
        ttk.Button(row,text='記録開始',command=self.start_recording).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='記録終了',command=self.finish_recording).pack(side='left',expand=True,fill='x',padx=2)
        ttk.Button(row,text='取消',command=self.cancel_recording).pack(side='left')
        ttk.Label(group,text='W A S D / 矢印: 1 px  ·  Q E: 0.1°\nShiftを押しながら操作すると10倍',wraplength=310).pack(anchor='w',pady=5)
        ttk.Label(group,textvariable=self.record_status,wraplength=310).pack(anchor='w')
        row = ttk.Frame(group); row.pack(fill='x',pady=5)
        ttk.Button(row,text='現在をStartに設定',command=lambda:self.set_endpoint(True)).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='現在をEndに設定',command=lambda:self.set_endpoint(False)).pack(side='left',expand=True,fill='x')
        ttk.Label(group,textvariable=self.range_status,wraplength=310).pack(anchor='w')
        ttk.Button(group,text='指定範囲に移動・回転を適用',command=self.apply_recorded).pack(fill='x',pady=5)

        group = self._group(side,'03  出力範囲')
        row = ttk.Frame(group); row.pack(fill='x')
        ttk.Button(row,text='切り抜き選択',command=self.start_crop).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='選択枠を消す',command=self.clear_crop).pack(side='left',expand=True,fill='x')
        row = ttk.Frame(group); row.pack(fill='x',pady=4)
        for label,var in zip(['X','Y','W','H'],self.crop_fields):
            cell = ttk.Frame(row); cell.pack(side='left',expand=True,fill='x')
            ttk.Label(cell,text=label).pack(anchor='w')
            ttk.Entry(cell,textvariable=var,width=6).pack(fill='x',padx=(0,3))
        ttk.Label(group,textvariable=self.crop_status,wraplength=310).pack(anchor='w')
        row = ttk.Frame(group); row.pack(fill='x',pady=4)
        ttk.Button(row,text='切り抜きを適用',command=self.apply_crop).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='切り抜きを戻す',command=self.undo_crop).pack(side='left',expand=True,fill='x')
        ttk.Button(group,text='四辺に100 pxずつ余白を追加',command=self.expand).pack(fill='x')

        group = self._group(side,'04  保存・再開')
        self._combo(group,'画像形式',self.format,['png','jpg'])
        ttk.Label(group,text='出力は8-bit PNG / JPEGです。\n16-bit TIFFも8-bitに変換して処理します。',wraplength=310).pack(anchor='w',pady=4)
        row = ttk.Frame(group); row.pack(fill='x')
        ttk.Button(row,text='フォルダへ保存',command=lambda:self.export(False)).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='ZIPへ保存',command=lambda:self.export(True)).pack(side='left',expand=True,fill='x')
        row = ttk.Frame(group); row.pack(fill='x',pady=(4,0))
        ttk.Button(row,text='変換JSONを保存',command=self.save_transforms).pack(side='left',expand=True,fill='x')
        ttk.Button(row,text='変換JSONを読込',command=self.load_transforms).pack(side='left',expand=True,fill='x')

        right = ttk.Frame(body,padding=(5,0,10,0))
        right.pack(side='left',fill='both',expand=True)
        tools = ttk.Frame(right); tools.pack(fill='x',pady=(0,5))
        for value in ['Before','After']:
            ttk.Radiobutton(tools,text=value,variable=self.view,value=value,command=self.request_draw).pack(side='left',padx=3)
        ttk.Label(tools,text='重ね合わせ').pack(side='left',padx=(12,4))
        control = ttk.Combobox(tools,textvariable=self.overlay,values=['None','Previous','Next'],state='readonly',width=10)
        control.pack(side='left'); control.bind('<<ComboboxSelected>>',lambda e:self.request_draw())
        ttk.Checkbutton(tools,text='交互表示',variable=self.flicker,command=self.request_draw).pack(side='left',padx=6)
        ttk.Scale(tools,from_=0,to=1,variable=self.opacity,command=lambda _:self.request_draw(),length=100).pack(side='left')
        ttk.Button(tools,text='Fit',command=self.fit).pack(side='right')
        self.viewer = tk.Canvas(right,bg='#202a35',highlightthickness=0,takefocus=1,cursor='fleur')
        self.viewer.pack(fill='both',expand=True)
        self.viewer.bind('<Configure>',lambda e:self.request_draw())
        self.viewer.bind('<MouseWheel>',self._wheel)
        self.viewer.bind('<ButtonPress-1>',self._press)
        self.viewer.bind('<B1-Motion>',self._drag)
        self.viewer.bind('<ButtonRelease-1>',self._release)
        nav = ttk.Frame(right,padding=(0,6)); nav.pack(fill='x')
        ttk.Button(nav,text='◀ R',command=lambda:self.navigate(-1)).pack(side='left')
        self.slice_spin = ttk.Spinbox(nav,from_=1,to=1,textvariable=self.slice_number,width=6,command=self._spin)
        self.slice_spin.pack(side='left',padx=4)
        self.slice_spin.bind('<Return>',lambda e:self._spin())
        ttk.Button(nav,text='F ▶',command=lambda:self.navigate(1)).pack(side='left')
        ttk.Label(nav,textvariable=self.slice_caption).pack(side='left',padx=8)
        self.slider = ttk.Scale(right,from_=1,to=1,command=self._slide)
        self.slider.pack(fill='x')
        ttk.Label(right,textvariable=self.qc_status,wraplength=850,foreground='#965300').pack(fill='x',pady=4)
        self.table = ttk.Treeview(right,columns=('name','qc'),show='headings',height=4,selectmode='browse')
        self.table.heading('name',text='切片 / ファイル名'); self.table.heading('qc',text='NCC / 確認')
        self.table.column('name',width=420); self.table.column('qc',width=230)
        self.table.pack(fill='x')
        self.table.bind('<<TreeviewSelect>>',self._table_select)
        bottom = ttk.Frame(self,padding=10); bottom.pack(fill='x')
        self.cancel_button = ttk.Button(bottom,text='処理を中止',command=self.cancel_event.set,state='disabled')
        self.cancel_button.pack(side='right')
        self.progress = ttk.Progressbar(bottom,length=170,maximum=100)
        self.progress.pack(side='right',padx=10)
        ttk.Label(bottom,textvariable=self.status,wraplength=850).pack(side='left')
        self.request_draw()

    @staticmethod
    def _group(parent,title):
        frame = ttk.LabelFrame(parent,text=title,padding=8)
        frame.pack(fill='x',pady=(0,9))
        return frame

    def _combo(self,parent,label,variable,values,command=None):
        row = ttk.Frame(parent); row.pack(fill='x',pady=3)
        ttk.Label(row,text=label).pack(side='left')
        widget = ttk.Combobox(row,textvariable=variable,values=values,state='readonly',width=12)
        widget.pack(side='right')
        if command: widget.bind('<<ComboboxSelected>>',lambda e:command())

    def _sidebar_wheel(self,event):
        widget = event.widget
        while widget is not None:
            if widget is self.sidebar:
                return
            if str(widget).startswith(str(self.sidebar)+'.'):
                self.sidebar.yview_scroll(-int(event.delta/120),'units')
                return 'break'
            widget = getattr(widget,'master',None)

    def _callback_exception(self,kind,value,traceback):
        logging.error('GUI operation failed',exc_info=(kind,value,traceback))
        messagebox.showerror('操作エラー',str(value),parent=self)

    def ready(self,allow_record=False):
        if self.busy:
            return False
        if self.project is None:
            self.status.set('まず画像を読み込んでください')
            return False
        if self.recording and not allow_record:
            self.status.set('先に手動補正の記録を終了または取り消してください')
            return False
        return True

    def _discard_ok(self):
        return not self.dirty or messagebox.askyesno('未保存の作業','変換や画像の保存を確認してください。現在の作業を閉じますか？',parent=self)

    def choose_files(self):
        if self.busy or self.recording: return
        paths = filedialog.askopenfilenames(parent=self,title='画像を選択（1切片1ファイル・複数ページTIFF非対応）',
                                           filetypes=[('画像','*.png *.jpg *.jpeg *.tif *.tiff')])
        if paths: self._load(paths)

    def choose_folder(self):
        if self.busy or self.recording: return
        folder = filedialog.askdirectory(parent=self,title='画像フォルダを選択（サブフォルダも読み込み）')
        if folder:
            paths = [p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS]
            self._load(paths)

    def _load(self,paths):
        if not self._discard_ok(): return
        def done(result):
            sources,store = result
            self.project = Project(sources,store)
            self.project.background = self.background.get()
            self.current = len(sources)//2
            self.recorded = None
            self.record_status.set('補正記録なし')
            self.range_start = self.range_end = None
            self.range_status.set('適用範囲: 未設定')
            self.clear_crop()
            self.dirty = True
            self.meta.set(f'{len(sources)}切片 · 共通キャンバス {self.project.canvas[0]} × {self.project.canvas[1]} px · すべてローカルで処理')
            self.slice_spin.configure(to=len(sources))
            self.slider.configure(to=max(1,len(sources)))
            self._refresh_table()
            self.fit()
            self._sync_slice()
            self.status.set('読み込み完了。基準・チャンネルを選んで位置合わせを実行してください')
        self._task('画像を読み込み中',lambda progress,cancel:load_sources(paths,progress,cancel),done)

    def _task(self,title,work,done):
        if self.busy: return
        self.busy = True
        self.cancel_event.clear()
        self.status.set(title)
        self._disabled = []
        def disable(parent):
            for widget in parent.winfo_children():
                if isinstance(widget,ttk.Widget) and not isinstance(widget,(ttk.Frame,ttk.LabelFrame,ttk.Label,ttk.Progressbar)) and widget is not self.cancel_button:
                    state = widget.state()
                    self._disabled.append((widget,state))
                    widget.state(['disabled'])
                disable(widget)
        disable(self)
        self.cancel_button.state(['!disabled'])
        def progress(current,total,text):
            self.events.put(('progress',current,total,text))
        def run():
            try:
                value = work(progress,self.cancel_event)
                self.events.put(('done',done,value))
            except Cancelled:
                self.events.put(('cancelled',))
            except Exception as exc:
                logging.exception('Background operation failed')
                self.events.put(('error',str(exc)))
        threading.Thread(target=run,daemon=True).start()

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0]=='progress':
                    _,current,total,text = event
                    self.progress['value'] = current/max(1,total)*100
                    self.status.set(f'{current}/{total} · {text}')
                else:
                    self.busy = False
                    self.cancel_button.state(['disabled'])
                    for widget,state in self._disabled:
                        widget.state(['!disabled'])
                        widget.state(state)
                    self._disabled = []
                    self.progress['value'] = 0
                    if self._closing:
                        self.destroy()
                        return
                    if event[0]=='done':
                        event[1](event[2])
                    elif event[0]=='cancelled':
                        self.status.set('処理を中止しました。フォルダ保存中の場合、未完了の出力にはINCOMPLETE.txtが残ります')
                    else:
                        self.status.set(event[1])
                        messagebox.showerror('処理できません',event[1],parent=self)
        except queue.Empty:
            pass
        self.after(80,self._poll)

    def align(self):
        if not self.ready(): return
        if len(self.project.sources)<2:
            self.status.set('自動整列には2枚以上の画像が必要です'); return
        project = self.project
        reference = len(project.sources)//2 if self.reference.get()=='Middle' else self.current
        channel,limit,background = self.channel.get(),int(self.proxy.get()),project.background
        def work(progress,cancel):
            return register_series(len(project.sources),reference,
                lambda i:make_proxy(project.store,i,project.canvas,limit,channel,background),progress,cancel)
        def done(value):
            project.checkpoint()
            project.automatic,project.qc = value
            project.settings = {'reference':reference,'channel':channel,'proxyLimit':limit,'algorithm':'rigid-multiresolution-ncc'}
            self.dirty = True
            self.view.set('After')
            self._refresh_table()
            self._sync_slice()
            warnings = sum(bool(q and q['warning']) for q in project.qc)
            self.status.set(f'位置合わせ完了 · 要確認 {warnings}切片。全切片を重ね合わせ表示で確認してください')
        self._task('剛体位置合わせを計算中',work,done)

    def _refresh_table(self):
        self.table.delete(*self.table.get_children())
        if not self.project: return
        for index,source in enumerate(self.project.sources):
            qc = self.project.qc[index]
            text = f"{qc['score']:.3f}"+(' · 要確認' if qc['warning'] else '') if qc else '—'
            self.table.insert('','end',iid=str(index),values=(f'{index+1} · {source.name}',text))

    def _sync_slice(self):
        if not self.project: return
        self.slice_number.set(self.current+1)
        self.slider.set(self.current+1)
        source = self.project.sources[self.current]
        self.slice_caption.set(f'/ {len(self.project.sources)} · {source.name}')
        qc = self.project.qc[self.current]
        self.qc_status.set(f"NCC {qc['score']:.3f}"+(' · '+qc['warning'] if qc['warning'] else '') if qc else '未整列')
        self.table.selection_set(str(self.current))
        self.table.see(str(self.current))
        self.request_draw()

    def navigate(self,delta):
        if not self.ready(): return
        self.set_slice(self.current+delta)

    def set_slice(self,index):
        if not self.ready(): return
        self.current = max(0,min(len(self.project.sources)-1,int(index)))
        self._sync_slice()

    def _spin(self):
        if self.ready(): self.set_slice(self.slice_number.get()-1)

    def _slide(self,value):
        if not self.project or self.busy or self.recording: return
        index = max(0,min(len(self.project.sources)-1,round(float(value))-1))
        if index!=self.current: self.set_slice(index)

    def _table_select(self,event):
        selected = self.table.selection()
        if selected and int(selected[0])!=self.current:
            self.set_slice(int(selected[0]))

    def change_background(self):
        if not self.ready():
            if self.project: self.background.set(self.project.background)
            return
        if self.background.get()!=self.project.background:
            self.project.checkpoint()
            self.project.background = self.background.get()
            self.dirty = True
            self.request_draw()

    def start_recording(self):
        if not self.ready(): return
        self.recording = True
        self.recorded = None
        self.correction = [0.,0.,0.]
        x,y,w,h = self.project.viewport
        self.record_pivot = (x+w/2,y+h/2)
        self.cropping = False
        self.view.set('After')
        self.record_status.set('記録中: ΔX 0 px / ΔY 0 px / 回転 0°')
        self.viewer.focus_set()
        self.request_draw()

    def finish_recording(self):
        if not self.recording or self.busy: return
        self.recorded = rigid(*self.correction,*self.record_pivot)
        self.recording = False
        self.range_start = self.range_end = self.current
        self._range_label()
        self.record_status.set(f'記録済: ΔX {self.correction[0]:g} / ΔY {self.correction[1]:g} px / {self.correction[2]:g}°')
        self.status.set('補正を記録しました。Start / Endを確認して指定範囲へ適用してください')
        self.request_draw()

    def cancel_recording(self):
        if self.busy: return
        self.recording = False
        self.recorded = None
        self.record_status.set('補正記録なし')
        self.request_draw()

    def set_endpoint(self,start):
        if not self.ready(): return
        if self.recorded is None:
            self.status.set('先に補正を記録・終了してください'); return
        if start: self.range_start = self.current
        else: self.range_end = self.current
        self._range_label()

    def _range_label(self):
        self.range_status.set(f'Start {self.range_start+1} / End {self.range_end+1} · 両端を含む')

    def apply_recorded(self):
        if not self.ready(): return
        if self.recorded is None or self.range_start is None or self.range_end is None:
            self.status.set('補正記録とStart / Endを設定してください'); return
        self.project.apply_manual(self.recorded,self.range_start,self.range_end)
        self.dirty = True
        self.status.set(f'切片 {min(self.range_start,self.range_end)+1}–{max(self.range_start,self.range_end)+1} に補正を1回適用しました')
        self.request_draw()

    def undo(self):
        if self.ready() and self.project.undo():
            self.dirty = True
            self.background.set(self.project.background)
            self.clear_crop()
            self._refresh_table(); self._sync_slice()
            self.status.set('直前の変更を取り消しました')

    def start_crop(self):
        if not self.ready(): return
        self.cropping = True
        self.viewer.configure(cursor='crosshair')
        self.status.set('画像上をドラッグして切り抜き範囲を選んでください。X/Y/W/Hでも指定できます')

    def clear_crop(self):
        self.cropping = False
        self.crop_box = None
        self.crop_anchor = None
        self.crop_status.set('切り抜き範囲: 未設定')
        self.viewer.configure(cursor='fleur')
        self.request_draw()

    def apply_crop(self):
        if not self.ready(): return
        values = tuple(var.get() for var in self.crop_fields)
        self.project.crop(values)
        self.dirty = True
        self.clear_crop(); self.fit()
        self.status.set(f'出力範囲: {values}')

    def undo_crop(self):
        if self.ready() and self.project.undo_crop():
            self.dirty = True
            self.clear_crop(); self.fit()

    def expand(self):
        if self.ready():
            self.project.expand()
            self.dirty = True
            self.clear_crop(); self.fit()

    def fit(self):
        self.zoom = 1.
        self.pan = [0.,0.]
        self.request_draw()

    def request_draw(self):
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(25,self._draw)

    def _draw(self):
        self._render_job = None
        self.viewer.delete('all')
        cw,ch = max(1,self.viewer.winfo_width()),max(1,self.viewer.winfo_height())
        if not self.project:
            self.viewer.create_text(cw/2,ch/2,text='画像を選択して開始\n\nPNG / JPEG / 単一ページTIFF',fill='#d1dae4',font=('Yu Gothic UI',15),justify='center')
            return
        project = self.project
        vx,vy,vw,vh = project.viewport
        scale = min((cw-30)/vw,(ch-30)/vh)*self.zoom
        scale = max(.001,scale)
        # Render only the visible viewport, keeping zoom memory bounded.
        origin = ((cw-vw*scale)/2+self.pan[0],(ch-vh*scale)/2+self.pan[1])
        self.display_origin,self.display_scale = origin,scale
        visible = (vx-origin[0]/scale,vy-origin[1]/scale,cw/scale,ch/scale)
        if self.recording: self.view.set('After')
        matrix = identity() if self.view.get()=='Before' else project.final(self.current)
        if self.recording:
            matrix = compose(rigid(*self.correction,*self.record_pivot),matrix)
        rgb = render(project.store,self.current,project.canvas,matrix,visible,project.background,scale)
        neighbor = self.current+(-1 if self.overlay.get()=='Previous' else 1) if self.overlay.get()!='None' else -1
        if 0<=neighbor<len(project.sources):
            other = render(project.store,neighbor,project.canvas,project.final(neighbor),visible,project.background,scale)
            if self.flicker.get():
                if self._flicker_phase: rgb = other
            else:
                alpha = self.opacity.get()
                rgb = np.rint(rgb.astype(np.float32)*(1-alpha)+other.astype(np.float32)*alpha).astype(np.uint8)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.viewer.create_image(0,0,image=self._photo,anchor='nw')
        self.viewer.create_rectangle(origin[0],origin[1],origin[0]+vw*scale,origin[1]+vh*scale,outline='#5d7489')
        self.viewer.create_text(12,12,anchor='nw',text=f'{self.view.get().upper()} · SLICE {self.current+1} · {scale*100:.1f}%',fill='#46d5d0',font=('Segoe UI',11,'bold'))
        if self.crop_box:
            x,y,w,h = self.crop_box
            self.viewer.create_rectangle(origin[0]+(x-vx)*scale,origin[1]+(y-vy)*scale,
                                         origin[0]+(x+w-vx)*scale,origin[1]+(y+h-vy)*scale,
                                         outline='#efb843',width=2,dash=(5,3))

    def _flicker_tick(self):
        if self.flicker.get() and self.project and not self.busy:
            self._flicker_phase = not self._flicker_phase
            self.request_draw()
        self.after(450,self._flicker_tick)

    def _wheel(self,event):
        if not self.project or self.busy: return 'break'
        if event.state & 0x4:
            cx,cy = self.viewer.winfo_width()/2,self.viewer.winfo_height()/2
            factor = 1.15 if event.delta>0 else 1/1.15
            new_zoom = max(.2,min(30.,self.zoom*factor))
            ratio = new_zoom/self.zoom
            self.pan = [(event.x-cx)-(event.x-cx-self.pan[0])*ratio,
                        (event.y-cy)-(event.y-cy-self.pan[1])*ratio]
            self.zoom = new_zoom
            self.request_draw()
        elif not self.recording:
            self.navigate(1 if event.delta<0 else -1)
        return 'break'

    def _world(self,event):
        vx,vy,_,_ = self.project.viewport
        return (vx+(event.x-self.display_origin[0])/self.display_scale,
                vy+(event.y-self.display_origin[1])/self.display_scale)

    def _press(self,event):
        self.viewer.focus_set()
        if not self.project or self.busy: return
        self._drag_origin = (event.x,event.y,self.pan.copy())
        if self.cropping:
            self.crop_anchor = self._world(event)

    def _drag(self,event):
        if not self.project or self.busy or not hasattr(self,'_drag_origin'): return
        if self.cropping and self.crop_anchor:
            x,y = self.crop_anchor
            xx,yy = self._world(event)
            self.crop_box = (round(min(x,xx)),round(min(y,yy)),max(1,round(abs(xx-x))),max(1,round(abs(yy-y))))
            for var,value in zip(self.crop_fields,self.crop_box): var.set(value)
            self.crop_status.set(f'選択範囲: {self.crop_box}')
        else:
            x,y,pan = self._drag_origin
            self.pan = [pan[0]+event.x-x,pan[1]+event.y-y]
        self.request_draw()

    def _release(self,event):
        if hasattr(self,'_drag_origin'): del self._drag_origin

    def _key(self,event):
        widget = self.focus_get()
        if isinstance(widget,(ttk.Entry,ttk.Spinbox,ttk.Combobox,tk.Entry,tk.Text)) or self.busy: return
        key = event.keysym.lower()
        if event.state & 0x4 and key=='z':
            self.undo(); return 'break'
        if not self.project: return
        if self.recording:
            mapping = {'a':(0,-1),'left':(0,-1),'d':(0,1),'right':(0,1),
                       'w':(1,-1),'up':(1,-1),'s':(1,1),'down':(1,1),'q':(2,-.1),'e':(2,.1)}
            if key in mapping:
                axis,amount = mapping[key]
                self.correction[axis] += amount*(10 if event.state & 1 else 1)
                self.record_status.set(f'記録中: ΔX {self.correction[0]:g} / ΔY {self.correction[1]:g} px / {self.correction[2]:g}°')
                self.request_draw()
                return 'break'
        elif key in ('f','right','next'):
            self.navigate(1); return 'break'
        elif key in ('r','left','prior'):
            self.navigate(-1); return 'break'

    def save_transforms(self):
        if not self.ready(): return
        path = filedialog.asksaveasfilename(parent=self,title='変換情報を保存',initialfile='transforms.json',defaultextension='.json',filetypes=[('JSON','*.json')])
        if path:
            target = Path(path)
            temporary = target.with_name(target.name+'.tmp')
            temporary.write_text(json.dumps(self.project.to_dict(),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            temporary.replace(target)
            self.dirty = False
            self.status.set(f'変換情報を保存: {path}')

    def load_transforms(self):
        if not self.ready(): return
        path = filedialog.askopenfilename(parent=self,title='変換情報を読み込む',filetypes=[('JSON','*.json')])
        if path:
            self.project.load_dict(json.loads(Path(path).read_text(encoding='utf-8-sig')))
            self.background.set(self.project.background)
            self.dirty = True
            self.clear_crop(); self.fit(); self._refresh_table(); self._sync_slice()
            self.status.set('変換情報を読み込みました。対応する切片が同じ順序で並んでいるか確認してください')

    def export(self,as_zip):
        if not self.ready(): return
        directory = filedialog.askdirectory(parent=self,title='保存先フォルダ（新しい出力フォルダ/ZIPを作成します）')
        if not directory: return
        project,format = self.project,self.format.get()
        def done(target):
            self.dirty = False
            self.status.set(f'保存完了: {target}')
            messagebox.showinfo('保存完了',str(target),parent=self)
        self._task('画像を書き出し中',lambda progress,cancel:export_project(project,directory,format,as_zip,progress,cancel),done)

    def help(self):
        messagebox.showinfo('操作ガイド',
            '1. 画像またはフォルダを選択します。1切片1ファイルです。\n'
            '2. Gray / Red / Green / Blueと基準切片を選び、自動位置合わせを実行します。\n'
            '3. Previous / Nextの重ね合わせ、透明度、交互表示で全切片を確認します。\n'
            '4. 手動補正は「記録開始」→ WASD / QE →「記録終了」→ Start / End設定 → 適用の順です。\n'
            '5. 切り抜きや余白を設定し、PNG / JPEGとtransforms.jsonを保存します。\n\n'
            'F / R: 切片移動。Ctrl+ホイール: 拡大縮小。ドラッグ: 表示移動。Ctrl+Z: Undo。\n'
            '16-bit TIFFは0–65535を0–255へ変換して使用し、出力も8-bitになります。\n'
            '複数ページTIFF、OME-TIFFは非対応です。\n'
            '作業は自動保存されません。再開には元画像と変換JSONを使用します。',parent=self)

    def _close(self):
        if self.busy:
            if messagebox.askyesno('処理中','処理を中止して終了しますか？',parent=self):
                self._closing = True
                self.cancel_event.set()
                self.status.set('処理の終了を待っています…')
            return
        if self._discard_ok(): self.destroy()


def main():
    logging.basicConfig(filename=ROOT/'brainaligner.log',level=logging.WARNING,encoding='utf-8',
                        format='%(asctime)s %(levelname)s %(message)s')
    application = Application()
    application.mainloop()


if __name__=='__main__':
    main()
