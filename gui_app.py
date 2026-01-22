import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import queue
from backend import ConfigManager, PDFHandler, TextProcessor, TTSManager, VideoManager, ModelScanner

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF to Video Pipeline")
        self.geometry("1000x800")
        
        # Load Config
        try:
            self.config = ConfigManager.load_config()
        except Exception as e:
            messagebox.showerror("Config Error", str(e))
            self.destroy()
            return

        # Shared State 
        # Stores data passed between steps: 'pdf_path', 'raw_text', 'clean_text', 'wav_path', 'mp3_path'
        self.state = {} 
        
        # Initialize TTS Manager (Lazy load later, but init object)
        self.tts_manager = TTSManager(self.config)

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Tabs
        self.tab_pdf = PDFFrame(self.notebook, self)
        self.tab_clean = TextCleanerFrame(self.notebook, self)
        self.tab_tts = TTSFrame(self.notebook, self)
        self.tab_video = VideoFrame(self.notebook, self)
        
        self.notebook.add(self.tab_pdf, text="1. Extraction PDF")
        self.notebook.add(self.tab_clean, text="2. Nettoyage Texte")
        self.notebook.add(self.tab_tts, text="3. Synthèse Vocale")
        self.notebook.add(self.tab_video, text="4. Rendu Vidéo")

        # Events to update tabs when switching
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        selected_tab = self.notebook.select()
        tab_object = self.notebook.nametowidget(selected_tab)
        if hasattr(tab_object, 'update_state'):
            tab_object.update_state()

class PDFFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.LabelFrame(self, text="Sélection du Document", padding=10)
        frame.pack(fill="x", padx=10, pady=5)

        self.lbl_file = ttk.Label(frame, text="Aucun fichier sélectionné")
        self.lbl_file.pack(side="left", fill="x", expand=True)
        
        btn_browse = ttk.Button(frame, text="Parcourir", command=self.browse_pdf)
        btn_browse.pack(side="right")

        frame_preview = ttk.LabelFrame(self, text="Aperçu du Texte Extrait", padding=10)
        frame_preview.pack(fill="both", expand=True, padx=10, pady=5)

        self.txt_preview = scrolledtext.ScrolledText(frame_preview, height=20)
        self.txt_preview.pack(fill="both", expand=True)

        frame_btns = ttk.Frame(self)
        frame_btns.pack(pady=10)
        
        btn_extract = ttk.Button(frame_btns, text="Extraire le Texte", command=self.extract)
        btn_extract.pack(side="left", padx=5)
        
        self.btn_export = ttk.Button(frame_btns, text="Aller au Nettoyage >", command=self.go_to_cleaning, state="disabled")
        self.btn_export.pack(side="left", padx=5)

    def browse_pdf(self):
        initial_dir = self.app.config['paths']['pdf_input']
        if not os.path.exists(initial_dir): os.makedirs(initial_dir)
        filename = filedialog.askopenfilename(initialdir=initial_dir, filetypes=[("Document Files", "*.pdf *.epub"), ("PDF Files", "*.pdf"), ("EPUB Files", "*.epub")])
        if filename:
            self.app.state['pdf_path'] = filename
            self.lbl_file.config(text=filename)
            self.btn_export.config(state="disabled")

    def extract(self):
        path = self.app.state.get('pdf_path')
        if not path:
            messagebox.showwarning("Attention", "Veuillez sélectionner un fichier.")
            return
        
        # Using default margins (50 units approx) to skip header/footer
        text = PDFHandler.extract_text(path)
        if text:
            self.app.state['raw_text'] = text
            self.app.state['base_name'] = os.path.splitext(os.path.basename(path))[0]
            self.txt_preview.delete("1.0", tk.END)
            self.txt_preview.insert(tk.END, text)
            self.btn_export.config(state="normal")
            messagebox.showinfo("Succès", "Texte extrait (En-têtes et pieds de page filtrés).")
        else:
            messagebox.showerror("Erreur", "Échec de l'extraction.")

    def go_to_cleaning(self):
        self.app.notebook.select(1)

class TextCleanerFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        # -- Search Bar --
        frame_search = ttk.Frame(self)
        frame_search.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(frame_search, text="Rechercher :").pack(side="left")
        self.entry_search = ttk.Entry(frame_search)
        self.entry_search.pack(side="left", padx=5, fill="x", expand=True)
        self.entry_search.bind("<Return>", lambda e: self.on_search_next())
        
        btn_prev = ttk.Button(frame_search, text="<", width=3, command=self.on_search_prev)
        btn_prev.pack(side="left", padx=2)
        
        btn_next = ttk.Button(frame_search, text=">", width=3, command=self.on_search_next)
        btn_next.pack(side="left", padx=2)
        
        self.lbl_search_status = ttk.Label(frame_search, text="", foreground="gray")
        self.lbl_search_status.pack(side="left", padx=5)

        # -- Editor --
        ttk.Label(self, text="Éditez et nettoyez le texte avant la synthèse :").pack(anchor="w", padx=10, pady=5)
        
        self.txt_editor = scrolledtext.ScrolledText(self)
        self.txt_editor.pack(fill="both", expand=True, padx=10, pady=5)
        # Configure search highlight tag
        self.txt_editor.tag_config('search_highlight', background='yellow', foreground='black')

        btn_load = ttk.Button(self, text="Charger Fichier Texte", command=self.load_text_file)
        btn_load.pack(pady=5)

        frame_tools = ttk.LabelFrame(self, text="Outils de Nettoyage", padding=5)
        frame_tools.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(frame_tools, text="1. Nettoyer Caractères Spéciaux", command=self.clean_chars).pack(side="left", padx=5)
        ttk.Button(frame_tools, text="2. Recomposer Paragraphes", command=self.unwrap_lines).pack(side="left", padx=5)
        ttk.Button(frame_tools, text="3. Corriger Espaces", command=self.fix_spaces).pack(side="left", padx=5)
        ttk.Button(frame_tools, text="Tout (Auto)", command=self.auto_clean).pack(side="right", padx=5)
        
        btn_validate = ttk.Button(self, text="Valider pour Synthèse", command=self.validate)
        btn_validate.pack(pady=10)

    def on_search_next(self):
        self.perform_search(direction="next")
        
    def on_search_prev(self):
        self.perform_search(direction="prev")
        
    def perform_search(self, direction="next"):
        query = self.entry_search.get()
        if not query:
            self.lbl_search_status.config(text="")
            return
            
        # Clean highlight
        self.txt_editor.tag_remove('search_highlight', '1.0', tk.END)
        
        start_pos = '1.0'
        if direction == "next":
            # Search from cursor insert position
            start_pos = self.txt_editor.index(tk.INSERT)
            # If nothing selected or at end, might need to adjust logic, but default search goes forward
            pos = self.txt_editor.search(query, start_pos, stopindex=tk.END, nocase=True)
            if not pos:
                # Wrap around
                pos = self.txt_editor.search(query, '1.0', stopindex=tk.END, nocase=True)
        else:
            # Prev
            start_pos = self.txt_editor.index(tk.INSERT)
            # search backwards
            pos = self.txt_editor.search(query, start_pos, stopindex='1.0', backwards=True, nocase=True)
            if not pos:
                # Wrap around (search from end)
                pos = self.txt_editor.search(query, tk.END, stopindex='1.0', backwards=True, nocase=True)

        if pos:
            # Match found
            # Calculate end pos
            end_pos = f"{pos}+{len(query)}c"
            self.txt_editor.tag_add('search_highlight', pos, end_pos)
            self.txt_editor.see(pos)
            self.txt_editor.mark_set(tk.INSERT, end_pos if direction == "next" else pos)
            self.txt_editor.focus_set()
            self.lbl_search_status.config(text="Trouvé")
        else:
            self.lbl_search_status.config(text="Introuvable")

    def update_state(self):
        # Load raw text if not already loaded or if cleaner is empty
        if 'raw_text' in self.app.state and not self.txt_editor.get("1.0", tk.END).strip():
            self.txt_editor.insert(tk.END, self.app.state['raw_text'])

    def load_text_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filename:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    text = f.read()
                self.txt_editor.delete("1.0", tk.END)
                self.txt_editor.insert(tk.END, text)
                
                # If no base_name yet (skipped PDF step), use filename
                if 'base_name' not in self.app.state:
                    self.app.state['base_name'] = os.path.splitext(os.path.basename(filename))[0]
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur de lecture : {e}")

    def apply_cleaner(self, func):
        text = self.txt_editor.get("1.0", tk.END)
        cleaned = func(text)
        self.txt_editor.delete("1.0", tk.END)
        self.txt_editor.insert(tk.END, cleaned)

    def clean_chars(self):
        self.apply_cleaner(TextProcessor.clean_special_chars)
        
    def unwrap_lines(self):
        self.apply_cleaner(TextProcessor.unwrap_paragraphs)
        
    def fix_spaces(self):
        self.apply_cleaner(TextProcessor.fix_whitespaces)

    def auto_clean(self):
        self.apply_cleaner(TextProcessor.clean_text)

    def validate(self):
        self.app.state['clean_text'] = self.txt_editor.get("1.0", tk.END).strip()
        if not self.app.state['clean_text']:
            messagebox.showwarning("Attention", "Le texte est vide.")
            return
            
        # Save text to file
        base_name = self.app.state.get('base_name', 'output')
        output_dir = self.app.config['paths']['audio_output']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        text_path = os.path.join(output_dir, f"{base_name}.txt")
        try:
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(self.app.state['clean_text'])
            messagebox.showinfo("Validé", f"Texte validé et sauvegardé dans :\n{text_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde du texte : {e}")
            
        self.app.notebook.select(2)

class TTSFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        info_frame = ttk.LabelFrame(self, text="Configuration du Modèle", padding=10)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(info_frame, text="Sélectionner un modèle entraîné :").pack(anchor="w")
        
        self.cbo_models = ttk.Combobox(info_frame, state="readonly")
        self.cbo_models.pack(fill="x", pady=5)
        self.cbo_models.bind("<<ComboboxSelected>>", self.on_model_select)
        
        self.models_data = []
        self.refresh_models()
        
        self.lbl_status = ttk.Label(self, text="Status: Prêt", foreground="blue")
        self.lbl_status.pack(pady=10)
        
        self.progress = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)
        
        frame_btns = ttk.Frame(self)
        frame_btns.pack(pady=20)
        
        self.btn_generate = ttk.Button(frame_btns, text="Lancer la Synthèse", command=self.start_synthesis)
        self.btn_generate.pack(side="left", padx=10)
        
        self.btn_stop = ttk.Button(frame_btns, text="Arrêter & Sauvegarder", command=self.stop_synthesis, state="disabled")
        self.btn_stop.pack(side="left", padx=10)
        
        self.btn_cancel = ttk.Button(frame_btns, text="Annuler", command=self.cancel_synthesis, state="disabled")
        self.btn_cancel.pack(side="left", padx=10)

        # -- Reference Audio / Voice Cloning --
        frame_voice = ttk.LabelFrame(self, text="Clonage de Voix / Référence Audio (Pour XTTS)", padding=10)
        frame_voice.pack(fill="x", padx=10, pady=5)

        self.lbl_ref_audio = ttk.Label(frame_voice, text="Aucun fichier sélectionné (Optionnel)")
        self.lbl_ref_audio.pack(side="left", fill="x", expand=True)

        btn_browse_ref = ttk.Button(frame_voice, text="Choisir Audio (.wav)", command=self.browse_ref_audio)
        btn_browse_ref.pack(side="right")

    def refresh_models(self):
        # Scan voice-train directory or use configured base
        current_model_dir = self.app.config['paths']['model_dir']
        voice_train_base = os.path.dirname(current_model_dir) 
        
        self.models_data = ModelScanner.scan_for_models(voice_train_base)
        
        if self.models_data:
            model_names = [m['name'] for m in self.models_data]
            self.cbo_models['values'] = model_names
            # Default to first or current
            current_name = os.path.basename(current_model_dir)
            match_index = -1
            for i, m in enumerate(self.models_data):
                if m['type'] == 'custom' and m['name'].endswith(current_name):
                     match_index = i
                     break
            
            if match_index >= 0:
                self.cbo_models.current(match_index)
            else:
                self.cbo_models.current(0)
                
    def on_model_select(self, event):
        idx = self.cbo_models.current()
        if idx >= 0:
            selected = self.models_data[idx]
            self.app.tts_manager.set_model(selected)
            # Update status
            self.lbl_status.config(text=f"Modèle sélectionné : {selected['name']}", foreground="blue")

    def update_state(self):
        pass

    def stop_synthesis(self):
        self.app.tts_manager.request_stop()
        self.lbl_status.config(text="Arrêt demandé... Fin du chunk en cours...", foreground="red")
        self.btn_stop.config(state="disabled")
        self.btn_cancel.config(state="disabled")

    def cancel_synthesis(self):
        self.app.tts_manager.request_abort()
        self.lbl_status.config(text="Annulation demandée...", foreground="red")
        # Buttons will be reset in on_synthesis_complete which receives success=False
        self.btn_stop.config(state="disabled")
        self.btn_cancel.config(state="disabled")

    def browse_ref_audio(self):
        filename = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")])
        if filename:
            self.app.state['speaker_wav'] = filename
            self.lbl_ref_audio.config(text=os.path.basename(filename))

    def start_synthesis(self):
        if 'clean_text' not in self.app.state:
            messagebox.showwarning("Erreur", "Aucun texte préparé.")
            return
        
        if 'base_name' not in self.app.state:
            self.app.state['base_name'] = "output"

        self.btn_generate.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_cancel.config(state="normal")
        
        self.lbl_status.config(text="Chargement du modèle et synthèse en cours...", foreground="orange")
        self.progress['value'] = 0
        
        # Threading
        threading.Thread(target=self.run_tts_thread, daemon=True).start()

    def run_tts_thread(self):
        text = self.app.state['clean_text']
        speaker_wav = self.app.state.get('speaker_wav')
        
        output_dir = self.app.config['paths']['audio_output']
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        wav_filename = f"temp_{self.app.state['base_name']}.wav"
        wav_path = os.path.join(output_dir, wav_filename)
        
        def progress_callback(current, total):
            # Schedule UI update on main thread
            self.app.after(0, lambda: self.update_progress(current, total))

        success = self.app.tts_manager.synthesize(text, wav_path, progress_callback, speaker_wav=speaker_wav)
        
        self.app.after(0, self.on_synthesis_complete, success, wav_path)

    def update_progress(self, current, total):
        self.progress['maximum'] = total
        self.progress['value'] = current
        self.lbl_status.config(text=f"Synthèse : {current}/{total} chunks")

    def on_synthesis_complete(self, success, wav_path):
        self.btn_generate.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_cancel.config(state="disabled")
        
        if success:
            self.app.state['wav_path'] = wav_path
            self.lbl_status.config(text="Synthèse terminée (ou arrêtée) avec succès!", foreground="green")
            self.progress['value'] = 100
            messagebox.showinfo("Succès", f"Audio généré : {wav_path}")
            self.app.notebook.select(3)
        else:
            # Check if it was aborted
            if hasattr(self.app.tts_manager, 'abort_requested') and self.app.tts_manager.abort_requested:
                 self.lbl_status.config(text="Synthèse annulée.", foreground="red")
                 self.progress['value'] = 0
            else:
                 self.lbl_status.config(text="Erreur lors de la synthèse ou Annulation.", foreground="red")
                 # Optional: differentiate explicit abort vs error if needed, but 'success=False' handles both.
                 # If we assume success=False means error unless aborted:
                 pass
            # Don't show error box if aborted, or show "Annulé" info? 
            # If aborted, user expects it.
            if hasattr(self.app.tts_manager, 'abort_requested') and self.app.tts_manager.abort_requested:
                pass # Silent return to ready state
            else:
                 messagebox.showerror("Erreur", "La synthèse a échoué ou a été annulée.")

    def btn_generate_state(self, state):
        # Deprecated helper, kept if needed or remove
        pass

class VideoFrame(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Génération des fichiers finaux").pack(pady=20)
        
        frame_actions = ttk.Frame(self)
        frame_actions.pack(pady=10)
        
        btn_mp3 = ttk.Button(frame_actions, text="Convertir en MP3", command=self.convert_mp3)
        btn_mp3.pack(side="left", padx=10)
        
        btn_video = ttk.Button(frame_actions, text="Générer Vidéo", command=self.generate_video)
        btn_video.pack(side="left", padx=10)
        
        self.lbl_result = ttk.Label(self, text="")
        self.lbl_result.pack(pady=20)
        
        btn_open = ttk.Button(self, text="Ouvrir Dossier de Sortie", command=self.open_folder)
        btn_open.pack(pady=10)

    def update_state(self):
        pass
    
    def convert_mp3(self):
        if 'wav_path' not in self.app.state:
            messagebox.showwarning("Erreur", "Aucun fichier audio WAV généré.")
            return
            
        wav_path = self.app.state['wav_path']
        mp3_path = wav_path.replace("temp_", "").replace(".wav", ".mp3")
        
        if VideoManager.convert_to_mp3(wav_path, mp3_path):
            self.lbl_result.config(text=f"MP3 créé: {os.path.basename(mp3_path)}", foreground="green")
            # Cleanup temp if desired
            # os.remove(wav_path)
        else:
            self.lbl_result.config(text="Erreur MP3", foreground="red")

    def generate_video(self):
        if 'wav_path' not in self.app.state:
            messagebox.showwarning("Erreur", "Aucun fichier audio WAV généré.")
            return
            
        wav_path = self.app.state['wav_path']
        base_name = self.app.state.get('base_name', 'output')
        image_dir = self.app.config['paths']['image_input']
        video_dir = self.app.config['paths']['video_output']
        if not os.path.exists(video_dir): os.makedirs(video_dir)
        
        # Find image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential = os.path.join(image_dir, base_name + ext)
            if os.path.exists(potential):
                image_path = potential
                break
        
        if not image_path:
            messagebox.showwarning("Erreur", f"Aucune image trouvée pour {base_name} dans {image_dir}")
            return
            
        video_path = os.path.join(video_dir, base_name + ".mp4")
        
        self.lbl_result.config(text="Génération vidéo en cours...", foreground="orange")
        threading.Thread(target=self.run_video_thread, args=(wav_path, image_path, video_path), daemon=True).start()

    def run_video_thread(self, wav, img, vid):
        success = VideoManager.create_video(wav, img, vid)
        self.app.after(0, self.on_video_complete, success, vid)

    def on_video_complete(self, success, vid_path):
        if success:
            self.lbl_result.config(text=f"Vidéo créée: {os.path.basename(vid_path)}", foreground="green")
            messagebox.showinfo("Terminé", "Vidéo générée avec succès !")
        else:
            self.lbl_result.config(text="Erreur Vidéo", foreground="red")
            messagebox.showerror("Erreur", "Impossible de créer la vidéo.")

    def open_folder(self):
        out_dir = self.app.config['paths']['audio_output']
        try:
            os.system(f'xdg-open "{out_dir}"')
        except:
            pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
