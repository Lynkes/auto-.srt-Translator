from faster_whisper import WhisperModel
import os
import tkinter as tk
from tkinter import ttk, filedialog
from colorama import Fore, Style
from googletrans import LANGUAGES, Translator

model_size = "large-v2"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

def format_duration(seconds):
    # Calculate hours, minutes, seconds, and milliseconds
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)  # Extract milliseconds

    # Format into hours:minutes:seconds,milliseconds
    formatted_time = f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"
    
    return formatted_time

def select_folder():
    folder_path = filedialog.askdirectory(title="Select Folder")
    
    if folder_path:
        print(Style.BRIGHT + Fore.GREEN,)
        print(f"Selected Folder: {folder_path}")
        return folder_path
    else:
        print(Style.BRIGHT + Fore.RED,)
        print("No folder selected.")

def translate_text(text, target_language):
    try:
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        if translation:
            return translation.text
        else:
            print(Style.BRIGHT + Fore.RED + "Translation failed. Empty response received.")
            return ""
    except Exception as e:
        print(Style.BRIGHT + Fore.RED + f"Translation error: {e}")
        return ""

def ui():
    root = tk.Tk()
    root.title("Video Subtitle Translator")
    
    # Create language selection dropdown
    language_options = sorted([(code, name) for code, name in LANGUAGES.items()], key=lambda x: x[1])
    language_names = [name for _, name in language_options]
    language_codes = [code for code, _ in language_options]
    language_mapping = dict(zip(language_names, language_codes))
    
    language_dropdown = ttk.Combobox(root, values=language_names)
    language_dropdown.pack(padx=10, pady=10)
    language_dropdown.set("English")  # Default language selection
    
    translate_button = tk.Button(root, text="Translate", command=lambda: process_files_and_translate(select_folder(), language_mapping[language_dropdown.get()]))
    translate_button.pack(pady=10)
    
    root.mainloop()


# Uncomment the following lines if you want to use the process_files_and_translate function directly
# folder_path = select_folder()
# target_language = 'en'  # Default target language
# process_files_and_translate(folder_path, target_language)

def process_files_and_translate(folder_path, target_language):
    # List all files in the specified folder
    files = os.listdir(folder_path)
    
    # Filter for .mp4 files
    mp4_files = [file for file in files if file.lower().endswith('.mp4')]
    
    # Process each .mp4 file
    for mp4_file in mp4_files:
        # Construct the full path to the .mp4 file
        mp4_file_path = os.path.join(folder_path, mp4_file)
        
        # Process the .mp4 file (replace this with your own processing logic)
        print(Style.BRIGHT + Fore.MAGENTA,)
        print(f"Processing file: {mp4_file_path}")
        
        # Example: You can add your segmentation or other processing logic here
        # For instance, if you wanted to process segments:
        segments, info = model.transcribe(mp4_file_path, beam_size=5)
        print(Style.BRIGHT + Fore.GREEN,)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        
        # Get the base filename from the input path
        filename, _ = os.path.splitext(os.path.basename(mp4_file_path))
        # Create the output filename with .srt extension
        output_file_path = os.path.join(folder_path, f"{filename}.srt")

        # Write segments to the output file
        with open(output_file_path, 'w') as output_file:
            for segment in segments:
                output_file.write(f"{segment.id}\n")
                output_file.write(f"{format_duration(segment.start)} --> {format_duration(segment.end)}\n")
                
                if target_language:
                    translated_text = translate_text(segment.text, target_language)
                    output_file.write(f"{translated_text}\n")
                else:
                    output_file.write(f"{segment.text}\n")
                
                output_file.write("\n")
                print(Style.BRIGHT + Fore.WHITE,)
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
    print(Style.BRIGHT + Fore.GREEN,)
    print("ALL .srt files Generated:")
    print(Style.BRIGHT + Fore.RESET,)

# Launch the UI
ui()
