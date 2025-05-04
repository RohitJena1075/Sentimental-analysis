import os
import tkinter as tk
from gui import SentimentAnalysisApp

if not os.environ.get("DISPLAY"):
    print("Error: No display environment found. Please run this script on a machine with a GUI.")
    exit(1)

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()