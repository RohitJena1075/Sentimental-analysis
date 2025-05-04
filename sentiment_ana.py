import tkinter as tk
from gui import SentimentAnalysisApp

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalysisApp(root)
    root.mainloop()