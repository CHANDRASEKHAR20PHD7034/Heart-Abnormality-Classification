import tkinter
from tkinter import *

LARGE_FONT = ("Algerian", 16)

# create the gui window
window = Tk()
window.title("Main window")
window.geometry('900x400')
label = tkinter.Label(window, text="HEART DISEASE PREDICTION WITH SEVERITY CLASSIFICATION USING \n ECG AND PCG SIGNALS BY NOVEL PJM-DJRNN ALGORITHM", fg="maroon4", bg="azure3", font=LARGE_FONT)
label.pack(pady=70, padx=10, side='top')
button = tkinter.Button(window, text="Start", height=2, width=15, command=window.destroy)
button.pack(pady=20, padx=15)
window.configure(bg='azure3')
window.resizable(0, 0)
window.mainloop()