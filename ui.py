from tkinter import *
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2

class DigitsClassifier(Frame):
    """Handwritten digits classifier class"""
    def __init__(self, parent):
            Frame.__init__(self, parent)
            self.parent = parent
            self.color = "black"
            self.brush_size = 12
            self.setUI()
    def set_color(self, new_color):
            """Additional brush color change"""
            self.color = new_color
    def set_brush_size(self, new_size):
            """Changes brush size for testing different lines width"""
            self.brush_size = new_size
    def draw(self, event):
            """Method to draw"""
            self.canv.create_oval(event.x - self.brush_size,
                                event.y - self.brush_size,
                                event.x + self.brush_size,
                                event.y + self.brush_size,
                                fill=self.color, outline=self.color)

    def save(self):
            """Save the current canvas state as the postscript
            uses classify method and shows the result"""
            self.canv.update()
            ps = self.canv.postscript(colormode='mono')
            img = Image.open(io.BytesIO(ps.encode('utf-8')))
            img.save('result.png')
            a = DigitsClassifier.classify()
            print(a)
            self.show_digit(a)

    @staticmethod
    def classify():
            """
            Process the input digit image and returns the result
            :return: digit
            """
            classifier = load_model("./res-valid/model.pth")
            im = cv2.imread("result.png", 0)
            im2 = cv2.resize(im, (28, 28))
            im = im2.reshape(28, 28, -1)
            im = im.reshape(1, 28, 28, 1)
            im = cv2.bitwise_not(im)
            plt.imshow(im.reshape(28, 28), cmap='Greys')
            result = classifier.predict(im)
            a = np.argmax(result)
            return a

    def show_digit(self, digit):
            """
            Show the digit on the canvas
            :param digit: int
            :return: None
            """
            text_label = Label(self, text=digit)
            text_label.grid(row=0, column=5, padx=5, pady=5)

    def setUI(self):
            """Setup for all UI elements"""
            self.parent.title("Drawn Digit Classifier")
            self.pack(fill=BOTH, expand=1)
            self.columnconfigure(6,weight=1)
            self.rowconfigure(2, weight=1)
            self.canv = Canvas(self, bg="white")
            self.canv.grid(row=2, column=0, columnspan=7,
                        padx=5, pady=5,
                        sticky=E + W + S + N)
            self.canv.bind("<B1-Motion>",
                        self.draw)
            color_lab = Label(self, text="Color: ")
            color_lab.grid(row=0, column=0,
                        padx=6)
            black_btn = Button(self, text="Black", width=10,
                            command=lambda: self.set_color("black"))
            black_btn.grid(row=0, column=2)
            white_btn = Button(self, text="White", width=10,
                            command=lambda: self.set_color("white"))
            white_btn.grid(row=0, column=3)
            clear_btn = Button(self, text="Clear all", width=10,
                            command=lambda: self.canv.delete("all"))
            clear_btn.grid(row=0, column=4, sticky=W)
            size_lab = Label(self, text="Brush size: ")
            size_lab.grid(row=1, column=0, padx=5)
            five_btn = Button(self, text="Seven", width=10,
                            command=lambda: self.set_brush_size(7))
            five_btn.grid(row=1, column=2)
            seven_btn = Button(self, text="Ten", width=10,
                            command=lambda: self.set_brush_size(10))
            seven_btn.grid(row=1, column=3)
            ten_btn = Button(self, text="Twenty", width=10,
                            command=lambda: self.set_brush_size(20))
            ten_btn.grid(row=1, column=4)
            done_btn = Button(self, text="Done", width=10,
                            command=lambda: self.save())
            done_btn.grid(row=1, column=5)

def main():
    root = Tk()
    root.geometry("450x450")
    root.resizable(0, 0)
    app = DigitsClassifier(root)
    root.mainloop()
if __name__ == '__main__':
    main()