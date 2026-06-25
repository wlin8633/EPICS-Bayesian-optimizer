import sys
import os

# Add lib folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib.OptimizerGUI import OptimizerGUI, tk

if __name__ == '__main__':
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()
