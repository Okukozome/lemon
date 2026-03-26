import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from app import LemonQualityApp

if __name__ == '__main__':
    app = LemonQualityApp()
    app.mainloop()