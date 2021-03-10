############################################
# taken from https://stackoverflow.com/questions/20790926/ipython-notebook-open-select-file-with-gui-qt-dialog
############################################

from sys import executable, argv
from subprocess import check_output
from PyQt5.QtWidgets import QFileDialog, QApplication


def gui_fname(directory='C:\\Users\\Luca\\Desktop\\V anno\\I Semestre\\Recommender Systems [Paolo Cremonesi]\\MyReccomender\\optimization_data'):
    """Open a file dialog, starting in the given directory, and return
    the chosen filename"""
    # run this exact file in a separate process, and grab the result
    file = check_output([executable, __file__, directory])
    return file.strip()


if __name__ == "__main__":
    directory = argv[1]
    #print("Starting in directory: {}".format(directory))
    app = QApplication([directory])
    fname = QFileDialog.getOpenFileName(None, "Select a file...",
            directory, filter="All files (*)")
    print(fname[0])