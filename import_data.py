from tkinter import Tk, filedialog
import pandas as pd
from scipy.io import arff

def import_file():
    '''
    This function is used to convert any file into pandas dataframe
    '''

    root = Tk()
    root.withdraw()
    File_R = filedialog.askopenfilenames(filetypes=[("Data Files", ".arff .csv .xlsx")])

    def tupleTostr(tuple):
            str = ''
            for item in tuple:
                str= str + item
                return  str

    File_RE = tupleTostr(File_R)

    if (any('.arff' in i for i in File_R)) :
        data_arff = arff.loadarff(File_RE)
        df = pd.DataFrame(data_arff[0])
        return (df)
    elif(any('.csv'in i for i in File_R)):
        data_csv=pd.read_csv(File_RE)
        return (data_csv)
    elif(any('.xlsx' in i for i in File_R)):
        data_excel = pd.read_excel(File_RE)
        return (data_excel)