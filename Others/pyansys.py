#!/usr/bin/python
# coding: utf-8

'''
Interact with ANSYS MADPL

Author: Hao Bai (hao.bai@insa-rouen.fr)， Lujie Shi (lujie.shi@insa-rouen.fr)
Licence: 
'''

from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import platform
import re

#!------------------------------------------------------------------------------
#!                                     CLASSES
#!------------------------------------------------------------------------------
class Ansys(object):
    ''' Ansys Mechanical Ansys Parametric Design Language (MAPDL) for python
    '''
    ALL_PATH = {
        # "admin.mydomain.org": "/ansys_inc/v211/ansys/bin/mapdl" ,
        "tower": "/opt/ANSYS/v211/ansys/bin/mapdl" ,
        "lmn-cs.insa-rouen.fr": "/opt/ANSYS-17.2/v172/ansys/bin/mapdl",
        "DESKTOP-I8DJ6DT": r"E:\Program Files\ANSYS Inc\v190\ansys\bin\winx64\MAPDL.exe",
        }

    def __init__(self, inp_file=""):
        if inp_file != "":
            self.read_inp(inp_file)
        self.computer_name = platform.node()
        self.mapdl_path = self.ALL_PATH.get(self.computer_name)
        self.current_path = Path(".")
        self.material = {}
        self.champ = pd.DataFrame()
        self.summary_file = Path(self.project_name+".sum")
        self.error_file = Path(self.project_name+".err")
        self.output_ansys_csv = Path(self.project_name+"_ansys.csv")
        self.output_champ_csv = Path(self.project_name+"_champ.csv")

    def run(self, input_file="", verbose=False, memory=10240):
        if input_file == "":
            input_file = self.inp_file
        command = [self.mapdl_path, "-m", str(memory), "-b", "-i", input_file,
                   "-o", self.summary_file]
        try:
            if verbose == True and self.computer_name == "lmn-cs.insa-rouen.fr":
                subprocess.check_call(command[:-2], shell=False)
            else:
                print("Runing Ansys MAPDL on", input_file, "...")
                subprocess.check_output(command, shell=False)
        except Exception as error:
            if error.returncode != 8:
                self.error_file.write_text(str(error))
                raise Exception("|- [ERROR] Ansys incurs an error for {}"
                                "please see report in {}".format(input_file,
                                self.error_file))
        else:
            pass

    def rm(self, pattern):
        to_remove = list(self.current_path.glob(pattern))
        [f.unlink() for f in to_remove]

    def autoclean(self):
        self.rm("file.*")

    def read_inp(self, filename):
        self.inp_file = Path(filename)
        self.inp_text = self.inp_file.read_text()
        self.project_name = self.inp_file.stem
    
    def save_inp(self, filename=""):
        if filename == "":
            self.inp_file.write_text(self.inp_text)
        else:
            Path(filename).write_text(self.inp_text)
    
    def replace(self, old_str, new_str):
        self.inp_text = self.inp_text.replace(old_str, new_str)

    def get(self, keyword):
        if keyword[-1] != ",":
            keyword += ","
        pos = self.inp_text.find(keyword)
        para = (self.inp_text[pos:].split("\n")[0]).split(",")
        # ignore inline comment
        if "!" in para[-1]:
            value = para[-2]
        else:
            value = para[-1]
        # save to attribute
        if "EX" in keyword:
            self.material["EX"] = float(value)
        elif "DENS" in keyword:
            self.material["DENS"] = float(value)
        elif "PRXY" in keyword:
            self.material["PRXY"] = float(value)
        elif "ALPHAD" in keyword:
            self.material["ALPHAD"] = float(value)
        elif "BETAD" in keyword:
            self.material["BETAD"] = float(value)
        elif "DMPRAT" in keyword:
            self.material["DMPRAT"] = float(value)

    def set(self, keyword, value):
        if keyword[-1] != ",":
            keyword += ","
        pos = self.inp_text.find(keyword)
        sub_str = self.inp_text[pos:]
        sub_str = re.sub(".+\\n", keyword+" "+str(value)+"\n", sub_str, count=1)
        self.inp_text = self.inp_text[:pos] + sub_str

    def autoset(self, keyword):
        if keyword == "/OUTPUT":
            self.set("/OUTPUT",
                "'{}', 'csv', '.'".format(self.output_ansys_csv.stem))

    def read_csv(self, filename):
        return pd.read_csv(filename, delimiter=",", header=0,
            skip_blank_lines=True, skipinitialspace=True)

    def read_ansys_csv(self, filename=""):
        if filename == "":
            filename = self.output_ansys_csv
        self.ansys = self.read_csv(filename)

    def read_champ_csv(self, filename=""):
        if filename == "":
            filename = self.output_champ_csv
        self.champ = self.read_csv(filename)

    def get_champ(self, keyword):
        if keyword.upper() == "FREQUENCY" or keyword.upper() == "FREQ":
            self.champ["FREQUENCY"] = self.ansys["FREQ"]
        
        elif keyword.upper() == "ANGULAR_FREQUENCY" or keyword.upper() == "OMEGA":
            self.champ["ANGULAR_FREQUENCY"] = 2 * np.pi * self.ansys["FREQ"]

        elif keyword.upper() == "DISPLACEMENT" or keyword.upper() == "D":
            self.champ["DISPLACEMENT"] = np.linalg.norm([self.ansys["D_REAL"],
                self.ansys["D_IMAGINARY"]], axis=0)

        elif keyword.upper() == "VELOCITY" or keyword.upper() == "V":
            self.champ["VELOCITY"] = \
                self.champ["DISPLACEMENT"] * self.champ["ANGULAR_FREQUENCY"]

        elif keyword.upper() == "ACCELERATION" or keyword.upper() == "A":
            self.champ["ACCELERATION"] = \
                -self.champ["DISPLACEMENT"] * self.champ["ANGULAR_FREQUENCY"]**2

        elif keyword.upper() == "STRESS" or keyword.upper() == "SIGMA":
            self.champ["STRESS"] = np.linalg.norm([self.ansys["S_REAL"],
                self.ansys["S_IMAGINARY"]], axis=0)

        elif keyword.upper() == "STRAIN" or keyword.upper() == "EPSILON":
            self.champ["STRAIN"] = np.linalg.norm([self.ansys["E_REAL"],
                self.ansys["E_IMAGINARY"]], axis=0)

    def get_champs(self, list_keyword):
        [self.get_champ(kw) for kw in list_keyword]

    def save_champ(self, filename=""):
        if filename == "":
            filename = self.output_champ_csv
        self.champ.to_csv(filename)


#!------------------------------------------------------------------------------
#!                                    FUNCTIONS
#!------------------------------------------------------------------------------
def main(debug = True):
    analyse = Ansys(inp_file="Harmonic_bai.inp")

    ## Read MAPDL input file
    # analyse.read_inp("Harmonic_bai.inp")
    
    ## Retrieve values from MAPDL input file
    # analyse.get("MPDATA, EX")
    # analyse.get("DMPRAT")
    # if debug: print(analyse.material)

    ## Change MAPDL input file
    # #* Method 1: replace exactly
    # analyse.replace("DMPRAT, 0.85e-2,       !the damping ratio", "DMPRAT, 1.0e-2,")

    # #* Method 2: through keyword
    # analyse.set("DMPRAT", 0.1e-2)
    # analyse.set("ALPHAD", "66.66")
    # analyse.set("BETAD,", "66.66e-6")
    # analyse.set("MPDATA, EX", "1, , 1.7e11")
        
    analyse.autoset("/OUTPUT") # keep the same root for output filename

    analyse.save_inp()

    ## Execute MAPDL
    analyse.run("Harmonic_bai.inp", False)
    # analyse.run()
    #* Remove temporary files
    # analyse.rm("file.db")
    # analyse.rm("*.full")

    analyse.autoclean()

    ## Read MAPDL output file
    analyse.read_ansys_csv()

    ## Post-processing
    analyse.get_champ("DISPLACEMENT")
    analyse.get_champs(["ANGULAR_FREQUENCY", "STRESS", "STRAIN"])
    if debug: print(analyse.champ)

    analyse.save_champ()
    

#!------------------------------------------------------------------------------
#!                                     TESTING
#!------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

    