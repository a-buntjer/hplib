# Import packages
import os
import pandas as pd
import scipy
import hplib as hpl
from functools import partial
import concurrent.futures
# Functions

def import_heating_data():
    # read in keymark data from *.txt files in /input/txt/
    # save a dataframe to database_heating.csv in folder /output/
    Modul = []
    Manufacturer = []
    Date = []
    Refrigerant = []
    Mass = []
    Poff = []
    Psb = []
    Prated = []
    SPLindoor = []
    SPLoutdoor = []
    Type = []
    Climate = []
    Guideline = []
    T_in = []
    T_out = []
    P_th = []
    COP = []
    df = pd.DataFrame()
    os.chdir('../')
    root = os.getcwd()
    Scanordner = (root + '/input/txt')
    os.chdir(Scanordner)
    Scan = os.scandir(os.getcwd())
    with Scan as dir1:
        for file in dir1:
            with open(file, 'r', encoding='utf-8') as f:
                contents = f.readlines()
                date = 'NaN'
                modul = 'NaN'
                prated_low = 'NaN'
                prated_medium = 'NaN'
                heatpumpType = 'NaN'
                refrigerant = 'NaN'
                splindoor_low = 'NaN'
                splindoor_medium = 'NaN'
                sploutdoor_low = 'NaN'
                sploutdoor_medium = 'NaN'
                poff = 'NaN'
                climate = 'NaN'
                NumberOfTestsPerNorm = []
                NumberOfTestsPerModule = []
                i = 1  # indicator for the line wich is read
                d = 0  # indicator if only medium Temperature is given
                p = 0  # -15° yes or no
                date = contents[1]
                date = date[61:]
                if (date == '17 Dec 2020\n'):
                    date = '17.12.2020\n'
                if (date == '18 Dec 2020\n'):
                    date = '18.12.2020\n'
                if (date.startswith('5 Mar 2021')):
                    date = '05.03.2021\n'
                if (date.startswith('15 Feb 2021')):
                    date = '15.02.2021\n'
                if (date.startswith('22 Feb 2021')):
                    date = '22.02.2021\n'
                for lines in contents:
                    i = i + 1
                    if (lines.startswith('Name\n') == 1):
                        manufacturer = (contents[i])
                        if (manufacturer.find('(') > 0):
                            manufacturer = manufacturer.split('(', 1)[1].split('\n')[0]
                        if manufacturer.endswith('GmbH\n'):
                            manufacturer = manufacturer[:-5]
                        if manufacturer.endswith('S.p.A.\n'):
                            manufacturer = manufacturer[:-6]
                        if manufacturer.endswith('s.p.a.\n'):
                            manufacturer = manufacturer[:-6]
                        if manufacturer.endswith('S.p.A\n'):
                            manufacturer = manufacturer[:-5]
                        if manufacturer.endswith('S.L.U.\n'):
                            manufacturer = manufacturer[:-6]
                        if manufacturer.endswith('s.r.o.\n'):
                            manufacturer = manufacturer[:-6]
                        if manufacturer.endswith('S.A.\n'):
                            manufacturer = manufacturer[:-4]
                        if manufacturer.endswith('S.L.\n'):
                            manufacturer = manufacturer[:-4]
                        if manufacturer.endswith('B.V.\n'):
                            manufacturer = manufacturer[:-4]
                        if manufacturer.endswith('N.V.\n'):
                            manufacturer = manufacturer[:-4]
                        if manufacturer.endswith('GmbH & Co KG\n'):
                            manufacturer = manufacturer[:-12]
                        elif manufacturer.startswith('NIBE'):
                            manufacturer = 'Nibe\n'
                        elif manufacturer.startswith('Nibe'):
                            manufacturer = 'Nibe\n'
                        elif manufacturer.startswith('Mitsubishi'):
                            manufacturer = 'Mitsubishi\n'
                        elif manufacturer.startswith('Ochsner'):
                            manufacturer = 'Ochsner\n'
                        elif manufacturer.startswith('OCHSNER'):
                            manufacturer = 'Ochsner\n'
                        elif manufacturer.startswith('Viessmann'):
                            manufacturer = 'Viessmann\n'

                    elif (lines.endswith('Date\n') == 1):
                        date = (contents[i])
                        if (date == 'basis\n'):
                            date = contents[i - 3]
                            date = date[14:]
                    elif (lines.startswith('Model') == 1):
                        modul = (contents[i - 2])
                        splindoor_low = 'NaN'
                        splindoor_medium = 'NaN'
                        sploutdoor_low = 'NaN'
                        sploutdoor_medium = 'NaN'
                    elif lines.endswith('Type\n'):
                        heatpumpType = contents[i][:-1]
                        if heatpumpType.startswith('A'):
                            heatpumpType = 'Outdoor Air/Water'
                        if heatpumpType.startswith('Eau glycol'):
                            heatpumpType = 'Brine/Water'
                    elif (lines.startswith('Sound power level indoor')):

                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                splindoor_low = contents[i + 4][:-7]
                                splindoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            splindoor_medium = contents[i + 4][:-7]
                            splindoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                splindoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                splindoor_low = contents[i][:-7]

                            if (contents[i - 6].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            else:
                                splindoor_low = contents[i][:-7]
                                splindoor_medium = contents[i][:-7]

                    elif (lines.startswith('Sound power level outdoor')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                sploutdoor_low = contents[i + 4][:-7]
                                sploutdoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            sploutdoor_medium = contents[i + 4][:-7]
                            sploutdoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                sploutdoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            else:
                                sploutdoor_low = contents[i][:-7]
                                sploutdoor_medium = contents[i][:-7]

                    elif (lines.startswith('Puissance acoustique extérieure')):
                        b = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                sploutdoor_low = contents[i + 4][:-7]
                                sploutdoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            sploutdoor_medium = contents[i + 4][:-7]
                            sploutdoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                sploutdoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            else:
                                sploutdoor_low = contents[i][:-7]
                                sploutdoor_medium = contents[i][:-7]
                    elif (lines.startswith('Potencia sonora de la unidad interior')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                splindoor_low = contents[i + 4][:-7]
                                splindoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            splindoor_medium = contents[i + 4][:-7]
                            splindoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                splindoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            else:
                                splindoor_low = contents[i][:-7]
                                splindoor_medium = contents[i][:-7]
                    elif (lines.startswith('Potencia sonora de la unidad exterior')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                sploutdoor_low = contents[i + 4][:-7]
                                sploutdoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            sploutdoor_medium = contents[i + 4][:-7]
                            sploutdoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                sploutdoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            else:
                                sploutdoor_low = contents[i][:-7]
                                sploutdoor_medium = contents[i][:-7]
                    elif (lines.startswith('Nivel de Potência sonora interior')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                splindoor_low = contents[i + 4][:-7]
                                splindoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            splindoor_medium = contents[i + 4][:-7]
                            splindoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                splindoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            else:
                                splindoor_low = contents[i][:-7]
                                splindoor_medium = contents[i][:-7]
                    elif (lines.startswith('Nivel de Potência sonora exterior')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                sploutdoor_low = contents[i + 4][:-7]
                                sploutdoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            sploutdoor_medium = contents[i + 4][:-7]
                            sploutdoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                sploutdoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            else:
                                sploutdoor_low = contents[i][:-7]
                                sploutdoor_medium = contents[i][:-7]
                    elif (lines.startswith('Livello di potenza acustica interna')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                splindoor_low = contents[i + 4][:-7]
                                splindoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            splindoor_medium = contents[i + 4][:-7]
                            splindoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                splindoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                splindoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                splindoor_medium = contents[i + 2][:-7]
                            else:
                                splindoor_low = contents[i][:-7]
                                splindoor_medium = contents[i][:-7]
                    elif (lines.startswith('Livello di potenza acustica externa')):
                        SPL = 1
                        if (contents[i].startswith('Low')):
                            if contents[i + 2].startswith('Medium'):
                                sploutdoor_low = contents[i + 4][:-7]
                                sploutdoor_medium = contents[i + 6][:-7]
                        if contents[i].startswith('Medium'):
                            sploutdoor_medium = contents[i + 4][:-7]
                            sploutdoor_low = contents[i + 6][:-7]
                        elif (contents[i].endswith('dB(A)\n')):
                            if (contents[i - 3].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 3].startswith('Medium')):
                                sploutdoor_medium = contents[i][:-7]
                            if (contents[i - 6].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 6].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            if (contents[i - 4].startswith('Low')):
                                sploutdoor_low = contents[i][:-7]
                            if (contents[i - 4].startswith('Medium')):
                                sploutdoor_medium = contents[i + 2][:-7]
                            else:
                                sploutdoor_low = contents[i][:-7]
                                sploutdoor_medium = contents[i][:-7]
                    elif (lines == 'Refrigerant\n'):
                        if (contents[i - 3] == 'Mass Of\n'):
                            continue
                        refrigerant = (contents[i])
                    elif (lines.startswith('Mass Of') == 1):
                        if (lines == 'Mass Of\n'):
                            mass = contents[i + 1]
                        elif (lines.endswith('kg\n') == 1):
                            mass = contents[i - 2]
                            mass = mass[20:]
                        else:
                            mass = contents[i]

                    elif lines.startswith('Average'):
                        climate = 'average'
                    elif lines.startswith('Cold'):
                        climate = 'cold'
                    elif lines.startswith('Warmer Climate'):
                        climate = 'warm'

                    elif (lines.startswith('EN') == 1):
                        if (p == 1):
                            Poff.append(poff)
                            Psb.append(psb)
                        if (p == 2):
                            Poff.append(poff)
                            Poff.append(poff)
                            Psb.append(psb)
                            Psb.append(psb_medium)
                        guideline = (contents[i - 2])
                        d = 0  # Medium or Low Content
                        p = 0  # -15 yes or no

                        NumberOfTestsPerNorm = []
                        if (contents[i - 1].startswith('Low') == 1):
                            d = 0
                            continue
                        if (contents[i - 1] == '\n'):
                            continue
                        if (contents[i - 1].startswith('Medium')):
                            d = 1
                        else:
                            d = 0
                    if lines.startswith('Prated'):
                        prated_low = contents[i][:-4]
                        if (contents[i + 2].endswith('kW\n')):
                            prated_medium = contents[i + 2][:-4]


                    elif (lines.startswith('Pdh Tj = -15°C') == 1):  # check
                        if (contents[i].endswith('Cdh\n') == 1):  # wrong content
                            continue
                        if (contents[i] == '\n'):  # no content
                            continue
                        else:
                            minusfifteen_low = contents[i]
                            P_th.append(minusfifteen_low[:-4])
                            T_in.append('-15')
                            if d == 0:  # first low than medium Temperatur
                                if (climate == 'average'):
                                    T_out.append('35')
                                elif (climate == 'cold'):
                                    T_out.append('32')
                                elif (climate == 'warm'):
                                    T_out.append('35')

                            if d == 1:  # first medium Temperature
                                if (climate == 'average'):
                                    T_out.append('55')
                                elif (climate == 'cold'):
                                    T_out.append('49')
                                elif (climate == 'warm'):
                                    T_out.append('55')

                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            Mass.append(mass[:-4])
                            Prated.append(prated_low)
                            SPLindoor.append(splindoor_low)
                            # SPLindoor.append(splindoor_medium)
                            SPLoutdoor.append(sploutdoor_low)
                            # SPLoutdoor.append(sploutdoor_medium)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                            Type.append(heatpumpType)
                            if (contents[i + 2].startswith('COP')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('Disclaimer')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('EHPA')):  # End of page
                                if (contents[i + 8].startswith('COP')):  # end of page plus no medium heat
                                    continue
                            minusfifteen_medium = contents[i + 2]

                            P_th.append(minusfifteen_medium[:-4])
                            T_in.append('-15')
                            if (climate == 'average'):
                                T_out.append('55')
                            elif (climate == 'cold'):
                                T_out.append('49')
                            elif (climate == 'warm'):
                                T_out.append('55')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            Mass.append(mass[:-4])
                            Prated.append(prated_medium)
                            # SPLindoor.append(splindoor_low)
                            SPLindoor.append(splindoor_medium)
                            # SPLoutdoor.append(sploutdoor_low)
                            SPLoutdoor.append(sploutdoor_medium)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                    elif (lines.startswith('COP Tj = -15°C')):
                        if (contents[i] == '\n'):
                            continue
                        if (contents[i].startswith('EHPA')):
                            continue
                        COP.append(contents[i][:-1])
                        NumberOfTestsPerModule.append(i)
                        p = 1

                        if (contents[i + 2].startswith('Pdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('Cdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('EHPA')):  # no medium Climate
                            continue
                        COP.append(contents[i + 2][:-1])
                        NumberOfTestsPerModule.append(i)
                        p = 2


                    elif (lines.startswith('Pdh Tj = -7°C') == 1):  # check
                        minusseven_low = contents[i]
                        P_th.append(minusseven_low[:-4])
                        T_in.append('-7')
                        if d == 0:  # first low than medium Temperatur
                            if (climate == 'average'):
                                T_out.append('34')
                            elif (climate == 'cold'):
                                T_out.append('30')
                            elif (climate == 'warm'):
                                T_out.append('35')

                        if d == 1:  # first medium Temperature
                            if (climate == 'average'):
                                T_out.append('52')
                            elif (climate == 'cold'):
                                T_out.append('44')
                            elif (climate == 'warm'):
                                T_out.append('55')

                        Modul.append(modul[7:-1])
                        Manufacturer.append(manufacturer[:-1])
                        Date.append(date[:-1])
                        Refrigerant.append(refrigerant[:-1])
                        Mass.append(mass[:-4])
                        Prated.append(prated_low)
                        SPLindoor.append(splindoor_low)
                        # SPLindoor.append(splindoor_medium)
                        SPLoutdoor.append(sploutdoor_low)
                        # SPLoutdoor.append(sploutdoor_medium)
                        Type.append(heatpumpType)
                        Guideline.append(guideline[:-1])
                        Climate.append(climate)

                        if (contents[i + 2].startswith('COP') == 1):
                            continue
                        else:
                            minusseven_medium = contents[i + 2]
                            P_th.append(minusseven_medium[:-4])
                            T_in.append('-7')
                            if (climate == 'average'):
                                T_out.append('52')
                            elif (climate == 'cold'):
                                T_out.append('44')
                            elif (climate == 'warm'):
                                T_out.append('55')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            # SPLindoor.append(splindoor_low)
                            SPLindoor.append(splindoor_medium)
                            # SPLoutdoor.append(sploutdoor_low)
                            SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_medium)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                    elif (lines.startswith('COP Tj = -7°C')):
                        COP.append(contents[i][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)
                        if (contents[i + 2].startswith('Pdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('Cdh')):  # no medium Climate
                            continue
                        COP.append(contents[i + 2][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)


                    elif (lines.startswith('Pdh Tj = +2°C') == 1):
                        if (contents[i].endswith('Cdh\n') == 1):  # wrong content
                            continue
                        if (contents[i] == '\n'):  # no content
                            continue
                        else:
                            plustwo_low = contents[i]
                            P_th.append(plustwo_low[:-4])
                            T_in.append('2')
                            if d == 0:  # first low than medium Temperatur
                                if (climate == 'average'):
                                    T_out.append('30')
                                elif (climate == 'cold'):
                                    T_out.append('27')
                                elif (climate == 'warm'):
                                    T_out.append('35')

                            if d == 1:  # first medium Temperature
                                if (climate == 'average'):
                                    T_out.append('42')
                                elif (climate == 'cold'):
                                    T_out.append('37')
                                elif (climate == 'warm'):
                                    T_out.append('55')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            SPLindoor.append(splindoor_low)
                            # SPLindoor.append(splindoor_medium)
                            SPLoutdoor.append(sploutdoor_low)
                            # SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_low)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                            if (contents[i + 2].startswith('COP')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('Disclaimer')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('EHPA')):  # End of page
                                if (contents[i + 8].startswith('COP')):  # end of page plus no medium heat
                                    continue
                            plustwo_medium = contents[i + 2]
                            # if(plustwo_low[:-1].endswith('kW')==0):#test
                            # print(plustwo_low[:-1])
                            # if(plustwo_medium[:-1].endswith('kW')==0):#test
                            # print(file.name)#plustwo_medium[:-1]

                            P_th.append(plustwo_medium[:-4])
                            T_in.append('2')
                            if (climate == 'average'):
                                T_out.append('42')
                            elif (climate == 'cold'):
                                T_out.append('37')
                            elif (climate == 'warm'):
                                T_out.append('55')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            # SPLindoor.append(splindoor_low)
                            SPLindoor.append(splindoor_medium)
                            # SPLoutdoor.append(sploutdoor_low)
                            SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_medium)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                    elif (lines.startswith('COP Tj = +2°C')):  # check
                        if (contents[i] == '\n'):  # no infos
                            continue
                        if (contents[i].startswith('EHPA')):  # end of page
                            print(file.name)
                            continue
                        if (contents[i + 2].startswith('Warmer')):  # usless infos
                            continue
                        if (contents[i] == 'n/a\n'):  # usless infos
                            continue
                        COP.append(contents[i][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)

                        if (contents[i + 2].startswith('Pdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('Cdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('EHPA')):  # no medium Climate
                            continue
                        COP.append(contents[i + 2][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)


                    elif (lines.startswith('Pdh Tj = +7°C') == 1):
                        if (contents[i].endswith('Cdh\n') == 1):  # wrong content
                            continue
                        if (contents[i] == '\n'):  # no content
                            continue
                        else:
                            plusseven_low = contents[i]
                            P_th.append(plusseven_low[:-4])
                            T_in.append('7')
                            if d == 0:  # first low than medium Temperatur
                                if (climate == 'average'):
                                    T_out.append('27')
                                elif (climate == 'cold'):
                                    T_out.append('25')
                                elif (climate == 'warm'):
                                    T_out.append('31')

                            if d == 1:  # first medium Temperature
                                if (climate == 'average'):
                                    T_out.append('36')
                                elif (climate == 'cold'):
                                    T_out.append('32')
                                elif (climate == 'warm'):
                                    T_out.append('46')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            SPLindoor.append(splindoor_low)
                            # SPLindoor.append(splindoor_medium)
                            SPLoutdoor.append(sploutdoor_low)
                            # SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_low)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                            if (contents[i + 2].startswith('COP')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('Disclaimer')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('EHPA')):  # End of page
                                if (contents[i + 8].startswith('COP')):  # end of page plus no medium heat
                                    continue
                            plusseven_medium = contents[i + 2]

                            P_th.append(plusseven_medium[:-4])
                            T_in.append('7')
                            if (climate == 'average'):
                                T_out.append('36')
                            elif (climate == 'cold'):
                                T_out.append('32')
                            elif (climate == 'warm'):
                                T_out.append('46')

                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            # SPLindoor.append(splindoor_low)
                            SPLindoor.append(splindoor_medium)
                            # SPLoutdoor.append(sploutdoor_low)
                            SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_medium)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                    elif (lines.startswith('COP Tj = +7°C')):  # check
                        if (contents[i] == '\n'):  # no infos
                            continue
                        if (contents[i].startswith('EHPA')):  # end of page
                            continue
                        if (contents[i + 2].startswith('Warmer')):  # usless infos
                            continue
                        if (contents[i] == 'n/a\n'):  # usless infos
                            continue
                        COP.append(contents[i][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)

                        if (contents[i + 2].startswith('Pdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('Cdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('EHPA')):  # no medium Climate
                            continue
                        COP.append(contents[i + 2][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)


                    elif (lines.startswith('Pdh Tj = 12°C') == 1):

                        if (contents[i].endswith('Cdh\n') == 1):  # wrong content
                            continue
                        if (contents[i] == '\n'):  # no content
                            continue
                        if (contents[i].startswith('EHPA Secretariat') == 1):
                            plustwelfe_low = (contents[i - 11])

                            P_th.append(plustwelfe_low[:-4])
                            T_in.append('12')
                            if (climate == 'average'):
                                T_out.append('24')
                            elif (climate == 'cold'):
                                T_out.append('24')
                            elif (climate == 'warm'):
                                T_out.append('26')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            SPLindoor.append(splindoor_low)
                            # SPLindoor.append(splindoor_medium)
                            SPLoutdoor.append(sploutdoor_low)
                            # SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_low)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                            plustwelfe_medium = (contents[i - 9])

                            P_th.append(plustwelfe_medium[:-4])
                            T_in.append('12')
                            if (climate == 'average'):
                                T_out.append('30')
                            elif (climate == 'cold'):
                                T_out.append('28')
                            elif (climate == 'warm'):
                                T_out.append('34')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            # SPLindoor.append(splindoor_low)
                            SPLindoor.append(splindoor_medium)
                            # SPLoutdoor.append(sploutdoor_low)
                            SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_medium)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                        else:
                            plustwelfe_low = contents[i]

                            P_th.append(plustwelfe_low[:-4])
                            T_in.append('12')
                            if d == 0:  # first low than medium Temperatur
                                if (climate == 'average'):
                                    T_out.append('24')
                                elif (climate == 'cold'):
                                    T_out.append('24')
                                elif (climate == 'warm'):
                                    T_out.append('26')

                            if d == 1:  # first medium Temperature
                                if (climate == 'average'):
                                    T_out.append('30')
                                elif (climate == 'cold'):
                                    T_out.append('28')
                                elif (climate == 'warm'):
                                    T_out.append('34')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            SPLindoor.append(splindoor_low)

                            SPLoutdoor.append(sploutdoor_low)

                            Mass.append(mass[:-4])
                            Prated.append(prated_low)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                            if (contents[i + 2].startswith('COP')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('Disclaimer')):  # for PDF without medium heat
                                continue
                            if (contents[i + 2].startswith('EHPA')):  # End of page
                                if (contents[i + 8].startswith('COP')):  # end of page plus no medium heat
                                    continue

                            plustwelfe_medium = contents[i + 2]
                            P_th.append(plustwelfe_medium[:-4])
                            T_in.append('12')
                            if (climate == 'average'):
                                T_out.append('30')
                            elif (climate == 'cold'):
                                T_out.append('28')
                            elif (climate == 'warm'):
                                T_out.append('34')
                            Modul.append(modul[7:-1])
                            Manufacturer.append(manufacturer[:-1])
                            Date.append(date[:-1])
                            Refrigerant.append(refrigerant[:-1])
                            # SPLindoor.append(splindoor_low)
                            SPLindoor.append(splindoor_medium)

                            SPLoutdoor.append(sploutdoor_medium)
                            Mass.append(mass[:-4])
                            Prated.append(prated_medium)
                            Type.append(heatpumpType)
                            Guideline.append(guideline[:-1])
                            Climate.append(climate)

                    elif (lines.startswith('COP Tj = 12°C')):  # check
                        if (contents[i] == '\n'):  # no infos
                            continue
                        if (contents[i].startswith('EHPA')):  # end of page
                            print('W')
                            continue
                        if (contents[i + 2].startswith('Warmer')):  # usless infos
                            continue
                        if (contents[i] == 'n/a\n'):  # usless infos
                            continue
                        COP.append(contents[i][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)

                        if (contents[i + 2].startswith('Pdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('Cdh')):  # no medium Climate
                            continue
                        if (contents[i + 2].startswith('EHPA')):  # no medium Climate
                            continue
                        COP.append(contents[i + 2][:-1])
                        NumberOfTestsPerNorm.append(i)
                        NumberOfTestsPerModule.append(i)


                    elif (lines.startswith('Poff')):
                        l = 0  # l shows if Poff Medium is different to Poff Low Temperature
                        c = 2  # c is just an iterator to print every second Poff
                        poff = contents[i][:-2]
                        if poff.endswith(' '):
                            poff = poff[:-1]
                            if poff.endswith('.00'):
                                poff = poff[:-3]
                        second_poff = contents[i + 2][:-2]
                        if second_poff.endswith(' '):
                            second_poff = second_poff[:-1]
                            if second_poff.endswith('.00'):
                                second_poff = second_poff[:-3]
                        if (poff != second_poff):  # see if Poff Medium to Poff low
                            if (contents[i + 2].endswith('W\n')):
                                if (contents[i + 2] != 'W\n'):
                                    l = 1
                        for Tests in NumberOfTestsPerNorm:
                            if l == 0:
                                Poff.append(poff)
                            if l == 1:
                                c += 1
                                if c % 2 == 1:
                                    Poff.append(poff)
                                if c % 2 == 0:
                                    Poff.append(second_poff)
                    elif (lines.startswith('PSB')):
                        l = 0  # l shows if Poff Medium is different to Poff Low Temperature
                        c = 2  # c is just an iterator to print every second Poff
                        psb = contents[i][:-2]
                        if psb.endswith(' '):
                            psb = psb[:-1]
                            if psb.endswith('.00'):
                                psb = psb[:-3]
                        psb_medium = contents[i + 2][:-2]
                        if psb_medium.endswith(' '):
                            psb_medium = psb_medium[:-1]
                            if psb_medium.endswith('.00'):
                                psb_medium = psb_medium[:-3]
                        if (psb != psb_medium):  # see if Poff Medium to Poff low
                            if (contents[i + 2].endswith('W\n')):
                                if (contents[i + 2] != 'W\n'):
                                    l = 1

                        for Tests in NumberOfTestsPerNorm:
                            if l == 0:
                                Psb.append(psb)
                            if l == 1:
                                c += 1
                                if c % 2 == 1:
                                    Psb.append(psb)
                                if c % 2 == 0:
                                    Psb.append(psb_medium)

                if p == 1:
                    Poff.append(poff)
                    Psb.append(psb)
                if p == 2:
                    Poff.append(poff)
                    Poff.append(second_poff)
                    Psb.append(psb)
                    Psb.append(psb_medium)

    df['Manufacturer'] = Manufacturer
    df['Model'] = Modul
    df['Date'] = Date
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df['Type'] = Type
    df['SPL indoor [dBA]'] = SPLindoor
    df['SPL outdoor [dBA]'] = SPLoutdoor
    df['Refrigerant'] = Refrigerant
    df['Mass of Refrigerant [kg]'] = Mass
    df['Poff [W]'] = Poff
    df['Poff [W]'] = df['Poff [W]'].astype(int)
    df['PSB [W]'] = Psb
    df['PSB [W]'] = df['PSB [W]'].astype(int)
    df['Prated [W]'] = Prated

    df['Guideline'] = Guideline
    df['Climate'] = Climate
    df['T_in [°C]'] = T_in
    df['T_in [°C]'] = df['T_in [°C]'].astype(int)
    df['T_out [°C]'] = T_out
    df['T_out [°C]'] = df['T_out [°C]'].astype(int)
    """
    T_out for Low Temperature
            T-in:   -15 -7  2   7   12

    Cold Climate    32  30  27  25  24
    Average Climate 35  34  30  27  24
    Warm Climate    35  35  35  31  26


    T_out for Medium Temperature
            T-in:   -15 -7  2   7   12

    Cold Climate    49  44  37  32  28
    Average Climate 55  52  42  36  30
    Warm Climate    55  55  55  46  34
    """
    df['P_th [W]'] = P_th
    df['P_th [W]'] = ((df['P_th [W]'].astype(float)) * 1000).astype(int)
    df['COP'] = COP
    df['COP'] = round(df['COP'].astype(float), 2)
    df['P_el [W]'] = round(df['P_th [W]'] / df['COP'])
    df['P_el [W]'] = df['P_el [W]'].fillna(0).astype(int)
    df['PSB [W]'] = df['PSB [W]'].where(df['PSB [W]'] > df['Poff [W]'],
                                        df['Poff [W]'])  # Poff should not be bigger than PSB
    df.drop(columns=['Poff [W]'], inplace=True)  # not needed anymore
    filt = df['P_th [W]'] < 50  # P_th too small
    df.drop(index=df[filt].index, inplace=True)
    # add T_amb and change T_in to right values
    df['T_amb [°C]'] = df['T_in [°C]']
    filt = df['Type'] == 'Brine/Water'
    df.loc[filt, 'T_in [°C]'] = 0
    filt = df['Type'] == 'Water/Water'
    df.loc[filt, 'T_in [°C]'] = 10
    df = df[
        ['Manufacturer', 'Model', 'Date', 'Type', 'Refrigerant', 'Mass of Refrigerant [kg]', 'PSB [W]', 'Prated [W]',
         'SPL indoor [dBA]', 'SPL outdoor [dBA]', 'Climate', 'T_amb [°C]', 'T_in [°C]', 'T_out [°C]', 'P_th [W]',
         'P_el [W]', 'COP']]
    df.sort_values(by=['Manufacturer', 'Model'], inplace=True)
    os.chdir("../")
    df.to_csv(r'../output/database_heating.csv', index=False)
    os.chdir('../src/')


def import_cooling_data():
    # read in keymark data from *.txt files in /input/txt/
    # save a dataframe to database_heating.csv in folder /output/
    Modul = []
    Manufacturer = []
    Date = []
    Refrigerant = []
    Mass = []
    Type = []
    Pdesignc = []
    Temperatur = []
    T_outside = []
    PDC = []
    EER = []
    df = pd.DataFrame()
    os.chdir('../')
    root = os.getcwd()
    Scanordner = (root + '/input/txt')
    os.chdir(Scanordner)
    Scan = os.scandir(os.getcwd())
    with Scan as dir1:
        for file in dir1:
            with open(file, 'r', encoding='utf-8') as f:
                contents = f.readlines()
                T = 0
                i = 1  # indicator for the line wich is read
                date = contents[1]
                date = date[61:]
                if (date == '17 Dec 2020\n'):
                    date = '17.12.2020\n'
                if (date == '18 Dec 2020\n'):
                    date = '18.12.2020\n'
                if (date.startswith('5 Mar 2021')):
                    date = '05.03.2021\n'
                if (date.startswith('15 Feb 2021')):
                    date = '15.02.2021\n'
                if (date.startswith('22 Feb 2021')):
                    date = '22.02.2021\n'
                for lines in contents:
                    i = i + 1
                    if (lines.startswith('Name\n') == 1):
                        manufacturer = (contents[i][:-1])
                        if (manufacturer.find('(') > 0):
                            manufacturer = manufacturer.split('(', 1)[1].split(')')[0]
                        elif manufacturer.startswith('NIBE'):
                            manufacturer = 'Nibe'
                        elif manufacturer.startswith('Nibe'):
                            manufacturer = 'Nibe'
                        elif manufacturer.startswith('Mitsubishi'):
                            manufacturer = 'Mitsubishi'
                        elif manufacturer.startswith('Ochsner'):
                            manufacturer = 'Ochsner'
                        elif manufacturer.startswith('OCHSNER'):
                            manufacturer = 'Ochsner'
                        elif manufacturer.startswith('Viessmann'):
                            manufacturer = 'Viessmann'
                    elif (lines.endswith('Date\n') == 1):
                        date = (contents[i])
                        if (date == 'basis\n'):
                            date = contents[i - 3]
                            date = date[14:]
                    elif (lines.startswith('Model') == 1):
                        modul = (contents[i - 2][7:-1])
                        temperatur2 = ''
                    elif lines.endswith('Type\n'):
                        heatpumpType = contents[i][:-1]
                        if heatpumpType.startswith('A'):
                            heatpumpType = 'Outdoor Air/Water'
                        if heatpumpType.startswith('Eau glycol'):
                            heatpumpType = 'Brine/Water'
                    elif (lines == 'Refrigerant\n'):
                        if (contents[i - 3] == 'Mass Of\n'):
                            continue
                        refrigerant = (contents[i][:-1])
                    elif (lines.startswith('Mass Of') == 1):
                        if (lines == 'Mass Of\n'):
                            mass = contents[i + 1][:-4]
                        elif (lines.endswith('kg\n') == 1):
                            mass = contents[i - 2]
                            mass = mass[20:-4]
                        else:
                            mass = contents[i][:-4]


                    elif lines.startswith('+'):
                        if T == 0:
                            temperatur1 = contents[i - 2][:-1]
                            if (contents[i].startswith('+')):
                                temperatur2 = contents[i][:-1]
                                T = 1
                                temperatur2 = (temperatur2[1:3])
                            temperatur1 = (temperatur1[1:2])
                        else:
                            T = 0
                    elif lines.startswith('Pdesignc'):
                        pdesignc1 = contents[i][:-4]
                        if temperatur2 != '':
                            pdesignc2 = contents[i + 2][:-4]

                    elif lines.startswith('Pdc Tj = 30°C'):
                        pdcT1_30 = contents[i][:-4]

                        if contents[i + 2].endswith('W\n'):
                            pdcT2_30 = contents[i + 2][:-4]


                    elif lines.startswith('EER Tj = 30°C'):

                        eerT1_30 = (contents[i][:-1])
                        EER.append(eerT1_30)
                        PDC.append(pdcT1_30)
                        T_outside.append('30')
                        Pdesignc.append(pdesignc1)
                        Temperatur.append(temperatur1)
                        Modul.append(modul)
                        Manufacturer.append(manufacturer)
                        Date.append(date)
                        Refrigerant.append(refrigerant)
                        Mass.append(mass)
                        Type.append(heatpumpType)

                        if temperatur2 != '':
                            eerT2_30 = contents[i + 2][:-1]
                            EER.append(eerT2_30)
                            PDC.append(pdcT2_30)
                            T_outside.append('30')
                            Pdesignc.append(pdesignc2)
                            Temperatur.append(temperatur2)
                            Modul.append(modul)
                            Manufacturer.append(manufacturer)
                            Date.append(date)
                            Refrigerant.append(refrigerant)
                            Mass.append(mass)
                            Type.append(heatpumpType)

                    elif lines.startswith('Pdc Tj = 35°C'):
                        pdcT1_35 = contents[i][:-4]
                        if contents[i + 2].endswith('W\n'):
                            pdcT2_35 = contents[i + 2][:-4]

                    elif lines.startswith('EER Tj = 35°C'):
                        eerT1_35 = (contents[i][:-1])
                        EER.append(eerT1_35)
                        PDC.append(pdcT1_35)
                        T_outside.append('35')
                        Pdesignc.append(pdesignc1)
                        Temperatur.append(temperatur1)
                        Modul.append(modul)
                        Manufacturer.append(manufacturer)
                        Date.append(date)
                        Refrigerant.append(refrigerant)
                        Mass.append(mass)
                        Type.append(heatpumpType)
                        if temperatur2 != '':
                            eerT2_35 = contents[i + 2][:-1]
                            EER.append(eerT2_35)
                            PDC.append(pdcT2_35)
                            T_outside.append('35')
                            Pdesignc.append(pdesignc2)
                            Temperatur.append(temperatur2)
                            Modul.append(modul)
                            Manufacturer.append(manufacturer)
                            Date.append(date)
                            Refrigerant.append(refrigerant)
                            Mass.append(mass)
                            Type.append(heatpumpType)
                    elif lines.startswith('Pdc Tj = 25°C'):
                        pdcT1_25 = contents[i][:-4]
                        if contents[i + 2].endswith('W\n'):
                            pdcT2_25 = contents[i + 2][:-4]

                    elif lines.startswith('EER Tj = 25°C'):
                        eerT1_25 = (contents[i][:-1])
                        EER.append(eerT1_25)
                        PDC.append(pdcT1_25)
                        T_outside.append('25')
                        Pdesignc.append(pdesignc1)
                        Temperatur.append(temperatur1)
                        Modul.append(modul)
                        Manufacturer.append(manufacturer)
                        Date.append(date)
                        Refrigerant.append(refrigerant)
                        Mass.append(mass)
                        Type.append(heatpumpType)
                        if temperatur2 != '':
                            eerT2_25 = contents[i + 2][:-1]
                            EER.append(eerT2_25)
                            PDC.append(pdcT2_25)
                            T_outside.append('25')
                            Pdesignc.append(pdesignc2)
                            Temperatur.append(temperatur2)
                            Modul.append(modul)
                            Manufacturer.append(manufacturer)
                            Date.append(date)
                            Refrigerant.append(refrigerant)
                            Mass.append(mass)
                            Type.append(heatpumpType)

                    elif lines.startswith('Pdc Tj = 20°C'):
                        pdcT1_20 = contents[i][:-4]
                        if contents[i + 2].endswith('W\n'):
                            pdcT2_20 = contents[i + 2][:-4]

                    elif lines.startswith('EER Tj = 20°C'):
                        eerT1_20 = (contents[i][:-1])
                        EER.append(eerT1_20)
                        PDC.append(pdcT1_20)
                        T_outside.append('20')
                        Pdesignc.append(pdesignc1)
                        Temperatur.append(temperatur1)
                        Modul.append(modul)
                        Manufacturer.append(manufacturer)
                        Date.append(date)
                        Refrigerant.append(refrigerant)
                        Mass.append(mass)
                        Type.append(heatpumpType)
                        if temperatur2 != '':
                            eerT2_20 = contents[i + 2][:-1]
                            EER.append(eerT2_20)
                            PDC.append(pdcT2_20)
                            T_outside.append('20')
                            Pdesignc.append(pdesignc2)
                            Temperatur.append(temperatur2)
                            Modul.append(modul)
                            Manufacturer.append(manufacturer)
                            Date.append(date)
                            Refrigerant.append(refrigerant)
                            Mass.append(mass)
                            Type.append(heatpumpType)
    df['Manufacturer'] = Manufacturer
    df['Model'] = Modul
    df['Date'] = Date
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y\n')
    df['Type'] = Type
    df['Refrigerant'] = Refrigerant
    df['Mass of Refrigerant [kg]'] = Mass
    df['Pdesignc'] = Pdesignc
    df['T_outside [°C]'] = T_outside
    df['T_out [°C]'] = Temperatur

    df['Pdc [kW]'] = PDC
    df['EER'] = EER

    filt = df['EER'] == 'Cdc'  # P_th too small
    df.drop(index=df[filt].index, inplace=True)
    filt = df['EER'] == 'Pdc Tj = 30°C'  # P_th too small
    df.drop(index=df[filt].index, inplace=True)
    os.chdir("../..")
    df.to_csv(os.getcwd() + r'/output/database_cooling.csv', index=False)
    os.chdir("src")


def reduce_heating_data(filename, climate='all'):
    # reduce the hplib_database_heating to a specific climate measurement series (average, warm, cold)
    # delete redundant entries
    # climate = average, warm or cold
    df = pd.read_csv('../output/' + filename)
    n_climate = 3
    if climate != 'all':
        df = df.loc[df['Climate'] == climate]
        n_climate = 1
    delete = []
    Models = df['Model'].unique().tolist()
    for model in Models:
        Modeldf = df.loc[df['Model'] == model, :]
        climate_types = df[df['Model'] == model]['Climate'].unique()
        if Modeldf.shape[0] < 8:  # Models with less than 8 datapoints and not all climate types are deleted
            delete += Modeldf.index.tolist()

    df.drop(delete, inplace=True)
    df.to_csv(r'../output/database_heating_' + climate + '.csv', index=False)


def normalize_heating_data(filename):
    data_keys = pd.read_csv(r'../output/' + filename)  # read Dataframe of all models
    Models = data_keys['Model'].unique().tolist()

    P_th_ns = []
    P_el_ns = []
    drop_models = []
    for model in Models:
        data = data_keys.loc[((data_keys['Model'] == model) & (
                    data_keys['T_out [°C]'] == 52))].copy()  # only use data of model and ref point -7/52
        if data.empty:
            drop_models.append(model)
            continue
        Pel_ref = data['P_el [W]'].array[0]  # ref Point Pel
        Pth_ref = data['P_th [W]'].array[0]  # ref Point Pth
        data_key = data_keys.loc[data_keys['Model'] == model]  # only use data of model
        data_keys.loc[data_key.index, 'P_th_n'] = data_key['P_th [W]'] / Pth_ref # get normalized Value P_th_n
        data_keys.loc[data_key.index,'P_el_n'] = data_key['P_el [W]'] / Pel_ref  # get normalized Value P_el_n
    data_keys = data_keys[~data_keys['Model'].isin(drop_models)]
    filt1 = (data_keys['P_th_n'] >= 2) & (data_keys['T_out [°C]'] == 34)
    deletemodels = data_keys.loc[filt1, ['Model']].values.tolist()
    for model in deletemodels:
        data_keys = data_keys.loc[data_keys['Model'] != model[0]]
    data_keys.drop_duplicates(
        subset=['Manufacturer', 'Model', 'Mass of Refrigerant [kg]',
                'Climate', 'T_amb [°C]', 'T_in [°C]', 'T_out [°C]'],
        inplace=True)
    data_keys.to_csv(r'../output/' + filename[:-4] + '_normalized.csv', encoding='utf-8', index=False)


def get_subtype(P_th_minus7_34, P_th_2_30, P_th_7_27, P_th_12_24):
    if (P_th_minus7_34 <= P_th_2_30):
        if (P_th_2_30 <= P_th_7_27):
            if (P_th_7_27 <= P_th_12_24):
                modus = 'On-Off'
            else:
                modus = 'Regulated'  # Inverter, 2-Stages, etc.
        else:
            modus = 'Regulated'  # Inverter, 2-Stages, etc.
    else:
        modus = 'Regulated'  # Inverter, 2-Stages, etc.
    return modus


def identify_subtypes(filename):
    # Identify Subtype like On-Off or Regulated by comparing the thermal Power output at different temperature levels:
    # -7/34 |  2/30  |  7/27  |  12/24
    # assumptions for On-Off Heatpump: if temperature difference is bigger, thermal Power output is smaller
    # assumptions for Regulated: everythin else

    data_key = pd.read_csv(r'../output/' + filename)  # read Dataframe of all models
    Models = data_key['Model'].values.tolist()
    Models = list(dict.fromkeys(Models))
    data_keymark = data_key.rename(
        columns={'P_el [W]': 'P_el', 'P_th [W]': 'P_th', 'T_in [°C]': 'T_in', 'T_out [°C]': 'T_out'})
    data_keymark['deltaT'] = data_keymark['T_out'] - data_keymark['T_in']

    Subtypelist = []
    for model in Models:
        try:
            P_thermal = []
            filt1 = data_keymark['T_out'] == 34
            Tin_minus_seven = data_keymark.loc[filt1]
            filt2 = Tin_minus_seven['Model'] == model
            Model_minus_seven = Tin_minus_seven[filt2]
            P_th_minus_seven = Model_minus_seven['P_th'].array[0]
            P_thermal.append(P_th_minus_seven)

            filt1 = data_keymark['T_out'] == 30
            T_in_plus_two = data_keymark.loc[filt1]
            filt2 = T_in_plus_two['Model'] == model
            Model_plus_two = T_in_plus_two[filt2]
            P_th_plus_two = Model_plus_two['P_th'].array[0]
            P_thermal.append(P_th_plus_two)

            filt1 = data_keymark['T_out'] == 27
            Tin_plus_seven = data_keymark.loc[filt1]
            filt2 = Tin_plus_seven['Model'] == model
            Model_plus_seven = Tin_plus_seven[filt2]
            P_th_plus_seven = Model_plus_seven['P_th'].array[0]
            P_thermal.append(P_th_plus_seven)

            filt1 = data_keymark['T_out'] == 24
            Tin_plus_twelfe = data_keymark.loc[filt1]
            filt2 = Tin_plus_twelfe['Model'] == model
            Model_plus_twelfe = Tin_plus_twelfe[filt2]
            P_th_plus_twelfe = Model_plus_twelfe['P_th'].array[0]
            P_thermal.append(P_th_plus_twelfe)
            P_thermal
            Modus = get_subtype(P_thermal[0], P_thermal[1], P_thermal[2], P_thermal[3])
        except:
            print(model)
        Subtypelist.append(Modus)
    Subtype_df = pd.DataFrame()
    Subtype_df['Model'] = Models
    Subtype_df['Subtype'] = Subtypelist
    Subtype_df
    data_key = pd.read_csv(r'../output/' + filename)  # read Dataframe of all models
    data_key = data_key.merge(Subtype_df, how='inner', on='Model')

    ##assign group:

    filt1 = (data_key['Type'] == 'Outdoor Air/Water') & (data_key['Subtype'] == 'Regulated')
    data_key.loc[filt1, 'Group'] = 1
    filt1 = (data_key['Type'] == 'Exhaust Air/Water') & (data_key['Subtype'] == 'Regulated')
    data_key.loc[filt1, 'Group'] = 7
    filt1 = (data_key['Type'] == 'Brine/Water') & (data_key['Subtype'] == 'Regulated')
    data_key.loc[filt1, 'Group'] = 2
    filt1 = (data_key['Type'] == 'Water/Water') & (data_key['Subtype'] == 'Regulated')
    data_key.loc[filt1, 'Group'] = 3

    filt1 = (data_key['Type'] == 'Outdoor Air/Water') & (data_key['Subtype'] == 'On-Off')
    data_key.loc[filt1, 'Group'] = 4
    filt1 = (data_key['Type'] == 'Exhaust Air/Water') & (data_key['Subtype'] == 'On-Off')
    data_key.loc[filt1, 'Group'] = 7
    filt1 = (data_key['Type'] == 'Brine/Water') & (data_key['Subtype'] == 'On-Off')
    data_key.loc[filt1, 'Group'] = 5
    filt1 = (data_key['Type'] == 'Water/Water') & (data_key['Subtype'] == 'On-Off')
    data_key.loc[filt1, 'Group'] = 6

    data_key = data_key[
        ['Manufacturer', 'Model', 'Date', 'Type', 'Subtype', 'Group', 'Refrigerant', 'Mass of Refrigerant [kg]',
         'SPL indoor [dBA]', 'SPL outdoor [dBA]', 'PSB [W]', 'Climate', 'T_amb [°C]', 'T_in [°C]', 'T_out [°C]',
         'P_th [W]', 'P_el [W]', 'COP', 'P_th_n', 'P_el_n']]
    filt1 = data_key['Group'] != 7
    data_key = data_key.loc[filt1]
    data_key.to_csv(r'../output/' + filename[:-4] + '_subtypes.csv', encoding='utf-8', index=False)


n_heat_params = 6
def func_heat(para, t_in, t_out):
    k1, k2, k3, k4, k5, k6 = para
    t_in = (273.15+t_in)/273.15
    t_out = (273.15+t_out)/273.15
    z_calc = (k1 + k2*t_out + k3*t_in + k4*t_out*t_in + k5*t_out**2 + k6*t_in**2)
    return z_calc


def diff_fit_heat(para, t_in, t_out, z):
    z_calc = func_heat(para, t_in, t_out)
    z_diff = z_calc - z
    return z_diff


def fit_heat(t_in, t_out, z):
    p0 = [1 for x in range(n_heat_params)]  # starting values
    a = (t_in, t_out, z)
    para, _ = scipy.optimize.leastsq(diff_fit_heat, p0, args=a)
    return para


n_cool_params = 6
def func_cool(para, t_in, t_out):
    k1, k2, k3, k4, k5, k6 = para
    z_calc = (k1 + k2*t_in + k3*t_out + k4*t_out*t_in
              + k5*t_out**2 + k6*t_in**2)
    return z_calc


def diff_fit_cool(para, t_in, t_out, z):
    z_calc = func_cool(para, t_in, t_out)
    z_diff = z_calc - z
    return z_diff


def fit_cool(t_in, t_out, z):
    p0 = [1 for x in range(n_cool_params)]  # starting values
    a = (t_in, t_out, z)
    para, _ = scipy.optimize.leastsq(diff_fit_cool, p0, args=a)
    return para


param_map_h = {'P_th': 'P_th_n', 'P_el_h': 'P_el_n', 'COP': 'COP'}


def calculate_heating_parameter(data_keys, model):
    data_key = data_keys.loc[data_keys['Model'] == model].copy()  # get data of model
    group = data_key.Group.array[0]  # get Group of model

    Pel_REF = data_key.loc[data_key['P_el_n'] == 1, ['P_el_h']].values.tolist()[0][0]
    Pth_REF = data_key.loc[data_key['P_th_n'] == 1, ['P_th']].values.tolist()[0][0]
    data_key.fillna(0, inplace=True)

    model_results = {}
    model_results['model'] = model
    for param in ['P_th', 'P_el_h', 'COP']:
        model_results[param] = fit_heat(
            data_key['T_in'], data_key['T_out'],
            data_key[param_map_h[param]])
    model_results['group'] =  group
    model_results['Pel_REF'] =  Pel_REF
    model_results['Pth_REF'] =  Pth_REF
    return model_results


def calculate_heating_parameters(filename):
    # Calculate function parameters from normalized values
    data_keys = pd.read_csv('../output/' + filename)
    data_keys = data_keys.rename(
            columns={'P_el [W]': 'P_el_h', 'P_th [W]': 'P_th',
                     'T_in [°C]': 'T_in', 'T_out [°C]': 'T_out',
                     'T_amb [°C]': 'T_amb'})

    Models = data_keys['Model'].unique().tolist()

    Group = []
    Pel_ref = []
    Pth_ref = []

    params = {}
    for param in ['P_th', 'P_el_h', 'COP']:
        params[param] = {x: [] for x in range(1, n_heat_params+1) }

    func = partial(calculate_heating_parameter, data_keys)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(func, Models)

    for result in results:
        for param in ['P_th', 'P_el_h', 'COP']:
            for x in range(1, n_heat_params+1):
                params[param][x].append(result[param][x-1])

        Group.append(result['group'])
        Pel_ref.append(result['Pel_REF'])
        Pth_ref.append(result['Pth_REF'])

    paradf = pd.DataFrame()
    paradf['Model'] = Models
    for param in ['P_th', 'P_el_h', 'COP']:
        for x in range(1, n_heat_params+1):
            paradf[f'p{x}_{param}'] = params[param][x]

    paradf['Group'] = Group
    paradf['P_el_ref'] = Pel_ref
    paradf['P_th_h_ref'] = Pth_ref

    key = pd.read_csv('../output/' + filename)
    key = key.loc[key['T_out [°C]'] == 52]
    parakey = paradf.merge(key, how='left', on='Model')
    parakey = parakey.rename(
        columns={'Group_x': 'Group', 'P_el_ref': 'P_el_ref [W]',
                  'P_th_h_ref': 'P_th_h_ref [W]'})
    parakey['COP_ref'] = parakey['P_th_h_ref [W]'] / parakey['P_el_ref [W]']
    table = parakey[
        key.columns[:-8].tolist() + ['P_el_ref [W]', 'P_th_h_ref [W]', 'COP_ref']
        + paradf.columns[1:-3].tolist()]
    table.to_csv('hplib_database.csv', encoding='utf-8', index=False)


def validation_relative_error_heating(filename):
    # Simulate every set point for every heat pump and save csv file
    df=pd.read_csv(f'../output/{filename}')
    Models = df['Model'].unique().tolist()

    for model in Models:
        df_model=df.loc[df['Model']==model]
        para=hpl.get_parameters(model)
        results=hpl.simulate(
            df_model['T_in [°C]'], df_model['T_out [°C]']-5,
            para, df_model['T_amb [°C]'])
        df.loc[df_model.index, 'P_th_sim']=results.P_th
        df.loc[df_model.index, 'P_el_h_sim']=results.P_el_h
        df.loc[df_model.index, 'COP_sim']=results.COP
    # Relative error (RE) for every set point
    df['RE_P_th']=(1-df['P_th_sim']/df['P_th [W]'])*100
    df['RE_P_el_h']=(1-df['P_el_h_sim']/df['P_el [W]'])*100
    df['RE_COP']=(1-df['COP_sim']/df['COP'])*100
    df.to_csv(f"../output/{filename.replace('.csv', '_validation.csv')}", encoding='utf-8', index=False)


def validation_mape_heating(filename):
    #calculate the mean absolute percentage error for every heat pump and save in hplib_database.csv
    df=pd.read_csv(f'../output/{filename}')
    para=pd.read_csv('../output/hplib_database_heating.csv', delimiter=',')
    para=para.loc[para['Model']!='Generic']
    Models = para['Model'].unique().tolist()
    mape_cop=[]
    mape_pel=[]
    mape_pth=[]
    for model in Models:
        df_model=df.loc[df['Model']==model]
        mape_pth.append((((df_model['P_th [W]']-df_model['P_th_sim']).abs())/df_model['P_th [W]']*100).mean())
        mape_pel.append((((df_model['P_el [W]']-df_model['P_el_h_sim']).abs())/df_model['P_el [W]']*100).mean())
        mape_cop.append((((df_model['COP']-df_model['COP_sim']).abs())/df_model['COP']*100).mean())
    para['MAPE_P_el_h']=mape_pel
    para['MAPE_COP']=mape_cop
    para['MAPE_P_th']=mape_pth
    para.to_csv('../output/hplib_database_heating.csv', encoding='utf-8', index=False)

ref_temperatures_groups = {
            1: (-7, 52), # t_in, t_out
            2: (-7, 52),
            3: (3, 52),
            4: (-7, 52),
            5: ( -7, 52),
            6: (3, 52),
            }


def add_generic(filename):
    database = pd.read_csv('hplib_database.csv')
    database = database.loc[database['Model'] != 'Generic']
    data_keys_h = df=pd.read_csv(f'../output/{filename}')
    data_keys_h = data_keys_h.loc[data_keys_h['Model'] != 'Generic']
    data_keys_h = data_keys_h.rename(
            columns={'P_el [W]': 'P_el_h', 'P_th [W]': 'P_th',
                     'T_in [°C]': 'T_in', 'T_out [°C]': 'T_out',
                     'T_amb [°C]': 'T_amb'})
    data_keys_c = pd.read_csv(
        '../output/database_cooling_reduced_normalized_validation.csv')
    data_keys_c = data_keys_c.rename(
            columns={'P_el [W]': 'P_el_h', 'P_th [W]': 'P_th',
                     'T_in [°C]': 'T_in', 'T_out [°C]': 'T_out',
                     'T_outside [°C]': 'T_amb'})
    Groups = [1, 2, 3, 4, 5, 6]
    for group in Groups:
        if group == 1:
            Type = 'Outdoor Air/Water'
            modus = 'Regulated'
        elif group == 2:
            Type = 'Brine/Water'
            modus = 'Regulated'
        elif group == 3:
            Type = 'Water/Water'
            modus = 'Regulated'
        elif group == 4:
            Type = 'Outdoor Air/Water'
            modus = 'On-Off'
        elif group == 5:
            Type = 'Brine/Water'
            modus = 'On-Off'
        elif group == 6:
            Type = 'Water/Water'
            modus = 'On-Off'

        data_key_h = data_keys_h.loc[data_keys_h['Group'] == group]

        for param in ['P_th', 'P_el_h', 'COP']:
            data_key_h=data_key_h.loc[(data_key_h[f'RE_{param}'].abs()<=15)]
        data_key_h = data_key_h.groupby(
            by=['T_amb', 'T_in', 'T_out'], as_index=False).mean()

        if group in [2, 5]:
            data_key_h['T_in'] += data_key_h['T_amb']

        if not data_key_h.empty:
            heat_avgs = {}
            for param in ['P_th', 'P_el_h', 'COP']:
                heat_avgs[param] = fit_heat(
                    data_key_h['T_in'], data_key_h['T_out'],
                    data_key_h[param_map_h[param]])
            COP_ref = func_heat(heat_avgs['COP'], *ref_temperatures_groups[group])
            heat_avgs = [item for sublist in heat_avgs.values() for item in sublist]
        else:
            heat_avgs = [None for x in range(n_heat_params*3)]


        data_key_c = data_keys_c.loc[data_keys_c['Group'] == group]
        for param in ['Pdc', 'P_el', 'EER']:
            data_key_c=data_key_c.loc[(data_key_c[f'RE_{param}'].abs()<=15)]

        if not data_key_c.empty:
            cooling_avgs = {}
            for param in ['Pdc', 'P_el', 'EER']:
                cooling_avgs[param] = fit_cool(
                    data_key_c['T_amb'], data_key_c['T_out'],
                    data_key_c[param_map_c[param]])
            # EER_ref = func_cool(cooling_avgs['EER'], 35, 7)
            cooling_avgs = [item for sublist in cooling_avgs.values() for item in sublist]
        else:
            cooling_avgs = [None for x in range(n_cool_params*3)]



        database.loc[len(database.index)] = [
            'Generic', 'Generic', '', Type, modus, group, '', '', '', '', '',
            'average', '', '', COP_ref, '', ''] + heat_avgs\
            + ['', '', ''] + cooling_avgs + ['', '', '']
    database['COP_ref'] = database['COP_ref'].round(2)
    database.to_csv('hplib_database.csv', encoding='utf-8', index=False)


def reduce_to_unique():
    # Many heat pump models have several entries
    # because of different controller or storage configurations.
    # Reduce to unique heat pump models.
    df = pd.read_csv('hplib_database.csv', delimiter=',')
    df.drop_duplicates(subset=['p1_P_th', 'p1_P_el_h', 'p1_COP'], inplace=True)
    df.to_csv('../output/hplib_database_heating.csv', encoding='utf-8', index=False)


def reduce_cooling_data():
    df_cool=pd.read_csv('../output/database_cooling.csv')
    df_heat=pd.read_csv('../output/hplib_database_heating.csv')
    df = df_cool.merge(df_heat, on='Model', how='left')#merge with the ones from heating to get Group Number
    df=df.iloc[:,:16]
    df['Pdc [W]']=df['Pdc [kW]']*1000#get W
    df.drop(columns=['Pdc [kW]','Date_y','Type_y','Manufacturer_y','Subtype','Pdesignc','Refrigerant_x','Mass of Refrigerant [kg]_x','Type_x','Date_x'], inplace=True)
    df = df.rename(columns={'Manufacturer_x': 'Manufacturer','T_out [°C]_x':'T_out [°C]'})
    df=df.loc[df['Group']==1]
    df['P_el [W]']=df['Pdc [W]']/df['EER']#add P_el
    df.to_csv('../output/database_cooling_reduced.csv',encoding='utf-8', index=False)


def normalize_and_add_cooling_data():
    df = pd.read_csv(r'../output/database_cooling_reduced.csv')
    Models = df['Model'].values.tolist()
    Models = list(dict.fromkeys(Models))
    new_df = pd.DataFrame()
    for model in Models:
        data_key = pd.read_csv(r'../output/database_cooling_reduced.csv')
        data_key = data_key.loc[data_key['Model'] == model]  # get data of model
        group = data_key.Group.array[0]  # get Group of model
        if len(data_key)==4:
            data_key1 = data_key.loc[data_key['Model'] == model]
            data_key1['T_out [°C]'] = data_key1['T_out [°C]'] + 11#the following values are based on 3 heatpumps, which have those values in the keymark
            data_key1.loc[data_key1['T_outside [°C]']==35,'P_el [W]']=data_key1.loc[data_key1['T_outside [°C]']==35,'P_el [W]'] * 0.85
            data_key1.loc[data_key1['T_outside [°C]']==30,'P_el [W]']=data_key1.loc[data_key1['T_outside [°C]']==30,'P_el [W]'] * 0.82
            data_key1.loc[data_key1['T_outside [°C]']==25,'P_el [W]']=data_key1.loc[data_key1['T_outside [°C]']==25,'P_el [W]'] * 0.77
            data_key1.loc[data_key1['T_outside [°C]']==20,'P_el [W]']=data_key1.loc[data_key1['T_outside [°C]']==20,'P_el [W]'] * 0.63
            data_key1.loc[data_key1['T_outside [°C]']==35,'EER']=data_key1.loc[data_key1['T_outside [°C]']==35,'EER'] * 1.21
            data_key1.loc[data_key1['T_outside [°C]']==30,'EER']=data_key1.loc[data_key1['T_outside [°C]']==30,'EER'] * 1.21
            data_key1.loc[data_key1['T_outside [°C]']==25,'EER']=data_key1.loc[data_key1['T_outside [°C]']==25,'EER'] * 1.20
            data_key1.loc[data_key1['T_outside [°C]']==20,'EER']=data_key1.loc[data_key1['T_outside [°C]']==20,'EER'] * 0.95
            data_key1['Pdc [W]']=data_key1['P_el [W]']*data_key1['EER']
            data_key = pd.concat([data_key, data_key1])
        df_ref_pdc=data_key.loc[(data_key['T_outside [°C]']==35) & (data_key['T_out [°C]']==7),'Pdc [W]'].values[0]
        data_key['Pdc_n']=data_key['Pdc [W]']/df_ref_pdc
        df_ref_p_el=data_key.loc[(data_key['T_outside [°C]']==35) & (data_key['T_out [°C]']==7),'P_el [W]'].values[0]
        data_key['P_el_c_n']=data_key['P_el [W]']/df_ref_p_el
        new_df = pd.concat([new_df, data_key])  # merge new Dataframe with old one
    new_df.to_csv('../output/database_cooling_reduced_normalized.csv',encoding='utf-8', index=False)


param_map_c = {'Pdc': 'Pdc_n', 'P_el_c': 'P_el_c_n', 'EER': 'EER'}


def calculate_cooling_parameter(data_keys, model):
    data_key = data_keys.loc[data_keys['Model'] == model].copy()  # get data of model
    group = data_key.Group.array[0]  # get Group of model
    Pel_REF = data_key.loc[data_key['P_el_c_n'] == 1, ['P_el_c']].values.tolist()[0][0]
    Pdc_REF = data_key.loc[data_key['Pdc_n'] == 1, ['Pdc']].values.tolist()[0][0]
    data_key.fillna(0, inplace=True)
    data = data_key.loc[data_key['T_in'] > 24] #& (data_key['T_in'] != ))]
    model_results = {}
    model_results['model'] = model

    for param in ['Pdc', 'P_el_c', 'EER']:
        model_results[param] = fit_cool(
            data['T_in'], data['T_out'],
            data[param_map_c[param]])
    model_results['group'] =  group
    model_results['Pel_REF'] =  Pel_REF
    model_results['Pdc_REF'] =  Pdc_REF
    return model_results


def calculate_cooling_parameters():
    # Calculate function parameters from normalized values
    data_keys = pd.read_csv('../output/database_cooling_reduced_normalized.csv')
    data_keys = data_keys.rename(
            columns={'P_el [W]': 'P_el_c', 'Pdc [W]': 'Pdc',
                     'T_outside [°C]': 'T_in', 'T_out [°C]': 'T_out',
                     'P_el_n': 'P_el_c_n'})

    Models = data_keys['Model'].unique().tolist()

    Group = []
    Pel_ref = []
    Pdc_ref = []
    params = {}
    for param in ['Pdc', 'P_el_c', 'EER']:
        params[param] = {x: [] for x in range(1, n_cool_params+1) }

    func = partial(calculate_cooling_parameter, data_keys)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(func, Models)

    for result in results:
        for param in ['Pdc', 'P_el_c', 'EER']:
            for x in range(1, n_cool_params+1):
                params[param][x].append(result[param][x-1])

        Group.append(result['group'])
        Pel_ref.append(result['Pel_REF'])
        Pdc_ref.append(result['Pdc_REF'])

    # write List  in Dataframe
    paradf = pd.DataFrame()
    paradf['Model'] = Models
    for param in ['Pdc', 'P_el_c', 'EER']:
        for x in range(1, n_cool_params+1):
            paradf[f'p{x}_{param}'] = params[param][x]
    paradf['P_el_cooling_ref'] = Pel_ref
    paradf['Pdc_ref'] = Pdc_ref

    hplib=pd.read_csv('../output/hplib_database_heating.csv')
    para = hplib.merge(paradf, how='left', on='Model')
    para.rename(
        columns={'P_el_cooling_ref': 'P_el_c_ref [W]',
                 'P_el_ref [W]': 'P_el_h_ref [W]', 'Pdc_ref': 'P_th_c_ref [W]',
                 'P_th_ref [W]': 'P_th_h_ref [W]'}, inplace=True)
    para = para[
        para.columns[:15].tolist() + ['P_el_c_ref [W]', 'P_th_c_ref [W]']
        + para.columns[15:-2].tolist()]
    para.to_csv('hplib_database.csv', encoding='utf-8', index=False)


def validation_relative_error_cooling():
    # Simulate every set point for every heat pump and save csv file
    df=pd.read_csv('../output/database_cooling_reduced_normalized.csv')
    i=0
    prev_model='first Model'
    while i<len(df):
        Model=df.iloc[i,1]
        T_in=df.iloc[i,2]
        T_out=df.iloc[i,3]
        try:
            if prev_model!=Model:
                para=hpl.get_parameters(Model)
            results=hpl.simulate(T_in,T_out+5,para,T_in, 2)
            df.loc[i,'Pdc_sim']=-results.Pdc[0]
            df.loc[i,'P_el_sim']=results.P_el_c[0]
            df.loc[i,'EER_sim']=results.EER[0]
            prev_model=Model
        except:
            pass
        i=i+1

    # Relative error (RE) for every set point
    df['RE_Pdc']=(df['Pdc_sim']/df['Pdc [W]']-1)*100
    df['RE_P_el']=(df['P_el_sim']/df['P_el [W]']-1)*100
    df['RE_EER']=(df['EER_sim']/df['EER']-1)*100
    df.to_csv('../output/database_cooling_reduced_normalized_validation.csv', encoding='utf-8', index=False)


def validation_mape_cooling():
    #calculate the mean absolute percentage error for every heat pump and save in hplib_database.csv
    df=pd.read_csv('../output/database_cooling_reduced_normalized_validation.csv')
    para=pd.read_csv('hplib_database.csv', delimiter=',')
    para=para.loc[para['Model']!='Generic']
    Models = para['Model'].values.tolist()
    Models = list(dict.fromkeys(Models))
    mape_eer=[]
    mape_pel=[]
    mape_pdc=[]
    for model in Models:
        df_model=df.loc[df['Model']==model]
        mape_pdc.append((((df_model['Pdc [W]']-df_model['Pdc_sim']).abs())/df_model['Pdc [W]']*100).mean())
        mape_pel.append((((df_model['P_el [W]']-df_model['P_el_sim']).abs())/df_model['P_el [W]']*100).mean())
        mape_eer.append((((df_model['EER']-df_model['EER_sim']).abs())/df_model['EER']*100).mean())
    para['MAPE_P_el_cooling']=mape_pel
    para['MAPE_EER']=mape_eer
    para['MAPE_Pdc']=mape_pdc
    para.to_csv('hplib_database.csv', encoding='utf-8', index=False)
