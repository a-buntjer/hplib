"""
The ``hplib`` module provides a set of functions for simulating the performance of heat pumps.
"""
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from typing import Any, Tuple
import os
import numpy
import hplib_database as db


def load_database() -> pd.DataFrame:
    """
    Loads data from hplib_database.

    Returns
    -------
    df : pd.DataFrame
        Content of the database
    """
    df = pd.read_csv(cwd()+r'/hplib_database.csv')
    return df


def get_parameters(model: str, group_id: int = 0,
                   t_in: int = 0, t_out: int = 0, p_th: int = 0) -> pd.DataFrame:
    """
    Loads the content of the database for a specific heat pump model
    and returns a pandas ``DataFrame`` containing the heat pump parameters.

    Parameters
    ----------
    model : str
        Name of the heat pump model or "Generic".
    group_id : numeric, default 0
        only for model "Generic": Group ID for subtype of heat pump. [1-6].
    t_in : numeric, default 0
        only for model "Generic": Input temperature :math:`T` at primary side of the heat pump. [°C]
    t_out : numeric, default 0
        only for model "Generic": Output temperature :math:`T` at secondary side of the heat pump. [°C]
    p_th : numeric, default 0
        only for model "Generic": Thermal output power at setpoint t_in, t_out (and for
        water/water, brine/water heat pumps t_amb = -7°C). [W]

    Returns
    -------
    parameters : pd.DataFrame
        Data frame containing the model parameters.
    """
    parameters = pd.read_csv(cwd()+r'/hplib_database.csv', delimiter=',')
    parameters = parameters.loc[parameters['Model'] == model]
    parameters.rename(
        columns={'P_el_cooling_ref': 'P_el_c_ref [W]',
                 'P_el_ref [W]': 'P_el_h_ref [W]',
                 'Pdc_ref': 'P_th_c_ref [W]',
                 'P_th_ref [W]': 'P_th_h_ref [W]'}, inplace=True)
    parameters.drop(['Date', 'Type', 'Subtype',
       'Refrigerant', 'Mass of Refrigerant [kg]', 'SPL indoor [dBA]',
       'SPL outdoor [dBA]', 'PSB [W]', 'Climate'], axis=1, inplace=True)


    if model == 'Generic':
        parameters = parameters[parameters['Group'] == group_id]

        p_th_ref, p_el_ref = fit_p_th_ref(t_in, t_out, group_id, p_th, parameters)
        parameters.loc[:, 'P_th_h_ref [W]'] = p_th_ref
        parameters.loc[:, 'P_el_h_ref [W]'] = p_el_ref
    return parameters


def fit_p_th_ref(t_in: int, t_out: int, group_id: int,
                 p_th_set_point: int, parameters: pd.DataFrame) -> Any:
    """
    Determine the thermal output power in [W] at reference conditions (T_in = [-7, 0, 10] ,
    T_out=52, T_amb=-7) for a given set point for a generic heat pump, using a least-square method.

    Parameters
    ----------
    t_in : numeric
        Input temperature :math:`T` at primary side of the heat pump. [°C]
    t_out : numeric
        Output temperature :math:`T` at secondary side of the heat pump. [°C]
    group_id : numeric
        Group ID for a parameter set which represents an average heat pump of its group.
    p_th_set_point : numeric
        Thermal output power. [W]

    Returns
    -------
    p_th : Any
        Thermal output power. [W]
    """
    cop_ref = parameters.iloc[0]['COP_ref']
    p_th_0 = [p_th_set_point]  # starting values
    a = (t_in, t_out, group_id, p_th_set_point, parameters.copy())
    p_th_ref, _ = scipy.optimize.leastsq(fit_func_p_th_ref, p_th_0, args=a, epsfcn=10)
    p_el_ref = p_th_ref / cop_ref
    return p_th_ref, p_el_ref


def fit_func_p_th_ref(
        p_th:  float, t_in: float, t_out: float, group_id: int,
        p_th_set_point: float, parameters: pd.DataFrame) -> float:
    """
    Helper function to determine difference between given and calculated
    thermal output power in [W].

    Parameters
    ----------
    p_th : numeric
        Thermal output power. [W]
    t_in : numeric
        Input temperature :math:`T` at primary side of the heat pump. [°C]
    t_out : numeric
        Output temperature :math:`T` at secondary side of the heat pump. [°C]
    group_id : numeric
        Group ID for a parameter set which represents an average heat pump of its group.
    p_th_set_point : numeric
        Thermal output power. [W]

    Returns
    -------
    p_th_diff : numeric
        Thermal output power. [W]
    """
    t_amb = t_in
    parameters.loc[:, 'P_th_h_ref [W]'] = p_th[0]
    parameters.loc[:, 'P_el_h_ref [W]'] = p_th[0]/parameters['COP_ref']

    p_th_calc = simulate(t_in, t_out - 5, parameters, t_amb).iloc[0]['P_th']
    p_th_diff = p_th_set_point - p_th_calc
    return p_th_diff


def simulate(t_in_primary: any, t_in_secondary: any, parameters: pd.DataFrame,
             t_amb: any, modus: int = 1) -> pd.DataFrame:
    """
    Performs the simulation of the heat pump model.

    Parameters
    ----------
    t_in_primary : numeric or iterable (e.g. pd.Series)
        Input temperature on primry side :math:`T` (air, brine, water). [°C]
    t_in_secondary : numeric or iterable (e.g. pd.Series)
        Input temperature on secondary side :math:`T` from heating storage or system. [°C]
    parameters : pd.DataFrame
        Data frame containing the heat pump parameters from hplib.getParameters().
    t_amb : numeric or iterable (e.g. pd.Series)
        Ambient temperature :math:'T' of the air. [°C]
    modus : int
        for heating: 1, for cooling: 2

    Returns
    -------
    df : pd.DataFrame
        with the following columns
        T_in = Input temperature :math:`T` at primary side of the heat pump. [°C]
        T_out = Output temperature :math:`T` at secondary side of the heat pump. [°C]
        COP = Coefficient of performance.
        P_el = Electrical input Power. [W]
        P_th = Thermal output power. [W]
        m_dot = Mass flow at secondary side of the heat pump. [kg/s]
    """

    DELTA_T = 5 # Inlet temperature is supposed to be heated up by 5 K
    CP = 4200  # J/(kg*K), specific heat capacity of water
    t_in = t_in_primary#info value for dataframe
    group_id = parameters['Group'].array[0]


    params = {}
    for param in ['Pdc', 'P_el_c', 'EER']:
        params[param] = {x: [] for x in range(1, db.n_cool_params+1)}
    for param in ['P_th', 'P_el_h', 'COP']:
        params[param] = {x: [] for x in range(1, db.n_heat_params+1)}

    for param in ['P_th', 'P_el_h', 'COP']:
        for x in range(1, db.n_heat_params+1):
            params[param][x].append(parameters[f'p{x}_{param}'].array[0])
    for param in ['Pdc', 'P_el_c', 'EER']:
        for x in range(1, db.n_cool_params+1):
            if f'p{x}_{param}' in parameters.columns:
                params[param][x].append(parameters[f'p{x}_{param}'].array[0])
            else:
                params[param][x].append(numpy.nan)

    series_t = [t for t in [t_in, t_in_secondary, t_amb]
                if isinstance(t, pd.core.series.Series)]
    if series_t:
        index=series_t[0].index
    else:
        index=[0]

    df = pd.DataFrame(
        index=index,
        columns=['T_in', 'T_out', 'COP', 'P_el_h', 'P_th', 'EER',
                 'P_el_c', 'Pdc', 'm_dot'])

    df['Modus']=modus
    df['T_in'] = t_in
    df['T_in'] = df['T_in'].clip(upper=16)
    # if group_id in [2, 3, 5, 6]:
    #     df['T_in'] += t_amb
    df['T_out'] = t_in_secondary

    df.loc[df['Modus']==1,'T_out'] = df['T_out'] + DELTA_T
    df.loc[df['Modus']==2,'T_out'] = df['T_out'] - DELTA_T

    if modus == 1:
        COP_params = tuple(
            parameters[f'p{x}_COP'].array[0]
            for x in range(1, db.n_heat_params+1))
        df.loc[df['Modus']==1, 'COP'] = db.func_heat(
                COP_params, df['T_in'], df['T_out'])

        P_el_h_params = tuple(
            parameters[f'p{x}_P_el_h'].array[0]
            for x in range(1, db.n_heat_params+1))
        df.loc[df['Modus']==1, 'P_el_h'] = db.func_heat(
            P_el_h_params, df['T_in'], df['T_out'])\
                * parameters['P_el_h_ref [W]'].array[0]

        P_th_params = tuple(
            parameters[f'p{x}_P_th'].array[0]
            for x in range(1, db.n_heat_params+1))
        df.loc[df['Modus']==1, 'P_th'] = db.func_heat(
            P_th_params, df['T_in'], df['T_out'])\
                * parameters['P_th_h_ref [W]'].array[0]

        # df.loc[df['Modus']==1, 'P_th'] = df.loc[df['Modus']==1, 'P_el_h']\
        #     * df.loc[df['Modus']==1, 'COP']
    else:
        EER_params = tuple(
            parameters[f'p{x}_EER'].array[0]
            for x in range(1, db.n_cool_params+1))
        df.loc[df['Modus']==2, 'EER'] = db.func_cool(
                    EER_params, df['T_in'], df['T_out'])

        P_el_c_params = tuple(
            parameters[f'p{x}_P_el_c'].array[0]
            for x in range(1, db.n_cool_params+1))
        df.loc[df['Modus']==2, 'P_el_c'] = db.func_cool(
            EER_params, df['T_in'], df['T_out'])\
                * parameters['P_el_c_ref [W]'].array[0]
        df.loc[df['Modus']==2, 'Pdc'] = df.loc[df['Modus']==2, 'P_el_c']\
             * df.loc[df['Modus']==2, 'EER']

    df['m_dot'] = (df['P_th']/(DELTA_T * CP)).astype(float)
    df['P_el_h'] = df['P_el_h'].astype(float).round(0)
    df['P_el_c'] = df['P_el_c'].astype(float).round(0)
    df['Pdc'] = df['Pdc'].astype(float).round(0)
    df['P_th'] = df['P_th'].astype(float).round(0)
    df['EER'] = df['EER'].astype(float).round(2)
    df['COP'] = df['COP'].astype(float).round(2)
    return df


def cwd():
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    return dir_path


class HeatPump:
    def __init__(self, parameters: pd.DataFrame):
        self.group_id = float(parameters['Group'].array[0])
        self.p1_p_el = float(parameters['p1_P_el [1/°C]'].array[0])
        self.p2_p_el = float(parameters['p2_P_el [1/°C]'].array[0])
        self.p3_p_el = float(parameters['p3_P_el [-]'].array[0])
        self.p4_p_el = float(parameters['p4_P_el [1/°C]'].array[0])
        self.p1_cop = float(parameters['p1_COP [-]'].array[0])
        self.p2_cop = float(parameters['p2_COP [-]'].array[0])
        self.p3_cop = float(parameters['p3_COP [-]'].array[0])
        self.p4_cop = float(parameters['p4_COP [-]'].array[0])
        self.p_el_ref = float(parameters['P_el_h_ref [W]'].array[0])
        self.p_th_ref = float(parameters['P_th_h_ref [W]'].array[0])
        try:
            self.p1_eer = parameters['p1_EER [-]'].array[0]
            self.p2_eer = parameters['p2_EER [-]'].array[0]
            self.p3_eer = parameters['p3_EER [-]'].array[0]
            self.p4_eer = parameters['p4_EER [-]'].array[0]
            self.p5_p_el = parameters['p5_P_el [1/°C]'].array[0]
            self.p6_p_el = parameters['p6_P_el [1/°C]'].array[0]
            self.p7_p_el = parameters['p7_P_el [-]'].array[0]
            self.p8_p_el = parameters['p8_P_el [1/°C]'].array[0]
            self.p_el_col_ref=parameters['P_el_c_ref [W]'].array[0]
        except:
            self.p1_eer = numpy.nan
            self.p2_eer = numpy.nan
            self.p3_eer = numpy.nan
            self.p4_eer = numpy.nan
            self.p5_p_el = numpy.nan
            self.p6_p_el = numpy.nan
            self.p7_p_el = numpy.nan
            self.p8_p_el = numpy.nan
            self.p_el_col_ref=numpy.nan

        self.delta_t = 5  # Inlet temperature is supposed to be heated up by 5 K
        self.cp = 4200  # J/(kg*K), specific heat capacity of water

    def simulate(self, t_in_primary: float, t_in_secondary: float, modus: int = 1) -> dict:
        """
        Performs the simulation of the heat pump model.

        Parameters
        ----------
        t_in_primary : numeric or iterable (e.g. pd.Series)
            Input temperature on primry side :math:`T` (air, brine, water). [°C]
        t_in_secondary : numeric or iterable (e.g. pd.Series)
            Input temperature on secondary side :math:`T` from heating storage or system. [°C]
        parameters : pd.DataFrame
            Data frame containing the heat pump parameters from hplib.getParameters().
        t_amb : numeric or iterable (e.g. pd.Series)
            Ambient temperature :math:'T' of the air. [°C]
        modus : int
            for heating: 1, for cooling: 2

        Returns
        -------
        df : pd.DataFrame
            with the following columns
            T_in = Input temperature :math:`T` at primary side of the heat pump. [°C]
            T_out = Output temperature :math:`T` at secondary side of the heat pump. [°C]
            T_amb = Ambient / Outdoor temperature :math:`T`. [°C]
            COP = Coefficient of performance.
            P_el = Electrical input Power. [W]
            P_th = Thermal output power. [W]
            m_dot = Mass flow at secondary side of the heat pump. [kg/s]
        """

        t_in = t_in_primary  # info value for dataframe
        if modus==1:
            t_out = t_in_secondary + self.delta_t #Inlet temperature is supposed to be heated up by 5 K
            eer=numpy.nan
        if modus==2: # Inlet temperature is supposed to be cooled down by 5 K
            t_out = t_in_secondary - self.delta_t
            cop=numpy.nan
        # for subtype = air/water heat pump
        if self.group_id in (1, 4):
            t_amb = t_in
        t_ambient=t_amb
        # for regulated heat pumps
        if self.group_id in (1, 2, 3):
            if modus==1:
                cop = self.p1_cop * t_in + self.p2_cop * t_out + self.p3_cop + self.p4_cop * t_amb

                p_el = (self.p1_p_el * t_in
                        + self.p2_p_el * t_out
                        + self.p3_p_el
                        + self.p4_p_el * t_amb) * self.p_el_ref

                if self.group_id == 1:
                    t_in = -7
                    t_amb = t_in
                elif self.group_id == 2:
                    t_amb = -7

                # 25% of Pel @ -7°C T_amb = T_in
                if p_el < 0.25 * self.p_el_ref * (self.p1_p_el * t_in
                                                + self.p2_p_el * t_out
                                                + self.p3_p_el
                                                + self.p4_p_el * t_amb):

                    p_el = 0.25 * self.p_el_ref * (self.p1_p_el * t_in
                                                + self.p2_p_el * t_out
                                                + self.p3_p_el
                                                + self.p4_p_el * t_amb)

                p_th = p_el * cop

                if cop <= 1:
                    cop = 1
                    p_el = self.p_th_ref
                    p_th = self.p_th_ref
            if modus==2:
                eer = (self.p1_eer * t_in + self.p2_eer * t_out + self.p3_eer + self.p4_eer * t_amb)
                if t_in<25:
                    t_in=25
                    t_amb=t_in
                p_el = (self.p5_p_el * t_in + self.p6_p_el * t_out + self.p7_p_el + self.p8_p_el * t_amb) * self.p_el_col_ref
                if p_el<0:
                    eer = numpy.nan
                    p_el = numpy.nan
                p_th = -(eer*p_el)
                if eer < 1:
                    eer = numpy.nan
                    p_el = numpy.nan
                    p_th = numpy.nan

        # for subtype = On-Off
        elif self.group_id in (4, 5, 6):
            p_el = (self.p1_p_el * t_in
                    + self.p2_p_el * t_out
                    + self.p3_p_el
                    + self.p4_p_el * t_amb) * self.p_el_ref

            cop = self.p1_cop * t_in + self.p2_cop * t_out + self.p3_cop + self.p4_cop * t_amb

            p_th = p_el * cop

            if cop <= 1:
                cop = 1
                p_el = self.p_th_ref
                p_th = self.p_th_ref

        # massflow
        m_dot = abs(p_th / (self.delta_t * self.cp))

        # round
        result = dict()

        result['T_in'] = t_in_primary
        result['T_out'] = t_out
        result['T_amb'] = t_ambient
        result['COP'] = cop
        result['EER'] = eer
        result['P_el'] = p_el
        result['P_th'] = p_th
        result['m_dot']= m_dot

        return result
