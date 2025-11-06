"""
Carbonate Chemistry Calculations Module

This module provides functions for calculating carbonate chemistry parameters
including alkalinity, pH, and omega saturation states from pCO2 and tCO2/DIC.

Based on the calculations from pCO2toGoNotebook.ipynb
"""

import numpy as np
import math
from typing import Tuple, Dict, Any
import gsw

############ MANIPULATOR CALCULATIONS ############

def manipulator(
        Src_Alk: float, 
        Src_TCO2: float, 
        Trgt_Alk: float, 
        Trgt_TCO2: float,
        CRgt_Alk: float, 
        Argt_HCL: float,
        CRgt_TCO2: float, 
        source_TCO2: float, 
        Buffered_volume: float, 
        Argt_vol: float, 
        Crgt_vol: float
) -> Dict[str, float]:
    """
    Calculate carbonate chemistry parameters from pCO2 and TCO2/DIC.
    
    Args:
        Src_Alk:  
        Src_TCO2:  
        Trgt_Alk:  
        Trgt_TCO2: 
        CRgt_Alk:  
        Argt_HCL: 
        CRgt_TCO2:  
        source_TCO2:  
        Buffered_volume:  
        Argt_vol:  
        Crgt_vol: 
        
    Returns:
        Dictionary containing:
        - chk_tgt_TCO2: 
        - CrgtVol: 
        - ArgtVol: 
        
    Raises:
        ValueError: If the polynomial solver fails to find valid roots
    """
    
    # Check the 
    chk_tgt_TCO2 = source_TCO2*Buffered_volume + CRgt_TCO2*Crgt_vol

    # Calculate some of the variables
    gamma = (Src_TCO2-Trgt_TCO2)/(Trgt_TCO2-CRgt_TCO2)
    alpha = (20-Trgt_TCO2)/(Trgt_TCO2-CRgt_TCO2)
    # FA numerator
    FAnum = Src_Alk-Trgt_Alk + gamma*(CRgt_Alk-Trgt_Alk)
    # FA denominator
    FAden = Argt_HCL+Trgt_Alk + alpha*(Trgt_Alk-CRgt_Alk)

    # calculate ArgtVol
    ArgtVol = FAnum/FAden*Buffered_volume

    # Calcuate Crgt
    CrgtVol = gamma*Buffered_volume + 16*alpha*FAnum/FAden

    return {
        'chk_tgt_TCO2': chk_tgt_TCO2,
        'CrgtVol': CrgtVol,
        'ArgtVol': ArgtVol
    }


############ CARBONATE CHEMISTRY CALCULATIONS ############

def calculate_carbonate_chemistry_pCO2_tCO2(
    temperature: float,
    salinity: float, 
    pCO2: float,
    tCO2: float
) -> Dict[str, float]:
    """
    Calculate carbonate chemistry parameters from pCO2 and TCO2/DIC.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pCO2: Partial pressure of CO2 in microatm (150-750 range)
        tCO2: Total dissolved inorganic carbon in umol/L (20-5000 range)
        
    Returns:
        Dictionary containing:
        - CO3: Carbonate ion concentration (umol/L)
        - HCO3: Bicarbonate ion concentration (umol/L) 
        - CO2aq: Aqueous CO2 concentration (umol/L)
        - pCO2: Partial pressure of CO2 in microatm (150-750 range)
        - pH: pH (unitless, 7.5-8.5 range)
        - Omega_aragonite: Aragonite saturation state (1.1-4.2 range)
        - Omega_calcite: Calcite saturation state
        - alkalinity: Total alkalinity (umol/L)
        - tCO2: Total dissolved inorganic carbon in umol/L (20-5000 range)
        
    Raises:
        ValueError: If the polynomial solver fails to find valid roots
    """

    ### CALCULATE THE CONSTANTS ###
    # Thermodynamic constants (equilibrium constants)
    Kca = calculate_Kca(salinity, temperature)
    Kar = calculate_Kar(salinity, temperature)
    Kh = calculate_Kh(salinity, temperature)
    K1 = calculate_K1(salinity, temperature)
    K2 = calculate_K2(salinity, temperature)
    Kb = calculate_Kb(salinity, temperature)
    Kw = calculate_Kw(salinity, temperature)
    Kb12 = calculate_Kb12(Kb, K1, K2)
    Kw12 = calculate_Kw12(Kw, K1, K2)

    # CALCULATE ANCILLIARY CONSTANTS
    density = seawater_density(salinity, temperature)
    calcium = calculate_calcium(salinity) # preliminary calculation used for all cases
    CO2aq = Kh*pCO2
    Hplus = np.nan # 10**(-1*pH) 
    total_boron = calculate_total_boron(salinity) # preliminary calculation used for all cases
        
    # Calculate grouped constants for polynomial solution
    const_a = (K1/K2)**0.5
    const_b = CO2aq**0.5
    const_g = tCO2 - CO2aq
    const_h = const_a * const_b

    ### SOLVE FOR CO3 ###
    # Set up polynomial coefficients for solving CO3
    # Polynomial: x^2 - (const_h^2 + 2*const_g)*x + const_g^2 = 0
    coefficients = np.array([1, -(const_h**2 + 2*const_g), const_g**2])
    
    # Solve polynomial for CO3 concentration
    roots = np.roots(coefficients)
    
    # CHECK FOR VALID ROOTS 
    tCO2_check = 0.5 * tCO2
    CO3 = None
    
    for root in roots:
        real_root = root.real
        imag_root = root.imag
        
        # Check if root meets validity conditions
        if (real_root <= tCO2_check and 
            real_root >= 0 and 
            abs(imag_root) < 1e-10):  # Essentially zero imaginary part
            CO3 = real_root
            break
    
    if CO3 is None:
        raise ValueError("Polynomial solver failed to find valid CO3 concentration")
    
    ## CALCULATE OTHER COARBONATE SPECIES
    # Calculate other carbonate species
    HCO3 = const_a * ((CO2aq * CO3)**0.5)
    
    # Calculate hydrogen ion concentration and pH
    Hplus = (CO2aq / CO3 * K1 * K2)**0.5 # Hydrogen ion concentration [mol/L]
    pH = -math.log10(Hplus)
    
    # Calculate omega saturation states
    omega_aragonite = (CO3 * calcium * 1000) / (Kar * 1e12)
    omega_calcite = (CO3 * calcium * 1000) / (Kca * 1e12)
    
    # Calculate alkalinity components
    OHminus = 1e6 * Kw / Hplus # Hydroxide ion concentration (umol/L)
    Bminus = (Kb * total_boron) / (Hplus + Kb) # Borate ion concentration [umol/L]
    alkalinity = 2 * CO3 + HCO3 + OHminus + Bminus - Hplus
    
    return {
        'CO3': CO3,
        'HCO3': HCO3,
        'CO2aq': CO2aq,
        'pCO2': pCO2,
        'pH': pH,
        'Omega_aragonite': omega_aragonite,
        'Omega_calcite': omega_calcite,
        'alkalinity': alkalinity,
        'tCO2': tCO2,
        'temperature': temperature,
        'salinity': salinity
    }

def calculate_carbonate_chemistry_pCO2_alkalinity(
    temperature: float,
    salinity: float, 
    pCO2: float,
    alkalinity: float
) -> Dict[str, float]:
    """
    Calculate carbonate chemistry parameters from pCO2 and Alkalinity.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pCO2: Partial pressure of CO2 in microatm (150-750 range)
        Alkalinity: Total alkalinity (umol/L)
        
    Returns:
        Dictionary containing:
        - CO3: Carbonate ion concentration (umol/L)
        - HCO3: Bicarbonate ion concentration (umol/L) 
        - CO2aq: Aqueous CO2 concentration (umol/L)
        - pCO2: Partial pressure of CO2 in microatm (150-750 range)
        - pH: pH (unitless, 7.5-8.5 range)
        - Omega_aragonite: Aragonite saturation state (1.1-4.2 range)
        - Omega_calcite: Calcite saturation state
        - alkalinity: Total alkalinity (umol/L)
        - tCO2: Total dissolved inorganic carbon in umol/L (20-5000 range)
        
    Raises:
        ValueError: If the polynomial solver fails to find valid roots
    """

    ### CALCULATE THE CONSTANTS ###
    # Thermodynamic constants (equilibrium constants)
    Kca = calculate_Kca(salinity, temperature)
    Kar = calculate_Kar(salinity, temperature)
    Kh = calculate_Kh(salinity, temperature)
    K1 = calculate_K1(salinity, temperature)
    K2 = calculate_K2(salinity, temperature)
    Kb = calculate_Kb(salinity, temperature)
    Kw = calculate_Kw(salinity, temperature)
    Kb12 = calculate_Kb12(Kb, K1, K2)
    Kw12 = calculate_Kw12(Kw, K1, K2)

    # CALCULATE ANCILLIARY CONSTANTS
    density = seawater_density(salinity, temperature)
    calcium = calculate_calcium(salinity) # preliminary calculation used for all cases
    CO2aq = Kh*pCO2
    Hplus = np.nan # 10**(-1*pH) 
    total_boron = calculate_total_boron(salinity) # preliminary calculation used for all cases
        
    # Calculate grouped constants for polynomial solution
    const_A = (K1/K2)**0.5
    const_B = np.sqrt(CO2aq)
    const_C = total_boron
    const_D = np.sqrt(CO2aq)/Kb12
    const_E = Kw12/np.sqrt(CO2aq)
    const_F = np.sqrt(CO2aq)*np.sqrt(K1 * K2)

    const_G = -1*const_F*const_D
    const_H = -1*(const_D*alkalinity + const_F)
    const_I = (const_C+const_D*(const_E+const_A*const_B)) - alkalinity
    const_J = (const_D*2)+(const_A*const_B + const_E)

    ### SOLVE FOR CO3 ###
    # Set up polynomial coefficients for solving CO3
    # Polynomial: x^2 - (const_h^2 + 2*const_g)*x + const_g^2 = 0
    coefficients = np.array([2, const_J, const_I, const_H, const_G] )
    
    # Solve polynomial for CO3 concentration
    roots = np.roots(coefficients)
    
    # CHECK FOR VALID ROOTS 
    alkalinity_check = 0.7*np.sqrt(alkalinity)
    CO3 = None
    
    for root in roots:
        real_root = root.real
        imag_root = root.imag
        
        # Check if root meets validity conditions
        if (real_root <= alkalinity_check and 
            real_root >= 0 and 
            abs(imag_root) < 1e-10):  # Essentially zero imaginary part
            CO3 = real_root
            break
    
    if CO3 is None:
        raise ValueError("Polynomial solver failed to find valid CO3 concentration")
    
    ## CALCULATE OTHER COARBONATE SPECIES
    CO3 = CO3*CO3

    # Calculate other carbonate species
    HCO3 = np.sqrt(CO3*CO2aq)*const_A
    
    # Calculate hydrogen ion concentration and pH
    # Hplus = (CO2aq / CO3 * K1 * K2)**0.5 # Hydrogen ion concentration [mol/L]
    pH = -1*np.log10(np.sqrt(CO2aq/CO3 * K1 * K2))
    
    # Calculate omega saturation states
    omega_aragonite = CO3*(1000*calcium)/(Kar*1e12)
    omega_calcite = (CO3*(calcium*1000))/(Kca*1e12)

    # calculate Alkalinity
    tCO2 = CO2aq+CO3+HCO3
    
    return {
        'CO3': CO3,
        'HCO3': HCO3,
        'CO2aq': CO2aq,
        'pCO2': pCO2,
        'pH': pH,
        'Omega_aragonite': omega_aragonite,
        'Omega_calcite': omega_calcite,
        'alkalinity': alkalinity,
        'tCO2': tCO2,
        'temperature': temperature,
        'salinity': salinity
    }

def calculate_carbonate_chemistry_pCO2_OmegaA(
    temperature: float,
    salinity: float, 
    pCO2: float,
    omega_aragonite: float
) -> Dict[str, float]:
    """
    Calculate carbonate chemistry parameters from pCO2 and Omega aragonite.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pCO2: Partial pressure of CO2 in microatm (150-750 range)
        Omega_aragonite: Aragonite saturation state
        
    Returns:
        Dictionary containing:
        - CO3: Carbonate ion concentration (umol/L)
        - HCO3: Bicarbonate ion concentration (umol/L) 
        - CO2aq: Aqueous CO2 concentration (umol/L)
        - pCO2: Partial pressure of CO2 in microatm (150-750 range)
        - pH: pH (unitless, 7.5-8.5 range)
        - Omega_aragonite: Aragonite saturation state (1.1-4.2 range)
        - Omega_calcite: Calcite saturation state
        - alkalinity: Total alkalinity (umol/L)
        - tCO2: Total dissolved inorganic carbon in umol/L (20-5000 range)
        
    Raises:
        ValueError: If the polynomial solver fails to find valid roots
    """

    ### CALCULATE THE CONSTANTS ###
    # Thermodynamic constants (equilibrium constants)
    Kca = calculate_Kca(salinity, temperature)
    Kar = calculate_Kar(salinity, temperature)
    Kh = calculate_Kh(salinity, temperature)
    K1 = calculate_K1(salinity, temperature)
    K2 = calculate_K2(salinity, temperature)
    Kb = calculate_Kb(salinity, temperature)
    Kw = calculate_Kw(salinity, temperature)
    # Kb12 = calculate_Kb12(Kb, K1, K2)
    # Kw12 = calculate_Kw12(Kw, K1, K2)

    # CALCULATE ANCILLIARY CONSTANTS
    # density = seawater_density(salinity, temperature)
    calcium = calculate_calcium(salinity) 
    CO2aq = Kh*pCO2
    total_boron = calculate_total_boron(salinity) # preliminary calculation used for all cases
    
    # calculate omega calcite saturation state
    omega_calcite = omega_aragonite*Kar/Kca

    # calculate carbonate species
    CO3 = 1e9 * omega_aragonite * Kar / calcium
    HCO3 = np.sqrt(CO3*CO2aq*K1/K2)
    tCO2 = HCO3+CO3+CO2aq

    # calculate pH
    Hplus = CO2aq*K1/HCO3
    pH = -1*np.log10(Hplus)

    # calculate alkalinity
    alkalinity = 2*CO3 + HCO3 + (total_boron*Kb)/(Hplus + Kb) + 1e6 * Kw /Hplus - 1e6 * Hplus
    
    return {
        'CO3': CO3,
        'HCO3': HCO3,
        'CO2aq': CO2aq,
        'pCO2': pCO2,
        'pH': pH,
        'Omega_aragonite': omega_aragonite,
        'Omega_calcite': omega_calcite,
        'alkalinity': alkalinity,
        'tCO2': tCO2,
        'temperature': temperature,
        'salinity': salinity
    }

def print_calculation_summary( result: Dict[str, float] ):
    # print(f"*** {calculation_pair} SUMMARY ***")
    print(f"CO3 = {result["CO3"]}")
    print(f"HCO3 = {result["HCO3"]}")
    print(f"CO2aq = {result["CO2aq"]}")
    print(f"pCO2 = {result["pCO2"]}")
    print(f"pH = {result["pH"]}")
    print(f"Omega A = {result["Omega_aragonite"]}")
    print(f"Omega C = {result["Omega_calcite"]}")
    print(f"Alkalinity = {result["alkalinity"]}")
    print(f"tCO2 = {result["tCO2"]}")


# ****** CALCULATE THERMODYNAMIC CONSTANTS ******
# Kca, Kar, Kh, K1, K2, Kb, Kw, Kb12, K1/K2, K1*K2, Kw12
# 
def calculate_Kca(salinity: float, temperature: float) -> float:
    """
    Calculate Kca from salinity and temperature
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        pressure: Water pressure in dbar 
        
    Returns:
        Kca in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly = (171.9065+(0.077993*TK))-2839.319/TK - (71.595*np.log10(TK)) - ((-0.77712+TK*0.0028426 +178.34/TK)*np.sqrt(salinity)) + (0.07711*salinity) - 0.0041249*(salinity**1.5) - 0.02
    # calculate Kca
    Kca = 10**(-1*poly)

    return Kca


def calculate_Kar(salinity: float, temperature: float) -> float:
    """
    Calculate Kar from salinity and temperature following CaCO3_sol.vi
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        Kar in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly = (171.9065+(0.077993*TK))-2839.319/TK - (71.595*np.log10(TK)) - ((-0.77712+TK*0.0028426 +178.34/TK)*np.sqrt(salinity)) + (0.07711*salinity) - 0.0041249*(salinity**1.5) - 0.02
    # calculate Kar
    Kar = 10**(-1*(poly+0.0385-63.974/TK))

    return Kar


def calculate_Kh(salinity: float, temperature: float) -> float:
    """
    Calculate Weiss CO2 solubility (Kh) from salinity and temperature following Weiss_Kh.vi
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        Kh in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly = 9345.17/TK - 167.8108 + 23.3585*np.log(TK) +salinity*(4.7036e-7*TK**2 + 0.023517 - 0.00023656*TK)
    # calculate Kh
    Kh = np.exp(poly)

    return Kh


def calculate_K1(salinity: float, temperature: float) -> float:
    """
    Calculate Millero 2010 K1 from salinity and temperature following Millero_K1K2.vi
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        Kh in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly0 = 19.568224*np.log(TK) + 6320.813/TK - 126.34048
    poly1 = 13.40511173181*np.sqrt(salinity) + 0.03184972750547*salinity - 5.218336451311e-5*salinity**2
    poly2 = (-531.0949512384*np.sqrt(salinity) + -5.778928418011*salinity)/TK
    poly3 = -2.066275370119*np.log(TK)*np.sqrt(salinity) 
    # calculate K1
    K1 = 10**(-1*(poly0+poly1+poly2+poly3))

    return K1


def calculate_K2(salinity: float, temperature: float) -> float:
    """
    Calculate Millero 2010 K2 from salinity and temperature following Millero_K1K2.vi
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        K2 in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly0 = 14.613358*np.log(TK) + 5143.692/TK - 90.18333
    poly1 = 21.57241969749*np.sqrt(salinity) + 0.1212374508709*salinity - 0.0003714066864794*salinity**2
    poly2 = (-798.2916387922*np.sqrt(salinity) + -18.95099792607*salinity)/TK
    poly3 = -3.402872930641*np.sqrt(salinity)*np.log(TK)
    # calculate K1
    K2 = 10**(-1*(poly0+poly1+poly2+poly3))

    return K2


def calculate_Kb(salinity: float, temperature: float) -> float:
    """
    Calculate Dicksons 1990's Kb from salinity and temperature following Dickson_Kb.vi
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        Kb in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly0 = (-8966.9 - 2890.53*np.sqrt(salinity) - 77.942*salinity +1.728*salinity**1.5 - 0.0996*salinity**2)/TK
    poly1 = 148.0248 + 137.1942*np.sqrt(salinity) + 1.62142*salinity
    poly2 = (-24.4344 + -25.085*np.sqrt(salinity) + -0.2474*salinity)*np.log(TK)
    poly3 = 0.053105*np.sqrt(salinity)*TK
    # calculate Kb
    Kb = np.exp(poly0+poly1+poly2+poly3)

    return Kb


def calculate_Kw(salinity: float, temperature: float) -> float:
    """
    Calculate Kw from salinity and temperature using Millero's 1995 pHt KW following Millero_Kw.vi
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        Kw in ??
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    # convert temperature to Kelvin
    TK = temperature + 273.15
    # build the giant polymonial
    poly0 = 148.9802 - 13847.26/TK
    poly1 = 23.6521*np.log(TK)
    poly2 = (118.67/TK + 1.0495*np.log(TK) - 5.977)*np.sqrt(salinity)
    poly3 = 0.01615*salinity
    # calculate Kb
    Kw = 10**((poly0-poly1+poly2-poly3)/np.log(10))

    return Kw


def calculate_Kb12(Kb: float, K1: float, K2: float) -> float:
    """
    Calculate Kb12 from Kb, K1, and K2 following RTCarbCalc_new.vi
    
    Args:
        Kb: 
        K1:
        K2:
        
    Returns:
        Kb12
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    return Kb/np.sqrt(K1*K2)

def calculate_Kw12(Kw: float, K1: float, K2: float) -> float:
    """
    Calculate Kw12 from Kw, K1, and K2 following RTCarbCalc_new.vi
    
    Args:
        Kw: 
        K1:
        K2:
        
    Returns:
        Kb12
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    return Kw/np.sqrt(K1*K2)



# ****** CALCULATE ANCILLARY CONCENTRATIONS *****
# density, calcium, boron

def seawater_density(salinity: float, temperature: float) -> float:
    """
    Calculate seawater density from salinity, temperature and pressure
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        temperature: Temperature in degrees Celsius
        
    Returns:
        Seawater density in kg/m³
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    pressure = 0 # assume always at the surface
    # calculate absolute salinity
    absolute_salinity = gsw.SA_from_SP(salinity, pressure, 0, 0) # 
    # calculate conservative temperature
    conservative_temperature = gsw.CT_from_t(absolute_salinity, temperature, pressure)
    # calculate density
    density = gsw.rho(absolute_salinity, conservative_temperature, pressure)

    return density


def calculate_calcium(salinity: float) -> float:
    """
    Calculate calcium concentration from salinity.
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        
    Returns:
        Calcium concentration in mmol/kg
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    return salinity * (10.5 - 1.0) / 35 + 1


def calculate_total_boron(salinity: float) -> float:
    """
    Calculate total boron concentration from salinity.
    
    Args:
        salinity: Salinity in PSU (Practical Salinity Units)
        
    Returns:
        Total boron concentration in umol/kg
        
    Note:
        This is a preliminary calculation. Reference needed for verification.
    """
    return salinity * 12.12






    

# def calculate_alkalinity(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
#     """
#     Calculate alkalinity from pCO2 and TCO2/DIC.
    
#     Args:
#         temperature: Temperature in degrees Celsius
#         salinity: Salinity in PSU
#         pco2: Partial pressure of CO2 in microatm
#         tco2: Total dissolved inorganic carbon in umol/L
        
#     Returns:
#         Alkalinity in umol/L
#     """
#     result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
#     return result['alkalinity']


# def calculate_ph(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
#     """
#     Calculate pH from pCO2 and TCO2/DIC.
    
#     Args:
#         temperature: Temperature in degrees Celsius
#         salinity: Salinity in PSU
#         pco2: Partial pressure of CO2 in microatm
#         tco2: Total dissolved inorganic carbon in umol/L
        
#     Returns:
#         pH (unitless)
#     """
#     result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
#     return result['ph']


# def calculate_omega_aragonite(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
#     """
#     Calculate aragonite saturation state from pCO2 and TCO2/DIC.
    
#     Args:
#         temperature: Temperature in degrees Celsius
#         salinity: Salinity in PSU
#         pco2: Partial pressure of CO2 in microatm
#         tco2: Total dissolved inorganic carbon in umol/L
        
#     Returns:
#         Aragonite saturation state (unitless)
#     """
#     result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
#     return result['omega_aragonite']


# def calculate_omega_calcite(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
#     """
#     Calculate calcite saturation state from pCO2 and TCO2/DIC.
    
#     Args:
#         temperature: Temperature in degrees Celsius
#         salinity: Salinity in PSU
#         pco2: Partial pressure of CO2 in microatm
#         tco2: Total dissolved inorganic carbon in umol/L
        
#     Returns:
#         Calcite saturation state (unitless)
#     """
#     result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
#     return result['omega_calcite']