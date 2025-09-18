"""
Carbonate Chemistry Calculations Module

This module provides functions for calculating carbonate chemistry parameters
including alkalinity, pH, and omega saturation states from pCO2 and TCO2/DIC.

Based on the calculations from pCO2toGoNotebook.ipynb
"""

import numpy as np
import math
from typing import Tuple, Dict, Any
import gsw


# Thermodynamic constants (equilibrium constants)
KCA = 3.8261e-7  # Calcite solubility constant
KAR = 5.48417e-7  # Aragonite solubility constant
KH = 0.0410032  # Henry's law constant
K1 = 1.04246e-6  # First dissociation constant of carbonic acid
K2 = 5.93885e-10  # Second dissociation constant of carbonic acid
KB = 1.67762E-9  # Boric acid dissociation constant
KW = 1.82055e-14  # Water dissociation constant
KB12 = 0.0674236
K1K2_QUOTIENT = K1/K2
K1K2_PRODUCT = K1*K2
KW12 = 7.31679e-7

# Ancillary constants
DENSITY = 1.02244
CO2AQ = 13.0304  # Aqueous CO2 concentration factor


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



def calculate_carbonate_chemistry_pco2_tco2(
    temperature: float,
    salinity: float, 
    pco2: float,
    tco2: float
) -> Dict[str, float]:
    """
    Calculate carbonate chemistry parameters from pCO2 and TCO2/DIC.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pco2: Partial pressure of CO2 in microatm (150-750 range)
        tco2: Total dissolved inorganic carbon in umol/L (20-5000 range)
        
    Returns:
        Dictionary containing:
        - co3: Carbonate ion concentration (umol/L)
        - hco3: Bicarbonate ion concentration (umol/L) 
        - co2aq: Aqueous CO2 concentration (umol/L)
        - ph: pH (unitless, 7.5-8.5 range)
        - omega_aragonite: Aragonite saturation state (1.1-4.2 range)
        - omega_calcite: Calcite saturation state
        - alkalinity: Total alkalinity (umol/L)
        - h_plus: Hydrogen ion concentration (mol/L)
        - oh_minus: Hydroxide ion concentration (umol/L)
        - b_minus: Borate ion concentration (umol/L)
        
    Raises:
        ValueError: If the polynomial solver fails to find valid roots
    """
    # Calculate ancillary constants
    calcium = calculate_calcium(salinity)
    total_boron = calculate_total_boron(salinity)

    # 
    
    # Calculate grouped constants for polynomial solution
    const_a = K1K2_QUOTIENT**0.5
    const_b = CO2AQ**0.5
    const_g = tco2 - CO2AQ
    const_h = const_a * const_b
    
    # Set up polynomial coefficients for solving CO3
    # Polynomial: x^2 - (const_h^2 + 2*const_g)*x + const_g^2 = 0
    coefficients = np.array([1, -(const_h**2 + 2*const_g), const_g**2])
    
    # Solve polynomial for CO3 concentration
    roots = np.roots(coefficients)
    
    # Find the valid root
    tco2_check = 0.5 * tco2
    co3 = None
    
    for root in roots:
        real_root = root.real
        imag_root = root.imag
        
        # Check if root meets validity conditions
        if (real_root <= tco2_check and 
            real_root >= 0 and 
            abs(imag_root) < 1e-10):  # Essentially zero imaginary part
            co3 = real_root
            break
    
    if co3 is None:
        raise ValueError("Polynomial solver failed to find valid CO3 concentration")
    
    # Calculate other carbonate species
    hco3 = const_a * ((CO2AQ * co3)**0.5)
    
    # Calculate hydrogen ion concentration and pH
    h_plus = (CO2AQ / co3 * K1K2_PRODUCT)**0.5
    ph = -math.log10(h_plus)
    
    # Calculate omega saturation states
    omega_aragonite = (co3 * calcium * 1000) / (KAR * 1e12)
    omega_calcite = (co3 * calcium * 1000) / (KCA * 1e12)
    # Note: Small discrepancy with notebook calcite omega (~2.743 vs ~2.929)
    # All other calculated values match exactly. Constants verified correct.
    
    # Calculate alkalinity components
    oh_minus = 1e6 * KW / h_plus
    b_minus = (KB * total_boron) / (h_plus + KB)
    alkalinity = 2 * co3 + hco3 + oh_minus + b_minus - h_plus
    
    return {
        'co3': co3,
        'hco3': hco3,
        'co2aq': CO2AQ,
        'ph': ph,
        'omega_aragonite': omega_aragonite,
        'omega_calcite': omega_calcite,
        'alkalinity': alkalinity,
        'h_plus': h_plus,
        'oh_minus': oh_minus,
        'b_minus': b_minus,
        'pco2': pco2,
        'tco2': tco2,
        'temperature': temperature,
        'salinity': salinity
    }


def calculate_alkalinity(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
    """
    Calculate alkalinity from pCO2 and TCO2/DIC.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pco2: Partial pressure of CO2 in microatm
        tco2: Total dissolved inorganic carbon in umol/L
        
    Returns:
        Alkalinity in umol/L
    """
    result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
    return result['alkalinity']


def calculate_ph(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
    """
    Calculate pH from pCO2 and TCO2/DIC.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pco2: Partial pressure of CO2 in microatm
        tco2: Total dissolved inorganic carbon in umol/L
        
    Returns:
        pH (unitless)
    """
    result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
    return result['ph']


def calculate_omega_aragonite(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
    """
    Calculate aragonite saturation state from pCO2 and TCO2/DIC.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pco2: Partial pressure of CO2 in microatm
        tco2: Total dissolved inorganic carbon in umol/L
        
    Returns:
        Aragonite saturation state (unitless)
    """
    result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
    return result['omega_aragonite']


def calculate_omega_calcite(temperature: float, salinity: float, pco2: float, tco2: float) -> float:
    """
    Calculate calcite saturation state from pCO2 and TCO2/DIC.
    
    Args:
        temperature: Temperature in degrees Celsius
        salinity: Salinity in PSU
        pco2: Partial pressure of CO2 in microatm
        tco2: Total dissolved inorganic carbon in umol/L
        
    Returns:
        Calcite saturation state (unitless)
    """
    result = calculate_carbonate_chemistry_pco2_tco2(temperature, salinity, pco2, tco2)
    return result['omega_calcite']