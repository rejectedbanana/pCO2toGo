import streamlit as st
import carbonate_chemistry as cc

# Set up the Streamlit app
st.title("pCO2 to GO Calculator")
st.markdown(
    """
    This application calculates the required reagent to achieve a target 
    aragonite saturation state (Ω aragonite) based on user-defined input parameters.
    """
)

# make a time
st.sidebar.markdown("## Input Parameters")

# Select calculation type
calculation_type = st.sidebar.selectbox(
    "Select measured parameters for the calculation:",
    ["pCO2 and Alkalinity", "pCO2 and tCO2/DIC"]
)

# Input the source water parameters
st.sidebar.markdown("### Input current water parameters")
# define the input pCO2
source_pCO2 = st.sidebar.number_input(
        label="pCO2 [µatm]:",
        value=400.0,
        format="%.1f",
        key="source_pCO2"
)
# define alkalinity or tCO2 depending on the calculation type
if calculation_type == "pCO2 and Alkalinity":
    source_alkalinity = st.sidebar.number_input(
        "Alkalinity [µmol/kg]:",
        value=2300.0,
        format="%.1f", 
        key="source_alkalinity"
    )
elif calculation_type == "pCO2 and tCO2/DIC":
    source_tCO2 = st.sidebar.number_input(
        "Total CO2 [µmol/kg]:",
        value=2000.0,
        format="%.1f", 
        key="source_tCO2"
    )
# define the water temperature and salinity
temperature = st.sidebar.number_input(
    "Water temperature [°C]:",
    value=25, 
    key="source_temperature"
)
salinity = st.sidebar.number_input(
    label="Salinity [PSU]:",
    value=30,
    key="source_salinity"
)


# Input the target parameters
st.sidebar.markdown("### Define parameter targets for the calculation")
target_OmegaAragonite = st.sidebar.number_input(
    "Ω aragonite :",
    value=400.0,
    format="%.1f", 
    key="target_OmegaAragonite"
)

# Input the reagent parameters
st.sidebar.markdown("### Define reagent parameters for the calculation")
carbonate_reagent_alkalinity = st.sidebar.number_input(
    "Carbonate Reagent Alkalinity [µmol/kg]:",
    value=330045.0,
    format="%.1f", 
    key="CRgt_Alk"
)
carbonate_reagent_TCO2 = st.sidebar.number_input(
    "Carbonate Reagent tCO2/DIC [µmol/kg]:",
    value=20037.0,
    format="%.1f", 
    key="CRgt_TCO2"
)
acid_reagent_HCL = st.sidebar.number_input(
    "Acid Reagent Concentration [µmol/kg]:",
    value=100000.0,
    format="%.1f", 
    key="Argt_HCL"
)
buffer_volume = st.sidebar.number_input(
    "Buffer Volume [L]:",
    value=16.0,
    format="%.1f", 
    key="Buffered_volume"
)


# Perform calculations based on selected type
if calculation_type == "pCO2 and Alkalinity":
    source = cc.calculate_carbonate_chemistry_pCO2_alkalinity(
        temperature,
        salinity,
        source_pCO2, 
        source_alkalinity
    )
    
elif calculation_type == "pCO2 and tCO2/DIC":
    source = cc.calculate_carbonate_chemistry_pCO2_tCO2(
        temperature,
        salinity,
        source_pCO2,
        source_tCO2
    )

    
# calculate carbonate chemistry for target omega aragonite
target = cc.calculate_carbonate_chemistry_pCO2_OmegaA(
    temperature,
    salinity,
    source_pCO2,
    target_OmegaAragonite
)

# calculate the required reagent volumes
reagents = cc.manipulator(
    Src_Alk=source['alkalinity'],
    Src_TCO2=source['tCO2'],
    Trgt_Alk=target['alkalinity'],
    Trgt_TCO2=target['tCO2'],
    CRgt_Alk=carbonate_reagent_alkalinity,
    CRgt_TCO2=carbonate_reagent_TCO2,
    Argt_HCL=acid_reagent_HCL,
    Buffered_volume=buffer_volume
)

st.subheader("Required Reagent Volumes")
st.write(f"To achieve a Ω aragonite of {target_OmegaAragonite}, the required volumes are:")
st.write(f"- Carbonate Reagent Volume: **{reagents['CrgtVol']:.2f} L**")
st.write(f"- Acid Reagent Volume: **{reagents['ArgtVol']:.2f} L**")

st.subheader("Current water summary")

st.subheader("Calculated target water summary")
