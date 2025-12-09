import streamlit as st
import carbonate_chemistry as cc

# Set up the Streamlit app
st.title("pCO\u2082 to Go Calculator")
st.markdown(
    """
    This application calculates the required reagent to achieve a target 
    state based on the initial state of the water in the tank.
    """
)

# ***** SIDEBAR FOR THE CALCULATION PARAMETERS *****
st.sidebar.image("pco2-to-go-gif.gif", use_column_width=True)
# make a title
st.sidebar.markdown("# :blue[Starting Properties]")

# *** SOURCE/STARTING WATER *****
# Select calculation type
st.sidebar.markdown("### Select the measurement pair for the starting water")
# define the calculation type
input_type = st.sidebar.selectbox(
    "Select your measurement pair:",
    ["pCO2 and Alkalinity", "pCO2 and pH", "pCO2 and Total CO2/DIC"], 
    label_visibility="collapsed"
)
# Input current water properties
st.sidebar.markdown("### Input starting water properties")
# define the input pCO2
source_pCO2 = st.sidebar.number_input(
        label="pCO\u2082 [µatm]",
        min_value = 100,
        max_value = 2000,
        step = 10,
        value = 400,
        key="source_pCO2",
)
# define alkalinity, pH, or tCO2 depending on the calculation type
if input_type == "pCO2 and Alkalinity":
    source_alkalinity = st.sidebar.number_input(
        label = "Alkalinity [µmol/kg]",
        min_value = 400,
        max_value = 4000,
        step = 10,
        value=2300,
        key="source_alkalinity"
    )
elif input_type == "pCO2 and Total CO2/DIC":
    source_tCO2 = st.sidebar.number_input(
        label = "Total CO\u2082/DIC [µmol/kg]",
        min_value = 20,
        max_value = 5000,
        step = 10,
        value=2000,
        key="source_tCO2"
    )
elif input_type == "pCO2 and pH":
    source_pH = st.sidebar.number_input(
        label = "pH ",
        min_value = 6.0,
        max_value = 9.0,
        step = 0.1,
        value= 7.0,
        key="source_pH"
    )
# define the water temperature and salinity
source_temperature = st.sidebar.number_input(
    label = "Water temperature [°C]",
    format="%.1f",
    min_value = -2.0,
    max_value = 38.0,
    step = 1.0,
    value=25.0, 
    key="source_temperature"
)
salinity = st.sidebar.number_input(
    label="Salinity [PSU]",
    format="%.1f",
    min_value = 0.0,
    max_value = 40.0,
    step = 1.0,
    value=30.0,
    key="source_salinity"
)

## Perform calculations based on selected type
if input_type == "pCO2 and Alkalinity":
    source = cc.calculate_carbonate_chemistry_pCO2_alkalinity(
        source_temperature,
        salinity,
        source_pCO2,
        source_alkalinity
    )
elif input_type ==  "pCO2 and Total CO2/DIC":
    source = cc.calculate_carbonate_chemistry_pCO2_tCO2(
        source_temperature,
        salinity,
        source_pCO2,
        source_tCO2
    )
elif input_type == "pCO2 and pH":
    source = cc.calculate_carbonate_chemistry_pCO2_pH(
        source_temperature,
        salinity,
        source_pCO2,
        source_pH
    )

# *** TARGET WATER *****
# Input the target parameters
# This is Omega Aragonite or pH
st.sidebar.markdown("# :orange[Target Properties]")
# Select calculation type
st.sidebar.markdown("### Select the measurement pair for the target water")
# Select calculation type
target_type = st.sidebar.selectbox(
    "Select your target pair:",
    ["pCO2 and Ω aragonite", "pCO2 and pH"], 
    label_visibility="collapsed"
)

# Select calculation type
st.sidebar.markdown("### Input target water properties")
# define the target pCO2
target_pCO2 = st.sidebar.number_input(
        label="pCO\u2082 [µatm]",
        min_value = 100,
        max_value = 2000,
        step = 10,
        value = 400,
        key="target_pCO2",
)
# define target omega aragonite or pH depending on the calculation type
if target_type == "pCO2 and Ω aragonite":
    target_OmegaAragonite = st.sidebar.number_input(
        label = "Ω aragonite ",
        min_value = 0,
        max_value = 4,
        step = 1, 
        value= 4,
        key="target_OmegaAragonite"
    )
elif target_type == "pCO2 and pH":
    target_pH = st.sidebar.number_input(
        label = "pH ",
        min_value = 6.0,
        max_value = 9.0,
        step = 0.1,
        value= 8.1,
        key="target_pH"
    )
# define the water temperature and salinity
target_temperature = st.sidebar.number_input(
    label = "Water temperature [°C]",
    format="%.1f",
    min_value = -2.0,
    max_value = 38.0,
    step = 1.0,
    value=source['temperature'], 
    key="target_temperature"
)
if target_type == "pCO2 and Ω aragonite":
    # do the calculation for target omega aragonite
    target = cc.calculate_carbonate_chemistry_pCO2_OmegaA(
    target_temperature,
    salinity,
    target_pCO2,
    target_OmegaAragonite
    )
    print(target)
elif target_type == "pCO2 and pH":
    # do the calculation for target pH
    target = cc.calculate_carbonate_chemistry_pCO2_pH(
        target_temperature,
        salinity,
        target_pCO2,
        target_pH
)
    
    

# ****** SIDEBAR FOR THE REAGENT PARAMETERS *****
# select the volume units
st.sidebar.markdown("### Reagent Properties and Target Volumes")

# select volume units
volume_units = st.sidebar.selectbox(
    "Select volume units:",
    ["Liters", "Gallons", "Cubic Meters"]
)

if volume_units == "Liters":
    vol_units_label = "L"
elif volume_units == "Gallons":
    vol_units_label = "gal"
elif volume_units == "Cubic Meters":
    vol_units_label = "m³"

buffer_volume_label = "Target System/Tank Volume  [" + vol_units_label + "]"

buffer_volume = st.sidebar.number_input(
    buffer_volume_label,
    value=16.0, 
    key="Buffered_volume"
)

if volume_units == "Liters":
    st.sidebar.info("Note: All volume outputs will be in Liters.")
elif volume_units == "Gallons":
    st.sidebar.info("Note: All volume outputs will be in Gallons.")
elif volume_units == "Cubic Meters":
    st.sidebar.info("Note: All volume outputs will be in Cubic Meters.")

# select the reagent properties
carbonate_reagent_alkalinity = st.sidebar.number_input(
    "Carbonate Reagent Alkalinity  [µmol/kg]",
    value=330000,
    format="%.2e",
    step = 1000,
    key="CRgt_Alk"
)
carbonate_reagent_TCO2 = st.sidebar.number_input(
    "Carbonate Reagent tCO\u2082/DIC  [µmol/kg]",
    value=200000,
    format="%.2e",
    step = 1000,
    key="CRgt_TCO2"
)
acid_reagent_HCL = st.sidebar.number_input(
    "Acid Reagent Concentration  [µmol/kg]",
    value=100000,
    format="%.2e",
    step = 1000,
    key="Argt_HCL"
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

# ***** DISPLAY RESULTS *****

# Display the starting water summary
st.subheader(":blue[Starting Water Property Summary]")
st.markdown("Properties of the starting water before adding reagents:")

starting_summary =  {
    "pCO2": [f"{source['pCO2']:.0f} µatm"],
    "pH": [f"{source['pH']:.1f}"],
    "Alkalinity": [f"{source['alkalinity']:.0f} µmol/kg"],
    "Total CO2 (DIC)": [f"{source['tCO2']:.0f} µmol/kg"],
    "Ω aragonite": [f"{source['Omega_aragonite']:.0f}"]
}
st.dataframe(starting_summary, hide_index=True)

# Display the target water summary
st.subheader(":orange[Target Water Property Summary]")
st.markdown("Properties of the target water after adding reagents:")
target_summary =  {
    "pCO2": [f"{target['pCO2']:.0f} µatm"],
    "pH": [f"{target['pH']:.1f}"],
    "Alkalinity": [f"{target['alkalinity']:.0f} µmol/kg"],
    "Total CO2 (DIC)": [f"{target['tCO2']:.0f} µmol/kg"],
    "Ω aragonite": [f"{target['Omega_aragonite']:.0f}"]
}
st.dataframe(target_summary, hide_index=True)

# Display reagent volumes
st.markdown("Reagent volumes needed to achieve target water property:")
reagent_summary = {
    "Carbonate Reagent Volume": [f"{reagents['CrgtVol']:.2f} "+vol_units_label],
    "Acid Reagent Volume": [f"{reagents['ArgtVol']:.2f} "+vol_units_label]
}
st.dataframe(reagent_summary, hide_index=True)

st.write("")
st.write("")

## ****** ADD SOME ADDITIONAL INFO ABOUT THE PROJECT HERE ******

with st.expander("About this project"):
    st.write('''
        pCO2 to Go is a sensor system that fits in the palm of a hand and provides instant readouts of the amount of dissolved carbon dioxide in seawater (pCO2).
    ''')
    st.image("pCO2toGo.jpg")
