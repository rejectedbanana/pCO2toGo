import streamlit as st
import carbonate_chemistry as cc

# Custom CSS to make sidebar wider
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 450px;
        max-width: 450px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set up the Streamlit app
st.title("pCO2 to GO Calculator")
st.markdown(
    """
    This application calculates the required reagent to achieve a target 
    state based on user-defined input parameters.
    """
)

# ***** SIDEBAR FOR THE CALCULATION PARAMETERS *****
# make a title
st.sidebar.markdown("# :blue[Starting Properties]")

# *** Starting Water *****
# Select calculation type
input_type = st.sidebar.selectbox(
    "Select your measurement pair:",
    ["pCO2 and Alkalinity", "pCO2 and pH [coming soon!]", "pCO2 and Total CO2/DIC"]
)

# Input current water properties
st.sidebar.markdown("### Input current water properties")
# define the input pCO2
source_pCO2 = st.sidebar.number_input(
        label="pCO\u2082 [µatm]",
        min_value = 100,
        max_value = 2000,
        step = 10,
        value = 400,
        key="source_pCO2",
)
# define alkalinity or tCO2 depending on the calculation type
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
        label = "Total CO\u2082 [µmol/kg]",
        min_value = 20,
        max_value = 5000,
        step = 10,
        value=2000,
        key="source_tCO2"
    )
# define the water temperature and salinity
temperature = st.sidebar.number_input(
    label = "Water temperature [°C]",
    min_value = -2,
    max_value = 38,
    step = 1,
    value=25, 
    key="source_temperature"
)
salinity = st.sidebar.number_input(
    label="Salinity [PSU]",
    min_value = 0,
    max_value = 40,
    step = 1,
    value=30,
    key="source_salinity"
)

## Perform calculations based on selected type
if input_type == "pCO2 and Alkalinity":
    source = cc.calculate_carbonate_chemistry_pCO2_alkalinity(
        temperature,
        salinity,
        source_pCO2,
        source_alkalinity
    )

elif input_type ==  "pCO2 and Total CO2/DIC":
    source = cc.calculate_carbonate_chemistry_pCO2_tCO2(
        temperature,
        salinity,
        source_pCO2,
        source_tCO2
    )


# Input the target parameters
# This is Omega Aragonite or pH
st.sidebar.markdown("# :green[Target State]")
# Select calculation type
target_type = st.sidebar.selectbox(
    "Select your target pair:",
    ["pCO2 and Ω aragonite", "pCO2 and pH [coming soon!]"]
)
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
        min_value = 0.0,
        max_value = 4.0,
        step = 0.1,
        value= 4.0,
        format ="%.1f",
        key="target_OmegaAragonite"
    )
    # do the calculation for target omega aragonite
    target = cc.calculate_carbonate_chemistry_pCO2_OmegaA(
    temperature,
    salinity,
    target_pCO2,
    target_OmegaAragonite
    )
    print(target)
elif input_type == "pCO2 and pH":
    target_pH = st.sidebar.number_input(
        label = "pH ",
        min_value = 6.0,
        max_value = 9.0,
        step = 0.1,
        value= 8.1,
        key="target_pH"
    )
    # do the calculation for target pH

# ****** SIDEBAR FOR THE REAGENT PARAMETERS *****
st.sidebar.markdown("# :orange[Reagent Properties]")
carbonate_reagent_alkalinity = st.sidebar.number_input(
    "Carbonate Reagent Alkalinity  [µmol/kg]",
    value=330045,
    key="CRgt_Alk"
)
carbonate_reagent_TCO2 = st.sidebar.number_input(
    "Carbonate Reagent tCO\u2082/DIC  [µmol/kg]",
    value=20037,
    key="CRgt_TCO2"
)
acid_reagent_HCL = st.sidebar.number_input(
    "Acid Reagent Concentration  [µmol/kg]",
    value=100000,
    key="Argt_HCL"
)
buffer_volume = st.sidebar.number_input(
    "Buffer Volume  [L]",
    value=16.0, 
    key="Buffered_volume"
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


st.subheader(":blue[Starting Water Property Summary]")
st.markdown("Calculated properties of the starting water:")

starting_summary =  {
    "pCO2": [f"{source['pCO2']:.0f} µatm"],
    "pH": [f"{source['pH']:.1f}"],
    "Alkalinity": [f"{source['alkalinity']:.0f} µmol/kg"],
    "Total CO2 (DIC)": [f"{source['tCO2']:.0f} µmol/kg"],
    "Ω aragonite": [f"{source['Omega_aragonite']:.0f}"]
}
st.dataframe(starting_summary, hide_index=True)
# st.table(starting_summary)

st.subheader(":green[Target State Summary]")
st.markdown("Calculated properties of the final water after reagent addition:")
target_summary =  {
    "pCO2": [f"{target['pCO2']:.0f} µatm"],
    "pH": [f"{target['pH']:.1f}"],
    "Alkalinity": [f"{target['alkalinity']:.0f} µmol/kg"],
    "Total CO2 (DIC)": [f"{target['tCO2']:.0f} µmol/kg"],
    "Ω aragonite": [f"{target['Omega_aragonite']:.0f}"]
}
st.dataframe(target_summary, hide_index=True)

st.subheader(":orange[Target Reagent Volumes]")
reagent_summary = {
    "Target Ω aragonite": [f"{target_OmegaAragonite:.1f}"],
    "Carbonate Reagent Volume": [f"{reagents['CrgtVol']:.2f} L"],
    "Acid Reagent Volume": [f"{reagents['ArgtVol']:.2f} L"]
}
st.dataframe(reagent_summary, hide_index=True)
