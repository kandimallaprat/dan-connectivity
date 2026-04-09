"""
Fetching and organizing the PAM DAN connections from the hemibrain.

"""
import pandas as pd
import numpy as np
import navis
import os
import connectome_analysis
from neuprint import NeuronCriteria as NC

# Plotting
import plotly.express as px
from plotly.subplots import make_subplots

#------------------------------------------------------------------------------
raw_folder = "Analysis_Outputs/Connectivity_Weights"
# All MB ROIs
mb_rois    = ["b'L(R)", 'gL(R)', 'gL(L)', "b'L(L)", 'bL(R)',
              'bL(L)', 'PED(R)', 'aL(R)', 'CA(R)', "a'L(R)"]
# We also just want to split into lobes vs non lobes
mb_lobes   = ["b'L(R)", 'gL(R)', 'gL(L)', "b'L(L)", 'bL(R)',
              'bL(L)', 'aL(R)', "a'L(R)"]
#------------------------------------------------------------------------------
token = os.environ["HEMIBRAIN_TOKEN"]
hemibrain_analysis = connectome_analysis.neuPrintConnect(token = token)

#------------------------------------------------------------------------------
# Create Neuron Criteria
pam_criteria = NC(instance = "PAM.*", regex = True)

# Fetch Inputs
inputs, _ = hemibrain_analysis.connectivity(presynaptic_criteria   = None,
                                            postsynaptic_criteria  = pam_criteria,
                                            min_weight             = 0,
                                            render                 = False,
                                            )

# Fetch Inputs
outputs, _ = hemibrain_analysis.connectivity(presynaptic_criteria   = pam_criteria,
                                             postsynaptic_criteria  = None,
                                             min_weight             = 0,
                                             render                 = False,
                                             )

# Save Raw
inputs.to_parquet(f"{raw_folder}/PAM_Inputs.parquet", index = False, engine = "pyarrow")
outputs.to_parquet(f"{raw_folder}/PAM_Outputs.parquet", index = False, engine = "pyarrow")

# Retain only the right side for future use
inputs = inputs.loc[inputs["instance_post"].str.contains("R")].reset_index(drop = True)
outputs = outputs.loc[outputs["instance_pre"].str.contains("R")].reset_index(drop = True)

#------------------------------------------------------------------------------
# Grouping by instance and keeping only the right side
instance_inputs  = inputs.groupby(["type_pre", "type_post", "roi"])["weight"].sum().reset_index()
instance_outputs = outputs.groupby(["type_pre", "type_post", "roi"])["weight"].sum().reset_index()

# Renaming the KCs to be only the three basic types
kc_mask_in = instance_inputs["type_pre"].str.startswith("KC", na = False)
instance_inputs["type_pre"] = np.where(
    kc_mask_in,
    instance_inputs["type_pre"].str.split("-", n = 1).str[0],   # split once, take [0]
    instance_inputs["type_pre"]
)

kc_mask_out = instance_outputs["type_post"].str.startswith("KC", na = False)
instance_outputs["type_post"] = np.where(
    kc_mask_out,
    instance_outputs["type_post"].str.split("-", n = 1).str[0],   # split once, take [0]
    instance_outputs["type_post"]
)

# Remove the NA
instance_inputs  = instance_inputs.loc[instance_inputs["type_pre"] != "NA"].reset_index(drop = True)
instance_outputs = instance_outputs.loc[instance_outputs["type_post"] != "NA"].reset_index(drop = True)

# MB ROIs mark
instance_inputs["MB_Status"]  = np.where(instance_inputs["roi"].isin(mb_rois), "MB", "Outside-MB")
instance_outputs["MB_Status"] = np.where(instance_outputs["roi"].isin(mb_rois), "MB", "Outside-MB")

# Add Lobes vs Non-MB into the final ROI
instance_inputs["Final_ROI"]  = np.where(instance_inputs["roi"].isin(mb_lobes), "MB-Lobes", instance_inputs["roi"])
instance_outputs["Final_ROI"] = np.where(instance_outputs["roi"].isin(mb_lobes), "MB-Lobes", instance_outputs["roi"])

# Save these
instance_inputs.to_parquet(f"{raw_folder}/PAM_Inputs_Type-Grouped.parquet", index = False, engine = "pyarrow")
instance_outputs.to_parquet(f"{raw_folder}/PAM_Outputs_Type-Grouped.parquet", index = False, engine = "pyarrow")

#------------------------------------------------------------------------------
# Use plotly to make a sunburst plot of each PAM neuron's inputs and outputs
pams = instance_outputs["type_pre"].unique()

# Create a new Leveling 
instance_inputs["Level"] = np.where(instance_inputs["MB_Status"] == "Outside-MB",
                                    instance_inputs["Final_ROI"],
                                    instance_inputs["type_pre"])
instance_outputs["Level"] = np.where(instance_outputs["MB_Status"] == "Outside-MB",
                                     instance_outputs["Final_ROI"],
                                     instance_outputs["type_post"])

# Only keeping connections > 5 synapses
thresh = 5
instance_inputs  = instance_inputs.loc[instance_inputs["weight"] > thresh].reset_index(drop = True)
instance_outputs = instance_outputs.loc[instance_outputs["weight"] > thresh].reset_index(drop = True)

# Cols and Rows
n_rows, n_cols = 4, 6
cmap = {"(?)": "#ffffff", "MB" : "#929afcff", "Outside-MB" : "#f06e57ff",
        "KCg": "#ddc7f7ff", "KCab": "#8a58deff", "KCa'b'": "#b786f3ff", "APL": "#5740c8ff", "DPM": "#3758ceff",
        "CRE(L)" : "#ec8251ff", "CRE(R)" : "#ec8251ff",
        "LAL(L)" : "#ed7a3eff", "LAL(R)" : "#ed7a3eff",
        "SMP(L)" : "#e26252ff", "SMP(R)" : "#e26252ff",
        "SIP(L)" : "#ec817dff", "SIP(R)" : "#ec817dff",
        "SLP(L)" : "#f2c0cbff", "SLP(R)" : "#f2c0cbff",
        "AOTU(L)" : "#929afcff", "AOTU(R)" : "#929afcff",
        }


# Create a main figure for all the PAM inputs and outputs in the same place
fig_inputs_main = make_subplots(rows = n_rows,
                                cols = n_cols,
                                specs = [[{"type": "sunburst"}] * n_cols] * n_rows)
fig_outputs_main = make_subplots(rows = n_rows,
                                cols = n_cols,
                                specs = [[{"type": "sunburst"}] * n_cols] * n_rows)
# Loop through each pam
for i, pam in enumerate(pams):
    # Assign Row and Col
    row = (i // n_cols) + 1
    col = (i % n_cols) + 1
    # Inputs 
    sub_inputs  = instance_inputs.loc[instance_inputs["type_post"] == pam]
    sub_outputs = instance_outputs.loc[instance_outputs["type_pre"] == pam]

    # Make Figures
    # Inputs
    fig_inputs = px.sunburst(sub_inputs, 
                             path = ["type_post", "MB_Status", "Level"],
                             values = "weight",
                             color  = "Level",
                             color_discrete_map = cmap,
                             title  = "Inputs")
    
    # Then explicitly update the marker colors for the middle ring:
    labels = fig_inputs.data[0].labels
    colors = list(fig_inputs.data[0].marker.colors)
    for j, label in enumerate(labels):
        if label in ["MB", "Outside-MB"]:
            colors[j] = cmap[label]
    fig_inputs.update_traces(
        marker = dict(colors = colors)
    )

    # Outputs
    fig_outputs = px.sunburst(sub_outputs, 
                             path = ["type_pre", "MB_Status", "Level"],
                             values = "weight",
                             color  = "Level",
                             color_discrete_map = cmap,
                             title  = "Outputs")
    # Then explicitly update the marker colors for the middle ring:
    labels = fig_outputs.data[0].labels
    colors = list(fig_outputs.data[0].marker.colors)
    for j, label in enumerate(labels):
        if label in ["MB", "Outside-MB"]:
            colors[j] = cmap[label]
    fig_outputs.update_traces(
        marker = dict(colors = colors)
    )

    # Make one figure
    fig = make_subplots(rows = 1,
                        cols = 2,
                        specs = [[{"type": "sunburst"}, {"type": "sunburst"}]],
                        subplot_titles = ["Inputs", "Outputs"])
    # Add Traces
    fig.add_trace(fig_inputs.data[0], row = 1, col = 1)
    fig.add_trace(fig_outputs.data[0], row = 1, col = 2)

    # Add to the Main Figures
    fig_inputs_main.add_trace(fig_inputs.data[0], row = row, col = col)
    fig_outputs_main.add_trace(fig_outputs.data[0], row = row, col = col)

    # Titles
    fig.update_layout(
        title_text = f"{pam} Inputs and Outputs",
        title_x = 0.5,
        height  = 800,
        width   = 1200,
        margin  = dict(t = 80, l = 20, r = 20, b = 20),
        showlegend = True,)
    
    # Save the Figure
    fig.write_html(f"Analysis_Outputs/PAM_Sunbursts/{pam}.html")
    fig.write_image(f"Analysis_Outputs/PAM_Sunbursts/{pam}.pdf", scale = 2)

# Main Titles
fig_inputs_main.update_layout(
    title_text = "PAM Inputs",
    title_x = 0.5,
    height  = 2400,
    width   = 3600,
    margin  = dict(t = 80, l = 20, r = 20, b = 20),
    showlegend = True,)

# Main Titles
fig_outputs_main.update_layout(
    title_text = "PAM Outputs",
    title_x = 0.5,
    height  = 2400,
    width   = 3600,
    margin  = dict(t = 80, l = 20, r = 20, b = 20),
    showlegend = True,)

# Save the Input Figure
fig_inputs_main.write_html(f"Analysis_Outputs/PAM_Inputs.html")
fig_inputs_main.write_image(f"Analysis_Outputs/PAM_Inputs.pdf", scale = 2)
fig_inputs_main.write_image(f"Analysis_Outputs/PAM_Inputs.png", scale = 2)
fig_inputs_main.write_image(f"Analysis_Outputs/PAM_Inputs.svg", scale = 2)  

# Save the Output Figure
fig_outputs_main.write_html(f"Analysis_Outputs/PAM_Outputs.html")
fig_outputs_main.write_image(f"Analysis_Outputs/PAM_Outputs.pdf", scale = 2)
fig_outputs_main.write_image(f"Analysis_Outputs/PAM_Outputs.png", scale = 2)
fig_outputs_main.write_image(f"Analysis_Outputs/PAM_Outputs.svg", scale = 2)


#------------------------------------------------------------------------------
# Status
print("हो गया दोस्तों!")