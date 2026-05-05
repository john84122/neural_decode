
import pandas as pd
import streamlit as st

import h5py
import numpy as np
from temporaldata import ArrayDict, IrregularTimeSeries, RegularTimeSeries, Interval, Data

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
import numpy as np

from streamlit_bokeh import streamlit_bokeh
from bokeh.models import Button
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column


def plot_spikes(spikes, add_range_tool=False, x_range=None, width=800, height=400):
    """
    Plots an IrregularTimeSeries object defined by spikes.timestamps and spikes.unit_index.

    Parameters:
    spikes: An object containing 'timestamps' and 'unit_index' attributes.
    """
    if x_range is None:
        if add_range_tool:
            x_range = (spikes.timestamps[0] * 1e3, spikes.timestamps[0] * 1e3 + 20_000)
        else:
            x_range = (spikes.timestamps[0] * 1e3, spikes.timestamps[-1] * 1e3)

    # Create a figure
    p = figure(x_axis_label='Time', y_axis_label='Unit Index', width=width, height=height, x_axis_type="datetime", x_range=x_range, title="Spikes")

    # Prepare data for plotting
    x_values = spikes.timestamps * 1e3
    y_values = spikes.unit_index

    # Create a ColumnDataSource
    source = ColumnDataSource(data=dict(x=x_values, y=y_values))

    # Add scatter points to the plot
    p.scatter('x', 'y', source=source, size=5, color="navy", alpha=0.5, marker="dash", angle=np.pi/2)

    if add_range_tool:
        select = figure(height=height//5, width=width, tools="",
                        toolbar_location=None, background_fill_color="#efefef", x_axis_type="datetime",
                        title="Average Population Activity")
        select.xaxis.visible = False
        # select.yaxis.visible = False

        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        spike_times_int = spikes.timestamps.astype(int)
        population_activity =np.bincount(spike_times_int-spike_times_int[0])
        source = ColumnDataSource(data=dict(x=(np.arange(len(population_activity)) + spike_times_int[0])* 1e3, y=population_activity))

        select.line('x', 'y', source=source)
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        p = column(select, p)

    return p

def plot_time_series(data, field, index=None, x_range=None, add_range_tool=False, y_axis_label=None, width=800, height=200, include_control_panel=False, shade_area_list=[]):
    # Create a figure
    if x_range is None:
        if add_range_tool:
            x_range = (data.timestamps[0] * 1e3, data.timestamps[0] * 1e3 + 20_000)
        else:
            x_range = (data.timestamps[0] * 1e3, data.timestamps[-1] * 1e3)

    if y_axis_label is None:
        y_axis_label = field

    p = figure(x_axis_label='Time', y_axis_label=y_axis_label, width=width, height=height, x_axis_type="datetime", x_range=x_range)


    domain_start = data.domain.start * 1e3
    domain_end = data.domain.end * 1e3
    # Prepare data for plotting
    x_values = data.timestamps * 1e3
    y_values = getattr(data, field)

    x_values = np.concatenate([x_values, domain_start, domain_end])
    y_values = np.concatenate([y_values, np.nan * np.ones((len(data.domain), * y_values.shape[1:])), np.nan * np.ones((len(data.domain), *y_values.shape[1:]))])

    # sort x_values and reorder y_values
    sort_indices = np.argsort(x_values)
    x_values = x_values[sort_indices]
    y_values = y_values[sort_indices]

    if y_values.ndim == 1:
        # Add a line to the plot
        source = ColumnDataSource(data=dict(x=x_values, y=y_values))
    elif y_values.ndim == 2:
        assert index is not None, "Index must be provided for 2D data"
        source = ColumnDataSource(data=dict(x=x_values, y=y_values[:, index]))
    else:
        raise ValueError(f"Field {field} has {y_values.ndim} dimensions, expected 1 or 2")

    p.line(x='x', y='y', source=source, line_width=2, color="green")
    x_range = p.x_range

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']
    for i, shade_area in enumerate(shade_area_list):
        start = shade_area.start * 1e3
        end = shade_area.end * 1e3
        rect_x = (start + end)/2
        rect_width = end - start
        rect_y = np.ones_like(start) * (y_values.min() + y_values.max())/2
        rect_height = np.ones_like(start) * (y_values.max() - y_values.min())
        p.rect(x=rect_x, y=rect_y, width=rect_width, height=rect_height, fill_color=colors[i], fill_alpha=0.4)
        # legend_items.append((f"Shade Area {i+1}", [p.renderers[-1]]))  # Add the rectangle to the legend items

    # p.legend.items = legend_items
    # p.legend.location = "top_left"  # Set the legend location
    # p.legend.click_policy = "hide"  # Allow clicking to hide the legend items

    if add_range_tool:
        select = figure(height=height//5, width=width, y_range=p.y_range, tools="",
                        toolbar_location=None, background_fill_color="#efefef", x_axis_type="datetime",)
        select.xaxis.visible = False
        select.yaxis.visible = False

        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        select.line('x', 'y', source=source)
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        p = column(select, p)

        if include_control_panel:
            # Add a button to control the range tool
            play_button = Button(label="Play", button_type="success")
            pause_button = Button(label="Pause", button_type="warning")
            speed_buttons = [Button(label=f"{2**i}x", button_type="primary") for i in range(5)]

            # Create a shared ColumnDataSource to store the interval ID
            shared_data = ColumnDataSource(data=dict(interval_id=[None], step_size=[1000]))

            # Update the CustomJS to use the shared data source
            play_button.js_on_click(CustomJS(args=dict(range_tool=range_tool, x_values=x_values, shared_data=shared_data), code="""
                // Clear previous interval if it exists
                if (shared_data.data.interval_id[0]) {
                    clearInterval(shared_data.data.interval_id[0]);
                }

                // Create new interval and store it in the shared data
                let new_interval = setInterval(() => {
                    if (range_tool.x_range.end < x_values[x_values.length - 1]) {
                        range_tool.x_range.start += shared_data.data.step_size[0] / 10;
                        range_tool.x_range.end += shared_data.data.step_size[0] / 10;
                    } else {
                        clearInterval(shared_data.data.interval_id[0]);
                        shared_data.data.interval_id[0] = null;
                        shared_data.change.emit();
                    }
                }, 100);

                shared_data.data.interval_id[0] = new_interval;
                shared_data.change.emit();
            """));

            pause_button.js_on_click(CustomJS(args=dict(shared_data=shared_data), code="""
                if (shared_data.data.interval_id[0]) {
                    clearInterval(shared_data.data.interval_id[0]);
                    shared_data.data.interval_id[0] = null;
                    shared_data.change.emit();
                }
            """));

            for button in speed_buttons:
                button.js_on_click(CustomJS(args=dict(button=button, shared_data=shared_data), code="""
                    shared_data.data.step_size[0] = 1000 * parseInt(button.label.replace('x', ''));
                    shared_data.change.emit();
                """));

            # Add the buttons to the layout
            button_layout = row(play_button, pause_button, *speed_buttons)
            return p, x_range, button_layout

    return p


def plot_intervals(*interval, x_range=None, title=None, width=800, height=200):
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black"]
    # Create a figure
    if x_range is None:
      p = figure(title=title, x_axis_label='Time', y_axis_label='Intervals', y_range=(-len(interval), 1), width=width, height=height, x_axis_type="datetime",)
    else:
      p = figure(title=title, x_axis_label='Time', x_range=x_range, y_axis_label='Intervals', y_range=(-len(interval), 1), width=width, height=height, x_axis_type="datetime",)

    p.yaxis.visible = False

    for i in range(len(interval)):
        # Prepare data for plotting
        centers = (interval[i].start + interval[i].end) / 2.  * 1e3
        durations = (interval[i].end - interval[i].start)  * 1e3
        y_values = np.zeros_like(centers) - i  # y-values for the intervals

        # Create a ColumnDataSource
        source = ColumnDataSource(data=dict(x=centers, width=durations, y=y_values))

        # Add rectangles to the plot
        p.rect(x='x', y='y', width='width', height=0.8, source=source, fill_color=colors[i % 10], line_color="black", alpha=0.5)

    return p

def make_plot(data, add_play_controls=False):
    include_control_panel = add_play_controls
    if include_control_panel:
      p_cursor_pos_x, x_range, button_layout = plot_time_series(data.cursor, 'vel', y_axis_label='cursor position x', index=0, add_range_tool=include_control_panel, include_control_panel=include_control_panel, shade_area_list=[data.domain])
    else:
      p_cursor_pos_x = plot_time_series(data.cursor, 'vel', y_axis_label='cursor position x', index=0, add_range_tool=include_control_panel, include_control_panel=include_control_panel, shade_area_list=[data.domain])
      x_range=p_cursor_pos_x.x_range

    p_cursor_pos_y = plot_time_series(data.cursor, 'vel', y_axis_label='cursor position y', index=1, add_range_tool=include_control_panel, x_range=x_range, shade_area_list=[data.reach_intervals])
    p_cursor_vel_x = plot_time_series(data.cursor, 'vel', y_axis_label='cursor velocity x', index=0, add_range_tool=include_control_panel, x_range=x_range, shade_area_list=[data.reach_intervals])
    p_cursor_vel_y = plot_time_series(data.cursor, 'vel', y_axis_label='cursor velocity y', index=1, add_range_tool=include_control_panel, x_range=x_range, shade_area_list=[data.reach_intervals])
    p_spikes = plot_spikes(data.spikes, add_range_tool=include_control_panel, x_range=x_range, height=480)
    p_reach_intervals = plot_intervals(data.reach_intervals, x_range=x_range, height=120, title="Reach intervals")

    if include_control_panel:
      return column(button_layout, row(p_spikes, column(p_reach_intervals, p_cursor_vel_x, p_cursor_vel_y)))
    else:
      return row(p_spikes, column(p_reach_intervals, p_cursor_vel_x, p_cursor_vel_y))

def load_h5_full(filepath):
    def recursively_load(obj):
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        elif isinstance(obj, h5py.Group):
            return {key: recursively_load(obj[key]) for key in obj.keys()}
    
    with h5py.File(filepath, 'r') as f:
        data = recursively_load(f)
    return data

def plot_data(fl):

  data = load_h5_full(fl)

  spike_timestamps = np.array([])
  spike_unit_index = np.array([])


  # Create Spike Train
  #for i in range(len(data["spikes"]["train_mask"])):
  #    spike_train = data["spikes"]["train_mask"][i]
  #    spike_timestamps = np.concatenate([spike_timestamps, spike_train])
  #    spike_unit_index = np.concatenate([spike_unit_index, np.full_like(spike_train, fill_value=i)])

  spike_train = data["spikes"]["train_mask"]
  spike_timestamps = data["spikes"]["timestamps"]
  spike_unit_index = data["spikes"]["unit_index"]

  spikes = IrregularTimeSeries(
      timestamps=spike_timestamps,
      unit_index=spike_unit_index,
      domain="auto",
  )
  spikes.sort()

  # Wi did this.
  timestamps = data["cursor"]["timestamps"]
  cursor_pos = data["cursor"]["pos"]
  cursor_vel = data["cursor"]["vel"]
  cursor_acc = data["cursor"]["acc"]

  sampling_rate = 100 # Hz
  assert np.allclose(np.diff(timestamps), 1/sampling_rate)

  cursor = RegularTimeSeries(
      pos=cursor_pos,
      vel=cursor_vel,
      acc=cursor_acc,
      sampling_rate=sampling_rate,
      domain_start=timestamps[0],
      domain="auto",
  )

  reach_intervals = Interval(
      start=data["trials"]["start"],
      end=data["trials"]["end"],
      result=data["trials"]["result"],
      target_id=data["trials"]["target_id"],
  )

  data = Data(
    spikes=spikes,
    cursor=cursor,
    reach_intervals=reach_intervals,
    domain="auto",
  )

  p = make_plot(data, add_play_controls=True)
  #show(p)

  return p

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.write("# Background")

    st.write("In this section, we will discuss two main concepts that motivate and center our task")

    st.write("## Neural Decoding")

    st.write("- The first neuro science connection of our experiments is associated to neural decoding.")

    st.write("### Motivation and Definition Neural Decoding")

    st.write("As we have discussed in class, computation in the model occurs between the complex interaction of neurons in our brain. Neurons have a very simple structure:")
    st.write("  - **Dendrites:** These are the input connections that takes in spikes from neurons.")
    st.write("  - **Cell Body:** Takes in inputs and integrates it into a internal state.")
    st.write("  - **Axon:** when internal state reaches a critical point, a output signal travels through this component.")
    st.write("  - **Synapses:** This is the area that takes in the signal transmitted from other neurons.")
    st.write("*These definitions are adapted from the intro to neural networks given in the canvas modules.*")
    st.write()
    st.write("There are my different motivations for why we want to understand how complex interactions lead towards these behaviors. There are many different ways at capturing the connections between the macroscopic and microscopic cognition in the brain. In addition, machine learning approaches have been very useful of illuminating the connections between brain activity and high level cognition.")
    st.write("Some examples include:")

    c1, c2, c3 = st.columns(3, vertical_alignment="center")

    with c1:
        st.write("#### Analysis of Brain Syndromes")
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/split_brain.png", caption="Illustration of Split Brain Experiment (Reisbergch et al.)")
        st.write("We can people with brain damage to understand how large regions of the brain connect to higher level cognition.")


    with c2:
        st.write("#### Evolutionary Methods")
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/cma_es_image.png", caption="Illustration of Evolutionary Based Methods from Wang et al.")
        st.write("We can use machine learning methods to find optimal input stimuli to maximize activity of the brain. The hope is that we can interpret the activations.")

    with c3:
        st.write("#### Biological Models")
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/flywire.png", caption="Example of FlyWire Model (Dorkenwald et al. and Schlegel et al.)")
        st.write("People can build massive diagrams and computational models of the brain like the Fruit Fly")
    
    st.write("While all these methods come with strengths and weaknesses, the most direct appraoch of understanding the mapping between activity and action is through Neural Decoding.")

    st.write("### The Neural Decoding Task: Rapid Learning")

    st.write("Neural decoding is the basic concept of building models which take in inputs of spiking activity and predicting the action. Generally, one sticks sensors into the brain of a animal (like Macaque Monkey) and records neuron activity as well as the actions of the monkey on a task it was trained on.")
    st.write("The key idea idea of Perich et al is that when people learn something, it is often thought that plasticity of neurons lead toward the long term memory of that skill. However, some skills require one to learn rapidly, especially related to one shot learning which use motor skills (movement). Perich et al. believes that rather than plasticity inside the PMd and M1 cortex connected lead to rapid learning, it is rather the activity of the neurons themselves.")
    st.write()
    st.write("To measure this, they look at activity of two areas of the brain. First, they study the activations of the M1 and PMd areas of the brain of Macaque monkeys. They surjically placed multi-electrode arrays and collected 96 channels of neural activity. 137 to 257 PMd Neurals and 55 to 93M1 neurons were collected for analysis. The task that they were asked to do was that they were to move a 2-D planar cuersor to control a cursor on a computer screen. The goal was a center -out reaching task.")
    st.write()
    st.write("Their findings indicate that there were no indication that synaptic connectivity was the cause of the rapid learning. But also that null activity of neurons could be modified in order to change M1 activity which possibly explains the rapid learning.")

    st.write("### Our Dataset")

    st.write("The dataset associated to the perich miller et al. task consists of spiking activity which can be visualied below.")
    st.write("- The bottom left graph shows spiking activity of the 55 of 96 channels from the experiment. The x-axis is time and y axis indicates the sensor.")
    st.write("- The bottom most right graphs are the x and y velocity vectors of the monkey moving the cursors. The goal is to predict these movements.")

    data_path = "/Users/johannesbauer/Documents/Coding/neuro_comp_project/data/perich_miller_population_2018/t_20130819_center_out_reaching.h5"
    new_graph = plot_data(data_path)

    streamlit_bokeh(new_graph, use_container_width=True, theme = "light_minimal")
    st.write("The graph comes from adapted code in a Cosyne Tutorial on Transformers: https://cosyne-tutorial-2025.github.io")

    st.write("- As one can observe, it is really difficult to understand how these activities can be used to predict actions")
    st.write("- For example, there is a lot of activition in velocity at 20 seconds of this recording. But, there is little velocity movement at 13 minutes. Despite this, the spiking activity we observe looks almost complete same in. the left most graph.")

    st.divider()

    st.write("## Hopfield Networks, Transformers, and Equivalences between Models")

    st.write("Another aspect of our experiment is focused on the connections between transformers and hopfield networks.")
    st.write("Transformers and Hopfield Networks are two models that are defined below.")


    c1_1, c1_2 = st.columns(2, vertical_alignment="bottom")

    with c1_1:
        st.write("#### Hopfield Networks")
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/real_images/hopfield_network.png", caption="Visual Image of a Hopfield Layer from Thakur et al.")
        st.write("A energy based method in which neurons are all interconnected to each other. The output of the model is proportional to the similarity between weights of the neurons and input.")

    with c1_2:
        st.write("#### Transformers")
        st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/attention_mechanism.png", caption="Example of Attention Mechanism from Vaswani et al.")
        st.write("This model uses a machine learning notion of attention in order to learn and highlight components of an input that are important for a task.")

    
    st.write("One of the biggest findings of a paper named \"Hopfield Networks is All You Need\" is that there is a clear theoretical equivalents between these two seamingly different architectures. They demonstrate the similarity between these models using training on tabular datasets.")
    st.image("/Users/johannesbauer/Documents/Coding/neuro_comp_project/results/misc/hopfield_transformer_equiv_img.png", caption="Equivalence between New Hopfield Networks and Transformers from Ramsauer et al.")

    st.write("One of the problems with their paper is the limited quantitative evidence of the equivalence. Small datasets are fine for a simple experiment, but there is a want to show equivanece of these models in a more rigorous way.")

    st.write("## Our Main Problems")
    st.write("Given that we have shown the difficulty of neural decoding and the limitations of the Ramsauer et al. paper, we provide the two questions taht we want to answer in this paper:")

    st.write("  - How equivalent are Transformer Models to Hopfield Networks in their performance on Neural Decoding Tasks?")
    st.write("  - Are transformers and Hopfield Networks good for Neural Decoding?")

    st.write("Our Experiments are discussed in the experiment_and_design section.")
