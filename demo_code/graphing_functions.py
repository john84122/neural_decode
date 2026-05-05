'''
Augmented Code for Graphing.
'''

import h5py
import numpy as np
from temporaldata import ArrayDict, IrregularTimeSeries, RegularTimeSeries, Interval, Data

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
import numpy as np

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

def main():
    fl = "/Users/johannesbauer/Documents/Coding/neuro_comp_project/data/perich_miller_population_2018/t_20130819_center_out_reaching.h5"
    plot_data(fl)