import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.measure import find_contours
from scipy.stats import zscore
from scipy.ndimage import label
import ipywidgets as widgets
from IPython.display import display, clear_output
import io


def manual_curation_interactive(A, C, max_proj, spatial_shape, use_robust_zscore=False, chunk_size=100):
    """
    Interactive neuron curation widget for Jupyter notebooks.

    Displays neurons in chunks (default 100) with checkboxes to mark neurons
    for exclusion. Navigate chunks with Prev/Next buttons, then click Done
    to collect results.

    Parameters
    ----------
    A : xarray.DataArray
        Spatial footprints, shape (n_units, d1, d2).
    C : xarray.DataArray
        Temporal traces, shape (n_units, T).
    max_proj : np.ndarray
        Max-projection image, shape (d1, d2).
    spatial_shape : tuple
        (d1, d2) dimensions of the FOV.
    use_robust_zscore : bool
        If True, use median-based z-scoring for traces.
    chunk_size : int
        Number of neurons per page (default 100). Set to -1 to display
        all neurons on a single page with no pagination.

    Returns
    -------
    result : dict
        Has key 'good_units' which is populated after clicking Done.
        Access via result['good_units'].
    """
    d1, d2 = spatial_shape
    n_units = A.shape[0]
    unit_ids = A.coords['unit_id'].values

    # --- Pre-compute all neuron images as PNG bytes for speed ---
    neuron_pngs = []
    print(f"Pre-rendering {n_units} neurons...")

    max_proj_display = np.clip(max_proj, np.percentile(max_proj, 1), np.percentile(max_proj, 99))

    for j in range(n_units):
        fig, (ax_img, ax_zoom, ax_trace) = plt.subplots(1, 3, figsize=(20, 5))

        # --- Max projection + contour ---
        ax_img.imshow(max_proj_display, cmap='viridis')
        footprint = A[j]
        fp_max = np.max(footprint)
        zoom_bounds = None
        clean_mask = None

        if fp_max > 0:
            mask = footprint > (0.3 * fp_max)
            labeled_mask, num_features = label(mask)
            if num_features > 0:
                sizes = [(labeled_mask == k).sum() for k in range(1, num_features + 1)]
                largest_label = np.argmax(sizes) + 1
                clean_mask = (labeled_mask == largest_label)
                contours = find_contours(clean_mask.astype(float), 0.5)
                if contours:
                    ax_img.plot(contours[0][:, 1], contours[0][:, 0], color='red')
                y, x = np.where(clean_mask)
                if len(x) > 0 and len(y) > 0:
                    zoom_bounds = (np.min(y), np.max(y), np.min(x), np.max(x))

        ax_img.set_xlim([0, d2])
        ax_img.set_ylim([d1, 0])
        ax_img.set_title(f"Neuron {unit_ids[j]}: Max Proj", fontsize=12)
        ax_img.axis('off')

        # --- Zoomed footprint ---
        if zoom_bounds:
            minr, maxr, minc, maxc = zoom_bounds
            pad = 25
            slr = slice(max(minr - pad, 0), min(maxr + pad, d1))
            slc = slice(max(minc - pad, 0), min(maxc + pad, d2))
            zoom_fp = footprint[slr, slc]
            zoom_fp_norm = zoom_fp / (np.max(zoom_fp) + 1e-6)
            ax_zoom.imshow(zoom_fp_norm, cmap='viridis', interpolation='nearest')
            ax_zoom.set_title(f"Neuron {unit_ids[j]}: Footprint", fontsize=12)
        else:
            ax_zoom.text(0.5, 0.5, "No footprint", ha='center', va='center')
        ax_zoom.axis('off')

        # --- Trace ---
        trace = C[j]
        if use_robust_zscore:
            baseline = np.median(trace.values)
            noise_std = np.median(np.abs(trace - baseline)) / 0.6745
            z_trace = (trace - baseline) / (noise_std + 1e-6)
        else:
            z_trace = zscore(trace.values)
        ax_trace.plot(z_trace, color='blue', linewidth=0.5)
        ax_trace.set_title(f"Neuron {unit_ids[j]}: Z-scored Trace", fontsize=12)
        ax_trace.set_xlabel("Frames", fontsize=10)

        fig.tight_layout()

        # Render to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        neuron_pngs.append(buf.read())

        if (j + 1) % 50 == 0 or j == n_units - 1:
            print(f"  Rendered {j + 1}/{n_units}")

    print("Rendering complete. Launching widget...\n")

    # --- Build interactive widget ---
    if chunk_size == -1:
        chunk_size = n_units
    n_chunks = int(np.ceil(n_units / chunk_size))
    current_chunk = [0]  # mutable container for closure

    # Custom CSS to enlarge checkboxes and disable output scroll
    checkbox_style = widgets.HTML(
        value="""<style>
        .big-checkbox .widget-checkbox input[type="checkbox"] {
            width: 24px; height: 24px; min-width: 24px; min-height: 24px;
        }
        .big-checkbox .widget-checkbox label { font-size: 15px; }
        .jupyter-widgets-output-area .output_scroll { height: auto !important; max-height: none !important; overflow: visible !important; box-shadow: none !important; }
        .full-height-output .output_scroll { height: auto !important; max-height: none !important; overflow: visible !important; box-shadow: none !important; }
        </style>"""
    )

    # One checkbox per neuron — checked = EXCLUDE
    checkboxes = []
    for j in range(n_units):
        cb = widgets.Checkbox(
            value=False,
            description=f"Exclude neuron {unit_ids[j]}",
            indent=False,
            layout=widgets.Layout(width='250px')
        )
        cb.add_class('big-checkbox')
        checkboxes.append(cb)

    output_area = widgets.Output(layout=widgets.Layout(overflow='visible', max_height='none'))
    status_label = widgets.HTML()
    result = {'good_units': None}

    def render_chunk(chunk_idx):
        output_area.clear_output(wait=True)
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_units)
        status_label.value = (
            f"<b>Showing neurons {start + 1}–{end} of {n_units} "
            f"(Page {chunk_idx + 1}/{n_chunks})</b>"
        )
        with output_area:
            for j in range(start, end):
                img_widget = widgets.Image(
                    value=neuron_pngs[j], format='png',
                    layout=widgets.Layout(width='1100px')
                )
                row = widgets.HBox(
                    [img_widget, checkboxes[j]],
                    layout=widgets.Layout(
                        align_items='center',
                        margin='0 0 4px 0',
                        border='solid 1px #ddd'
                    )
                )
                display(row)

    def on_prev(_):
        if current_chunk[0] > 0:
            current_chunk[0] -= 1
            render_chunk(current_chunk[0])

    def on_next(_):
        if current_chunk[0] < n_chunks - 1:
            current_chunk[0] += 1
            render_chunk(current_chunk[0])

    done_output = widgets.Output()

    def on_done(_):
        excluded = {unit_ids[j] for j, cb in enumerate(checkboxes) if cb.value}
        good = [uid for uid in unit_ids if uid not in excluded]
        result['good_units'] = good
        with done_output:
            clear_output()
            print(f"Done! Kept {len(good)}/{n_units} neurons. "
                  f"Excluded {len(excluded)}.")
            print(f"Access kept IDs via the returned dict: result['good_units']")

    btn_prev = widgets.Button(description='← Prev', layout=widgets.Layout(width='100px'))
    btn_next = widgets.Button(description='Next →', layout=widgets.Layout(width='100px'))
    btn_done = widgets.Button(
        description='✓ Done', button_style='success',
        layout=widgets.Layout(width='120px')
    )
    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_done.on_click(on_done)

    if n_chunks <= 1:
        nav_bar = widgets.HBox(
            [status_label, btn_done],
            layout=widgets.Layout(justify_content='center', margin='8px 0')
        )
    else:
        nav_bar = widgets.HBox(
            [btn_prev, status_label, btn_next, btn_done],
            layout=widgets.Layout(justify_content='center', margin='8px 0')
        )

    render_chunk(0)
    output_area.add_class('full-height-output')
    display(widgets.VBox(
        [checkbox_style, nav_bar, output_area, done_output],
        layout=widgets.Layout(overflow='visible', max_height='none')
    ))

    return result