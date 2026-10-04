import time

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

from core.img_utils import (
    load_image,
    matrix_to_image
)

from core.filters import apply_filter
from core.kernels import get_kernel

from analysis.metrics import (
    calculate_mse,
    calculate_psnr
)

from analysis.benchmark import (
    benchmark_all_filters
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Matrix Image Processing",
    page_icon=None,
    layout="wide"
)


# =========================================================
# SESSION STATE FOR AUTO SCROLL
# =========================================================

if "scroll_counter" not in st.session_state:
    st.session_state.scroll_counter = 0


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .main .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1,
    h2,
    h3 {
        color: #1e3a5f;
    }

    .main-title {
        color: #1e3a5f;
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 17px;
        margin-bottom: 55px;
    }

    .section-title {
        color: #1e3a5f;
        font-size: 24px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    .center-section-title {
        color: #1e3a5f;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 12px;
    }

    .info-card {
        background-color: #eef6ff;
        border-left: 5px solid #60a5fa;
        padding: 15px 18px;
        border-radius: 8px;
        margin: 15px 0;
        color: #334155;
    }

    /* -----------------------------------------------------
       CENTERED INPUT AREA
       ----------------------------------------------------- */

    .center-input {
        max-width: 340px;
        margin: 0 auto;
    }

    /* -----------------------------------------------------
       APPLY FILTER BUTTON
       ----------------------------------------------------- */

    div.stButton > button {
        display: block;
        width: 340px;
        height: 48px;
        margin: 20px auto 0 auto;

        background-color: #ff4b4b;
        color: white;

        border: none;
        border-radius: 8px;

        font-size: 16px;
        font-weight: 500;
    }

    div.stButton > button:hover {
        background-color: #ff3333;
        color: white;
        border: none;
    }

    /* -----------------------------------------------------
       IMAGE INFORMATION
       ----------------------------------------------------- */

    div[data-testid="stMetric"] {
        background-color: #eef6ff;
        border: 1px solid #dbeafe;
        padding: 15px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #1e3a5f;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🖼️ Matrix-Based Image Processing
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Explore how matrix operations and convolution create image filters.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILTER DESCRIPTIONS
# =========================================================

FILTER_DESCRIPTIONS = {

    "Box Blur":
        "Each value in the kernel is equal. "
        "The surrounding pixels are averaged to create a smooth and blurred image.",

    "Gaussian Blur":
        "The Gaussian kernel gives more importance to pixels near the centre. "
        "Convolution produces a smoother and more natural blur.",

    "Sharpen":
        "The kernel emphasizes differences between a pixel and its neighbours. "
        "This increases edges and makes details appear sharper.",

    "Edge Detection":
        "The kernel detects large intensity differences between neighbouring pixels. "
        "These differences are highlighted as edges.",

    "Sobel X":
        "Detects intensity changes in the horizontal direction "
        "and highlights vertical edges.",

    "Sobel Y":
        "Detects intensity changes in the vertical direction "
        "and highlights horizontal edges.",

    "Emboss":
        "The kernel emphasizes directional intensity changes "
        "to create a raised or shadow-like appearance."
}


# =========================================================
# 1. UPLOAD AN IMAGE
# =========================================================

st.markdown(
    """
    <div class="center-section-title">
        1. Upload an Image
    </div>
    """,
    unsafe_allow_html=True
)

upload_left, upload_center, upload_right = st.columns(
    [1, 1.2, 1]
)

with upload_center:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )


# =========================================================
# MAIN APP
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image_matrix = load_image(
        uploaded_file
    )


    # =====================================================
    # 2. CHOOSE YOUR FILTER
    # =====================================================

    st.markdown(
        """
        <div class="center-section-title">
            2. Choose Your Filter
        </div>
        """,
        unsafe_allow_html=True
    )


    filter_left, filter_center, filter_right = st.columns(
        [1, 1.2, 1]
    )

    with filter_center:

        filter_name = st.selectbox(
            "Filter",
            list(FILTER_DESCRIPTIONS.keys())
        )


        # -------------------------------------------------
        # Filter description
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="info-card">
                <b>{filter_name}</b><br>
                {FILTER_DESCRIPTIONS[filter_name]}
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # Filter parameters
        # -------------------------------------------------

        filter_kwargs = {}


        if filter_name == "Box Blur":

            size = st.slider(
                "Kernel Size",
                min_value=3,
                max_value=15,
                value=7,
                step=2
            )

            filter_kwargs["size"] = size


        elif filter_name == "Gaussian Blur":

            size = st.slider(
                "Kernel Size",
                min_value=3,
                max_value=15,
                value=5,
                step=2
            )

            sigma = st.slider(
                "Sigma",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )

            filter_kwargs["size"] = size
            filter_kwargs["sigma"] = sigma


        elif filter_name == "Sharpen":

            amount = st.slider(
                "Sharpening Strength",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1
            )

            filter_kwargs["amount"] = amount


        elif filter_name == "Emboss":

            intensity = st.slider(
                "Emboss Strength",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1
            )

            filter_kwargs["intensity"] = intensity


        # -------------------------------------------------
        # Apply button
        # -------------------------------------------------

        apply_button = st.button(
            "Apply Filter"
        )


    # =====================================================
    # PROCESSING
    # =====================================================

    if apply_button:

        # Increase counter so the scroll script is
        # rendered again every time the button is clicked.

        st.session_state.scroll_counter += 1

        try:

            # =================================================
            # IMAGE INFORMATION
            # =================================================

            if image_matrix.ndim == 2:

                height, width = image_matrix.shape
                channels = 1

            else:

                height, width, channels = image_matrix.shape


            st.markdown(
                """
                <div class="section-title">
                    Image Information
                </div>
                """,
                unsafe_allow_html=True
            )


            info_col1, info_col2, info_col3 = st.columns(3)


            with info_col1:

                st.metric(
                    "Width",
                    f"{width} px"
                )


            with info_col2:

                st.metric(
                    "Height",
                    f"{height} px"
                )


            with info_col3:

                st.metric(
                    "Channels",
                    channels
                )


            # =================================================
            # APPLY SELECTED FILTER
            # =================================================

            start_time = time.perf_counter()

            result_matrix = apply_filter(
                image_matrix,
                filter_name,
                padding_mode="reflect",
                **filter_kwargs
            )

            processing_time = (
                time.perf_counter() - start_time
            )

            result_image = matrix_to_image(
                result_matrix
            )


            # =================================================
            # MSE + PSNR
            # =================================================

            mse = calculate_mse(
                image_matrix,
                result_matrix
            )

            psnr = calculate_psnr(
                image_matrix,
                result_matrix
            )


            # =================================================
            # 3. RESULTS
            # =================================================

            st.markdown(
                """
                <div id="results-section"></div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="section-title">
                    3. Results
                </div>
                """,
                unsafe_allow_html=True
            )


            result_col1, result_col2 = st.columns(2)


            with result_col1:

                st.markdown(
                    "### Original Image"
                )

                st.image(
                    uploaded_file,
                    width=340
                )


            with result_col2:

                st.markdown(
                    "### Processed Image"
                )

                st.image(
                    result_image,
                    width=340
                )


            # =================================================
            # PROCESSING INFORMATION
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    Processing Information
                </div>
                """,
                unsafe_allow_html=True
            )


            process_col1, process_col2 = st.columns(2)


            with process_col1:

                st.markdown(
                    "Filter Applied"
                )

                st.markdown(
                    f"""
                    <div style="
                        font-size: 30px;
                        color: #1e3a5f;
                        margin-top: 4px;
                    ">
                        {filter_name}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with process_col2:

                st.markdown(
                    "Processing Time"
                )

                st.markdown(
                    f"""
                    <div style="
                        font-size: 30px;
                        color: #1e3a5f;
                        margin-top: 4px;
                    ">
                        {processing_time:.4f} s
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # =================================================
            # MATHEMATICAL KERNEL
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    Mathematical Kernel
                </div>
                """,
                unsafe_allow_html=True
            )


            kernel = get_kernel(
                filter_name,
                **filter_kwargs
            )


            kernel_col1, kernel_col2 = st.columns(
                [1, 2]
            )


            with kernel_col1:

                st.code(
                    np.array2string(
                        kernel,
                        precision=3,
                        suppress_small=True
                    ),
                    language="text"
                )


            with kernel_col2:

                st.markdown(
                    f"**{filter_name}**"
                )

                st.write(
                    "This kernel is applied across the image "
                    "using 2D convolution."
                )

                st.write(
                    "Each position of the kernel performs a "
                    "mathematical operation on the corresponding "
                    "pixels of the image."
                )


            # =================================================
            # 4. MATHEMATICAL ANALYSIS
            # =================================================

            st.markdown(
                """
                <div id="analysis-section"></div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="section-title">
                    4. Mathematical Analysis
                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------------------------------
            # MSE + PSNR DEFINITIONS
            # -------------------------------------------------

            st.markdown(
                "MSE measures the average squared difference "
                "between the original and processed images."
            )

            st.markdown(
                "PSNR measures the difference on a logarithmic "
                "scale and is expressed in decibels."
            )


            metric_col1, metric_col2 = st.columns(2)


            with metric_col1:

                st.metric(
                    "Mean Squared Error (MSE)",
                    f"{mse:.4f}"
                )


            with metric_col2:

                if np.isinf(psnr):

                    psnr_text = "∞ dB"

                else:

                    psnr_text = f"{psnr:.2f} dB"

                st.metric(
                    "Peak Signal-to-Noise Ratio (PSNR)",
                    psnr_text
                )


            # =================================================
            # PROCESSING TIME COMPARISON
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    Processing Time Comparison
                </div>
                """,
                unsafe_allow_html=True
            )


            benchmark_results = benchmark_all_filters(
                image_matrix
            )


            benchmark_results = {
                name: value
                for name, value in benchmark_results.items()
                if value is not None
            }


            if benchmark_results:

                time_names = list(
                    benchmark_results.keys()
                )

                time_values = list(
                    benchmark_results.values()
                )


                time_fig = go.Figure()


                time_fig.add_trace(
                    go.Bar(
                        x=time_names,
                        y=time_values,
                        name="Processing Time",
                        marker_color="#4A90E2"
                    )
                )


                time_fig.update_layout(
                    title="Processing Time Comparison",
                    xaxis_title="Filter",
                    yaxis_title="Processing Time (seconds)",
                    template="plotly_white",
                    height=450,
                    margin=dict(
                        l=60,
                        r=30,
                        t=70,
                        b=100
                    ),
                    showlegend=False
                )


                st.plotly_chart(
                    time_fig,
                    width="stretch",
                    config={
                        "scrollZoom": True,
                        "displaylogo": False
                    },
                    key="processing_time_chart"
                )


            # =================================================
            # FILTER PERFORMANCE
            # =================================================

            st.markdown(
                """
                <div class="section-title">
                    Filter Performance
                </div>
                """,
                unsafe_allow_html=True
            )


            performance_filters = {

                "Box Blur": {
                    "size": 5
                },

                "Gaussian Blur": {
                    "size": 5,
                    "sigma": 1.0
                },

                "Sharpen": {
                    "amount": 1.0
                },

                "Edge Detection": {},

                "Sobel X": {},

                "Sobel Y": {},

                "Emboss": {
                    "intensity": 1.0
                }
            }


            mse_values = {}
            psnr_values = {}


            for name, kwargs in performance_filters.items():

                try:

                    filtered_image = apply_filter(
                        image_matrix,
                        name,
                        padding_mode="reflect",
                        **kwargs
                    )


                    mse_value = calculate_mse(
                        image_matrix,
                        filtered_image
                    )


                    psnr_value = calculate_psnr(
                        image_matrix,
                        filtered_image
                    )


                    mse_values[name] = mse_value
                    psnr_values[name] = psnr_value


                except Exception:

                    continue


            # =================================================
            # MSE GRAPH - BLUE
            # =================================================

            if mse_values:

                performance_names = list(
                    mse_values.keys()
                )

                mse_data = [
                    mse_values[name]
                    for name in performance_names
                ]


                mse_fig = go.Figure()


                mse_fig.add_trace(
                    go.Bar(
                        x=performance_names,
                        y=mse_data,
                        name="MSE",
                        marker_color="#4A90E2"
                    )
                )


                mse_fig.update_layout(
                    title="MSE Comparison",
                    xaxis_title="Filter",
                    yaxis_title="MSE",
                    template="plotly_white",
                    height=450,
                    margin=dict(
                        l=60,
                        r=30,
                        t=70,
                        b=100
                    ),
                    showlegend=False
                )


                st.plotly_chart(
                    mse_fig,
                    width="stretch",
                    config={
                        "scrollZoom": True,
                        "displaylogo": False
                    },
                    key="mse_chart"
                )


            # =================================================
            # PSNR GRAPH - ORANGE
            # =================================================

            if psnr_values:

                psnr_names = list(
                    psnr_values.keys()
                )

                psnr_data = [
                    psnr_values[name]
                    for name in psnr_names
                ]


                psnr_fig = go.Figure()


                psnr_fig.add_trace(
                    go.Bar(
                        x=psnr_names,
                        y=psnr_data,
                        name="PSNR",
                        marker_color="#F39C12"
                    )
                )


                psnr_fig.update_layout(
                    title="PSNR Comparison",
                    xaxis_title="Filter",
                    yaxis_title="PSNR (dB)",
                    template="plotly_white",
                    height=450,
                    margin=dict(
                        l=60,
                        r=30,
                        t=70,
                        b=100
                    ),
                    showlegend=False
                )


                st.plotly_chart(
                    psnr_fig,
                    width="stretch",
                    config={
                        "scrollZoom": True,
                        "displaylogo": False
                    },
                    key="psnr_chart"
                )


            # =================================================
            # AUTO SCROLL
            # =================================================

            components.html(
                f"""
                <p style="display:none;">
                    {st.session_state.scroll_counter}
                </p>

                <script>

                function scrollToAnalysis() {{

                    const target =
                        window.parent.document.getElementById(
                            "analysis-section"
                        );

                    if (target) {{

                        target.scrollIntoView({{
                            behavior: "smooth",
                            block: "start"
                        }});

                    }} else {{

                        setTimeout(
                            scrollToAnalysis,
                            100
                        );

                    }}

                }}

                setTimeout(
                    scrollToAnalysis,
                    500
                );

                </script>
                """,
                height=0
            )


        except Exception as e:

            st.error(
                f"Something went wrong while processing "
                f"the image: {e}"
            )
