import time

from core.filters import apply_filter


def benchmark_filter(
    image_matrix,
    filter_name,
    filter_kwargs=None
):
    """
    Measure the processing time of one filter.
    """

    if filter_kwargs is None:
        filter_kwargs = {}

    start_time = time.perf_counter()

    result = apply_filter(
        image_matrix,
        filter_name,
        padding_mode="reflect",
        **filter_kwargs
    )

    processing_time = (
        time.perf_counter() - start_time
    )

    return result, processing_time


def benchmark_all_filters(image_matrix):
    """
    Measure the processing time of all filters
    using standard parameters.
    """

    filters = {

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

    results = {}

    for filter_name, filter_kwargs in filters.items():

        try:

            start_time = time.perf_counter()

            apply_filter(
                image_matrix,
                filter_name,
                padding_mode="reflect",
                **filter_kwargs
            )

            processing_time = (
                time.perf_counter() - start_time
            )

            results[filter_name] = processing_time

        except Exception:

            results[filter_name] = None

    return results
