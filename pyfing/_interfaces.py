from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .definitions import Image, Minutia, Parameters


class SegmentationParameters(Parameters):
    """Base class for the parameters of a segmentation method."""
    pass


class SegmentationAlgorithm(ABC):
    """Base class for segmentation methods."""

    def __init__(self, parameters: SegmentationParameters):
        self.parameters = parameters

    @abstractmethod
    def run(self, image: Image, intermediate_results=None) -> Image:
        raise NotImplementedError

    def run_on_db(self, images: list[Image]) -> list[Image]:
        return [self.run(img) for img in images]


class GmfsParameters(SegmentationParameters):
    """Parameters of the GMFS segmentation method."""

    def __init__(
        self,
        sigma=13 / 3,
        percentile=95,
        threshold=0.2,
        closing_count=6,
        opening_count=12,
        image_dpi=500,
    ):
        self.sigma = sigma
        self.percentile = percentile
        self.threshold = threshold
        self.closing_count = closing_count
        self.opening_count = opening_count
        self.image_dpi = image_dpi


class SufsParameters(SegmentationParameters):
    """Parameters of the SUFS segmentation method."""

    def __init__(
        self,
        dnn_input_dpi=500,
        dnn_input_size_multiple=64,
        image_dpi=500,
        threshold=0.5,
        border=33,
    ):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple
        self.image_dpi = image_dpi
        self.threshold = threshold
        self.border = border


class OrientationEstimationParameters(Parameters):
    """Base class for the parameters of an orientation estimation method."""
    pass


class OrientationEstimationAlgorithm(ABC):
    """Base class for orientation estimation methods."""

    def __init__(self, parameters: OrientationEstimationParameters):
        self.parameters = parameters

    @abstractmethod
    def run(
        self,
        image: Image,
        mask: Image | None = None,
        dpi: int = 500,
        intermediate_results=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def run_on_db(
        self,
        images: list[Image],
        masks: list[Image],
        dpi_of_images: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self.run(img, mask, dpi) for img, mask, dpi in zip(images, masks, dpi_of_images)]


class GbfoeParameters(OrientationEstimationParameters):
    """Parameters of GBFOE (Gradient-Based Fingerprint Orientation Estimation)."""

    def __init__(
        self,
        sigma_base=27,
        sigma_multiplier=43,
        sigma_smooth=1.25,
        median_size=5,
        percentile=19,
    ):
        self.sigma_base = sigma_base
        self.sigma_multiplier = sigma_multiplier
        self.sigma_smooth = sigma_smooth
        self.median_size = median_size
        self.percentile = percentile


class SnfoeParameters(OrientationEstimationParameters):
    """Parameters of SNFOE (Simple Network for Fingerprint Orientation Estimation)."""

    def __init__(self, dnn_input_dpi=500, dnn_input_size_multiple=32):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple


class FrequencyEstimationParameters(Parameters):
    """Base class for the parameters of a frequency estimation method."""
    pass


class FrequencyEstimationAlgorithm(ABC):
    """Base class for frequency estimation methods."""

    def __init__(self, parameters: FrequencyEstimationParameters):
        self.parameters = parameters

    @abstractmethod
    def run(
        self,
        image: Image,
        mask: Image,
        orientation_field: np.ndarray,
        dpi: int = 500,
        intermediate_results=None,
    ) -> np.ndarray:
        raise NotImplementedError

    def run_on_db(
        self,
        images: list[Image],
        masks: list[Image],
        orientation_fields: list[np.ndarray],
        dpi_of_images: list[int],
    ) -> list[np.ndarray]:
        return [
            self.run(img, mask, orientation_field, dpi)
            for img, mask, orientation_field, dpi in zip(images, masks, orientation_fields, dpi_of_images)
        ]


class SkffeParameters(FrequencyEstimationParameters):
    """Parameters of SKFFE (Skeleton-based Fingerprint Frequency Estimation)."""

    def __init__(self, period_min=5, period_max=18, median_blur_size=5, final_blur_size=5):
        self.period_min = period_min
        self.period_max = period_max
        self.median_blur_size = median_blur_size
        self.final_blur_size = final_blur_size


class XsffeParameters(FrequencyEstimationParameters):
    """Parameters of XSFFE (X-Signature Fingerprint Frequency Estimation)."""

    def __init__(
        self,
        window_size=(23, 43),
        step=8,
        border=7,
        min_background_distance=11,
        period_min=5,
        period_max=20,
        min_valid_distances=4,
        diffusion_size=21,
        median_size=5,
        blur_size=3,
        final_blur_size=33,
    ):
        self.window_size = window_size
        self.step = step
        self.border = border
        self.min_background_distance = min_background_distance
        self.min_valid_distances = min_valid_distances
        self.period_min = period_min
        self.period_max = period_max
        self.median_size = median_size
        self.blur_size = blur_size
        self.final_blur_size = final_blur_size
        self.diffusion_size = diffusion_size


class SnffeParameters(FrequencyEstimationParameters):
    """Parameters of SNFFE (Simple Network for Fingerprint Frequency Estimation)."""

    def __init__(self, dnn_input_dpi=500, dnn_input_size_multiple=32):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple


class EnhancementParameters(Parameters):
    """Base class for the parameters of an enhancement method."""
    pass


class EnhancementAlgorithm(ABC):
    """Base class for enhancement methods."""

    def __init__(self, parameters: EnhancementParameters):
        self.parameters = parameters

    @abstractmethod
    def run(
        self,
        image: Image,
        mask: Image,
        orientation_field: np.ndarray,
        ridge_periods: np.ndarray,
        dpi: int = 500,
        intermediate_results=None,
    ) -> Image:
        raise NotImplementedError

    def run_on_db(
        self,
        images: list[Image],
        masks: list[Image],
        orientation_fields: list[np.ndarray],
        ridge_periods: list[np.ndarray],
        dpi_of_images: list[int],
    ) -> list[Image]:
        return [
            self.run(img, mask, orientation_field, rp, dpi)
            for img, mask, orientation_field, rp, dpi in zip(
                images, masks, orientation_fields, ridge_periods, dpi_of_images
            )
        ]


class GbfenParameters(EnhancementParameters):
    """Parameters of GBFEN (Gabor-Based Fingerprint Enhancement)."""

    def __init__(self, orientations_count=16, periods_count=9, period_min=5, period_max=20):
        self.orientations_count = orientations_count
        self.periods_count = periods_count
        self.period_min = period_min
        self.period_max = period_max


class SnfenParameters(EnhancementParameters):
    """Parameters of SNFEN (Simple Network for Fingerprint Enhancement)."""

    def __init__(self, dnn_input_dpi=500, dnn_input_size_multiple=32):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple


class EndToEndMinutiaExtractionParameters(Parameters):
    """Base class for the parameters of an end-to-end minutia extraction method."""
    pass


class EndToEndMinutiaExtractionAlgorithm(ABC):
    """Base class for end-to-end minutia extraction methods."""

    def __init__(self, parameters: EndToEndMinutiaExtractionParameters):
        self.parameters = parameters

    @abstractmethod
    def run(
        self,
        image: Image,
        dpi: int = 500,
        intermediate_results: list | None = None,
    ) -> list[Minutia]:
        raise NotImplementedError

    def run_on_db(self, images: list[Image], dpi_of_images: list[int] | None = None) -> list[list[Minutia]]:
        dpi_list = [500] * len(images) if dpi_of_images is None else dpi_of_images
        return [self.run(img, dpi) for img, dpi in zip(images, dpi_list)]


class LeaderParameters(EndToEndMinutiaExtractionParameters):
    """Parameters for LEADER minutia extraction."""

    def __init__(
        self,
        dnn_input_dpi=500,
        dnn_input_size_multiple=32,
        minutia_quality_threshold=0.15,
        type_threshold=0.5,
    ):
        self.dnn_input_dpi = dnn_input_dpi
        self.dnn_input_size_multiple = dnn_input_size_multiple
        self.minutia_quality_threshold = minutia_quality_threshold
        self.type_threshold = type_threshold
