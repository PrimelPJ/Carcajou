"""Dynamic-object masking.

Two implementations behind one interface:

:class:`SimulatedMask`
    Used by the synthetic benchmark. It does not look at pixels; it applies a
    known recall and false-positive rate to landmarks whose true label is
    already known. This is the honest way to sweep mask quality, because the
    thing the ablation is trying to isolate is *the effect of masking*, not the
    accuracy of one particular checkpoint on one particular dataset. Sweeping
    recall from 0 to 1 draws a curve; running one network draws a dot.

:class:`OnnxSemanticMask`
    Used on real imagery. Runs an ONNX semantic segmentation model and returns
    a per-pixel boolean for the dynamic classes. It is not exercised by any
    number in the benchmark, and it must not be until the KITTI loader has been
    validated against a real drive.

Failure structure matters more than the failure rate. An earlier version of
this class applied recall per landmark, i.i.d., which is not how segmenters
fail: a network does not randomly forget 8 % of the pixels on a car filling a
quarter of the frame. It misses *objects* (small, distant, occluded, unusual),
and it leaks *boundary pixels* even on objects it catches, which is what the
dilation in :class:`OnnxSemanticMask` exists to suppress. The simulated mask
mirrors that: one per-epoch draw decides whether the object was detected at
all, and a small per-point rate models residual boundary leak on detected
objects. The i.i.d. version overstated contamination by an order of magnitude
on exactly the near-field objects real segmenters are best at.

Default rates are a plausible operating point for a Cityscapes-trained
real-time segmenter with dilation, not a measurement, and nothing in the
benchmark should quote them as one. Sweep them; that is what they are for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Cityscapes train ids for things that move. Rider and bicycle are included
# because a cyclist's features are as poisonous to VO as a car's.
CITYSCAPES_DYNAMIC = (11, 12, 13, 14, 15, 16, 17, 18)  # person..bicycle


@dataclass(frozen=True)
class SimulatedMask:
    """Label-space mask with configurable, imperfect quality.

    Attributes
    ----------
    object_recall : probability, per epoch, that the segmenter detects the
        dynamic object at all. ``1.0`` is an oracle, ``0.0`` is the mask-off
        ablation. A missed epoch leaks *every* dynamic point at once, which is
        the catastrophic case; the filter's innovation gate is the last line
        of defence against it, and the ablation should show whether it holds.
    point_leak : per-point boundary leak on a *detected* object. Features
        sitting on the silhouette edge that dilation failed to cover.
    false_positive_rate : fraction of static landmarks wrongly masked out.
        Costs information but never corrupts the estimate, which is the
        asymmetry that makes segmenters worth tuning towards recall.
    """

    object_recall: float = 0.97
    point_leak: float = 0.02
    false_positive_rate: float = 0.02

    def keep(self, is_dynamic: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Boolean mask of landmarks that survive to the pose estimator."""
        is_dynamic = np.asarray(is_dynamic, bool)
        detected = rng.random() < self.object_recall
        u = rng.random(is_dynamic.shape)
        dyn_masked = detected & (u >= self.point_leak)
        masked = np.where(is_dynamic, dyn_masked, u < self.false_positive_rate)
        return ~masked


MASK_OFF = SimulatedMask(object_recall=0.0, point_leak=0.0, false_positive_rate=0.0)
MASK_ORACLE = SimulatedMask(object_recall=1.0, point_leak=0.0, false_positive_rate=0.0)
MASK_REALISTIC = SimulatedMask(object_recall=0.97, point_leak=0.02, false_positive_rate=0.02)


class OnnxSemanticMask:
    """ONNX semantic segmentation, for the real-image path only.

    Parameters
    ----------
    model_path : ONNX file exporting ``(1,3,H,W) float32 -> (1,C,H,W) logits``
    dynamic_classes : class indices treated as movable
    dilate_px : morphological dilation applied to the dynamic mask. Segmentation
        boundaries are systematically tight, and a feature sitting one pixel
        outside a car's silhouette is still on the car. Erring outward costs
        static features, which is the cheap direction to be wrong in.
    """

    def __init__(
        self,
        model_path: str,
        dynamic_classes: tuple[int, ...] = CITYSCAPES_DYNAMIC,
        input_size: tuple[int, int] = (512, 1024),
        dilate_px: int = 6,
        providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OnnxSemanticMask needs onnxruntime: pip install carcajou[vision]"
            ) from exc
        self._sess = ort.InferenceSession(model_path, providers=list(providers))
        self._input = self._sess.get_inputs()[0].name
        self.dynamic_classes = np.asarray(dynamic_classes)
        self.input_size = input_size
        self.dilate_px = dilate_px

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return an ``(H,W)`` boolean array, ``True`` where the pixel is dynamic."""
        import cv2  # local import: only the real-image path needs OpenCV

        h, w = image.shape[:2]
        net_h, net_w = self.input_size
        x = cv2.resize(image, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        x = x[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
        x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None].astype(np.float32))

        logits = self._sess.run(None, {self._input: x})[0][0]
        labels = logits.argmax(axis=0).astype(np.int32)
        dyn = np.isin(labels, self.dynamic_classes).astype(np.uint8)

        if self.dilate_px > 0:
            k = np.ones((self.dilate_px, self.dilate_px), np.uint8)
            dyn = cv2.dilate(dyn, k)
        return cv2.resize(dyn, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
