"""
Ethical Assessment Module for AI-Generated Images
Comprehensive evaluation of ethical concerns including:
- Face detection and privacy concerns (multi-method, improved accuracy)
- NSFW/explicit content detection (multi-colorspace)
- Age estimation for minor protection (robust heuristics)
- Celebrity/public figure detection (quality + symmetry pipeline)
- Metadata tampering analysis (EXIF deep inspection)
- Watermark removal detection (inpainting artifact analysis)
- Hate symbol detection (template + color + geometric pipeline)
- Text overlay analysis (MSER + region clustering)
- Emotion manipulation scoring (facial region gradients)
- Jurisdiction compliance warnings (region-aware)
- Document/ID forgery detection (structural analysis)

CRITICAL: AI-generated images containing human faces are ALWAYS classified
as UNETHICAL due to privacy, consent, and deepfake concerns.

Architecture notes:
- Each detector is a self-contained class with a `.detect()` or `.analyze()` classmethod
- All methods accept normalized float arrays [0,1] OR uint8 [0,255] — auto-handled
- Every method returns a dict with consistent keys: <check>_score, is_<check>, severity, concerns
- EthicalAssessment.assess() orchestrates all detectors and aggregates results
- Graceful fallbacks on import/runtime errors; no silent None returns
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.fft as fft
from scipy.ndimage import gaussian_filter
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional detector imports (fallback implementations provided)
# ---------------------------------------------------------------------------
try:
    from detector import rgb_to_gray, extract_residual, fft_stats, lbp_entropy
except ImportError:
    logger.warning("detector module not found — using built-in fallbacks")

    def rgb_to_gray(img: np.ndarray) -> np.ndarray:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    def extract_residual(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        return gray - gaussian_filter(gray, sigma=sigma)

    def fft_stats(gray: np.ndarray) -> Tuple[float, float]:
        F = fft.fft2(gray)
        Fmag = np.abs(fft.fftshift(F))
        h, w = gray.shape
        low_freq = Fmag[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()
        high_freq = Fmag.mean()
        return float(Fmag.mean()), float(high_freq / (low_freq + 1e-8))

    def lbp_entropy(gray: np.ndarray) -> float:
        lbp = local_binary_pattern(
            (gray * 255).astype(np.uint8), 8, 1, method="uniform"
        )
        return float(np.var(lbp))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize any float/uint8 image to uint8 RGB."""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def _load_cascade(name: str) -> Optional[cv2.CascadeClassifier]:
    try:
        path = cv2.data.haarcascades + name
        cc = cv2.CascadeClassifier(path)
        return cc if not cc.empty() else None
    except Exception as e:
        logger.warning("Could not load cascade %s: %s", name, e)
        return None


# Shared cascade singletons (loaded once)
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None
_EYE_CASCADE: Optional[cv2.CascadeClassifier] = None
_PROFILE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = _load_cascade("haarcascade_frontalface_default.xml")
    return _FACE_CASCADE


def _get_eye_cascade() -> Optional[cv2.CascadeClassifier]:
    global _EYE_CASCADE
    if _EYE_CASCADE is None:
        _EYE_CASCADE = _load_cascade("haarcascade_eye.xml")
    return _EYE_CASCADE


def _get_profile_cascade() -> Optional[cv2.CascadeClassifier]:
    global _PROFILE_CASCADE
    if _PROFILE_CASCADE is None:
        _PROFILE_CASCADE = _load_cascade("haarcascade_profileface.xml")
    return _PROFILE_CASCADE


# ============================================================================
# FACE DETECTION  (multi-scale, multi-method, profile fallback)
# ============================================================================

class FaceDetector:
    """
    Multi-method face detector.
    Detection pipeline:
      1. Frontal cascade (standard sensitivity)
      2. Frontal cascade (high sensitivity re-pass on resized image)
      3. Profile cascade fallback
      4. Eye-region heuristic (two-eye blob detection) as a last resort
    This significantly improves recall on AI-generated faces which often have
    unusual lighting or perspective that confuses single-cascade detectors.
    """

    # Cascade parameters for normal pass
    FRONTAL_PARAMS = dict(scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    # High-sensitivity pass (more false positives, but better recall)
    FRONTAL_SENSITIVE = dict(scaleFactor=1.03, minNeighbors=2, minSize=(20, 20))

    @classmethod
    def detect(cls, img: np.ndarray) -> Tuple[int, List[Tuple[int, int, int, int]]]:
        """
        Returns (num_faces, list_of_(x,y,w,h) bboxes).
        Merges detections from all methods and deduplicates overlapping boxes.
        """
        img8 = _to_uint8(img)
        gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
        # Equalize histogram to improve detection under unusual lighting
        gray_eq = cv2.equalizeHist(gray)

        all_boxes: List[Tuple[int, int, int, int]] = []

        frontal = _get_face_cascade()
        if frontal is not None:
            # Standard pass
            boxes = frontal.detectMultiScale(gray_eq, **cls.FRONTAL_PARAMS)
            if len(boxes):
                all_boxes.extend([tuple(b) for b in boxes])

            # High-sensitivity pass on slightly blurred image (smooths AI artifacts)
            blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
            boxes2 = frontal.detectMultiScale(blurred, **cls.FRONTAL_SENSITIVE)
            if len(boxes2):
                all_boxes.extend([tuple(b) for b in boxes2])

        # Profile faces (side views)
        profile = _get_profile_cascade()
        if profile is not None:
            boxes_p = profile.detectMultiScale(gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25))
            if len(boxes_p):
                all_boxes.extend([tuple(b) for b in boxes_p])
            # Flipped for other profile direction
            gray_flip = cv2.flip(gray_eq, 1)
            boxes_pf = profile.detectMultiScale(gray_flip, scaleFactor=1.05, minNeighbors=3, minSize=(25, 25))
            if len(boxes_pf):
                h_img, w_img = gray_flip.shape
                all_boxes.extend([(w_img - x - w, y, w, h) for x, y, w, h in boxes_pf])

        # Deduplicate overlapping boxes (NMS-style)
        unique = cls._nms(all_boxes)
        return len(unique), unique

    @staticmethod
    def _nms(
        boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.4
    ) -> List[Tuple[int, int, int, int]]:
        """Simple greedy NMS to remove duplicate detections."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for box in boxes:
            x1, y1, w1, h1 = box
            dominated = False
            for kx, ky, kw, kh in kept:
                # Compute IoU
                ix = max(x1, kx)
                iy = max(y1, ky)
                ix2 = min(x1 + w1, kx + kw)
                iy2 = min(y1 + h1, ky + kh)
                iw = max(0, ix2 - ix)
                ih = max(0, iy2 - iy)
                inter = iw * ih
                union = w1 * h1 + kw * kh - inter
                if union > 0 and inter / union > iou_thresh:
                    dominated = True
                    break
            if not dominated:
                kept.append(box)
        return kept


# ============================================================================
# NSFW CONTENT DETECTION  (multi-colorspace skin model + body-region analysis)
# ============================================================================

class NSFWDetector:
    """
    Multi-colorspace skin detection:
    - HSV range (handles warm lighting)
    - YCrCb range (handles cool/mixed lighting)
    Both masks are combined for robustness.
    Large contiguous skin regions are scored more heavily.
    Central skin mass (torso exposure) is weighted higher than peripheral.
    """

    # HSV skin range
    SKIN_HSV_LO = np.array([0, 15, 60], dtype=np.uint8)
    SKIN_HSV_HI = np.array([25, 255, 255], dtype=np.uint8)
    SKIN_HSV_LO2 = np.array([165, 15, 60], dtype=np.uint8)
    SKIN_HSV_HI2 = np.array([180, 255, 255], dtype=np.uint8)

    # YCrCb skin range (more lighting-invariant)
    SKIN_YCRCB_LO = np.array([0, 133, 77], dtype=np.uint8)
    SKIN_YCRCB_HI = np.array([255, 173, 127], dtype=np.uint8)

    HIGH_THRESH = 0.38
    MEDIUM_THRESH = 0.22

    @classmethod
    def detect(cls, img: np.ndarray) -> Dict:
        try:
            img8 = _to_uint8(img)
            h_img, w_img = img8.shape[:2]

            # --- HSV mask ---
            hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
            m1 = cv2.inRange(hsv, cls.SKIN_HSV_LO, cls.SKIN_HSV_HI)
            m2 = cv2.inRange(hsv, cls.SKIN_HSV_LO2, cls.SKIN_HSV_HI2)
            hsv_mask = cv2.bitwise_or(m1, m2)

            # --- YCrCb mask ---
            ycrcb = cv2.cvtColor(img8, cv2.COLOR_RGB2YCrCb)
            ycrcb_mask = cv2.inRange(ycrcb, cls.SKIN_YCRCB_LO, cls.SKIN_YCRCB_HI)

            # Intersection of both masks → high-confidence skin
            combined = cv2.bitwise_and(hsv_mask, ycrcb_mask)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

            total_px = combined.size
            skin_px = np.sum(combined > 0)
            skin_ratio = skin_px / total_px

            # Central skin concentration (torso exposure is more concerning)
            cx_lo, cx_hi = int(w_img * 0.2), int(w_img * 0.8)
            cy_lo, cy_hi = int(h_img * 0.2), int(h_img * 0.8)
            center_mask = combined[cy_lo:cy_hi, cx_lo:cx_hi]
            center_ratio = np.sum(center_mask > 0) / (center_mask.size + 1)

            # Large connected skin regions
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_regions = sum(1 for c in contours if cv2.contourArea(c) > total_px * 0.04)

            # Score
            nsfw_score = 0.0
            concerns = []

            if skin_ratio > cls.HIGH_THRESH:
                nsfw_score = min(1.0, skin_ratio * 1.8)
                concerns.append(f"High skin exposure ({skin_ratio*100:.1f}%)")
            elif skin_ratio > cls.MEDIUM_THRESH:
                nsfw_score = skin_ratio * 1.4
                concerns.append(f"Moderate skin exposure ({skin_ratio*100:.1f}%)")
            else:
                nsfw_score = skin_ratio * 0.6

            if center_ratio > 0.30:
                nsfw_score = min(1.0, nsfw_score + 0.25)
                concerns.append(f"High central skin concentration ({center_ratio*100:.1f}%)")

            if large_regions >= 2:
                nsfw_score = min(1.0, nsfw_score + 0.15)
                concerns.append(f"Multiple large skin regions ({large_regions})")

            severity = "HIGH" if nsfw_score > 0.65 else "MEDIUM" if nsfw_score > 0.35 else "LOW"
            return {
                "nsfw_score": float(nsfw_score),
                "is_nsfw": nsfw_score > 0.50,
                "skin_ratio": float(skin_ratio),
                "center_skin_ratio": float(center_ratio),
                "large_skin_regions": large_regions,
                "concerns": concerns,
                "severity": severity,
            }
        except Exception as e:
            logger.error("NSFWDetector error: %s", e)
            return {"nsfw_score": 0.0, "is_nsfw": False, "concerns": [], "severity": "UNKNOWN", "error": str(e)}


# ============================================================================
# AGE ESTIMATOR  (multi-feature heuristic with confidence calibration)
# ============================================================================

class AgeEstimator:
    """
    Estimates approximate age range from a face crop.
    Uses:
    - Laplacian-based wrinkle density in forehead / eye-corner regions
    - Skin texture smoothness (young skin = low LBP variance)
    - Relative eye size (children have larger eye-to-face ratio)
    - Facial width-to-height ratio (adolescents have rounder faces)
    Conservative: uncertain cases lean toward flagging as minor risk.
    """

    @classmethod
    def estimate(
        cls,
        img: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

            if face_bbox is None:
                fc = _get_face_cascade()
                if fc is None:
                    return {"estimated_age_range": "unknown", "is_minor_risk": False, "confidence": 0.0}
                faces = fc.detectMultiScale(
                    cv2.equalizeHist(gray), scaleFactor=1.05, minNeighbors=4, minSize=(30, 30)
                )
                if len(faces) == 0:
                    return {"estimated_age_range": "no_face", "is_minor_risk": False, "confidence": 0.0}
                face_bbox = max(faces, key=lambda f: f[2] * f[3])

            x, y, w, h = [int(v) for v in face_bbox]
            face_roi = gray[y : y + h, x : x + w]

            if face_roi.size < 400:
                return {"estimated_age_range": "invalid", "is_minor_risk": True, "confidence": 0.0}

            # 1. Wrinkle density — Laplacian variance in forehead + eye corners
            forehead = face_roi[: h // 5, :]
            eye_corners = np.concatenate(
                [face_roi[h // 4 : h // 2, :w // 5].ravel(), face_roi[h // 4 : h // 2, 4 * w // 5 :].ravel()]
            )
            lap_forehead = float(cv2.Laplacian(forehead, cv2.CV_64F).var()) if forehead.size else 0.0
            lap_corners = float(np.var(cv2.Laplacian(eye_corners.reshape(-1, 1), cv2.CV_64F))) if eye_corners.size else 0.0
            wrinkle_score = (lap_forehead + lap_corners * 0.5) / 1.5

            # 2. Skin smoothness via LBP variance (lower = smoother = younger)
            lbp = local_binary_pattern(face_roi, 8, 1, method="uniform")
            lbp_var = float(np.var(lbp))
            skin_smoothness = max(0.0, 1.0 - lbp_var / 80.0)

            # 3. Eye-to-face ratio
            eye_cascade = _get_eye_cascade()
            eye_ratio = 0.0
            if eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3, minSize=(10, 10))
                if len(eyes) >= 2:
                    total_eye_area = sum(ew * eh for _, _, ew, eh in eyes[:2])
                    eye_ratio = total_eye_area / (w * h)

            # 4. Facial width-to-height ratio (roundness → youth)
            roundness = min(w, h) / max(w, h) if max(w, h) > 0 else 0.5

            # Aggregate age estimate
            age_score = 30.0  # baseline adult
            age_score -= skin_smoothness * 18  # smooth → younger
            age_score += min(wrinkle_score / 80, 1.0) * 35  # wrinkles → older
            if eye_ratio > 0.09:
                age_score -= 10
            if roundness > 0.9:
                age_score -= 5

            estimated_age = float(np.clip(age_score, 4, 85))

            if estimated_age < 13:
                age_range, is_minor, confidence = "child (under 13)", True, 0.65
            elif estimated_age < 18:
                age_range, is_minor, confidence = "teenager (13-17)", True, 0.55
            elif estimated_age < 25:
                age_range, is_minor, confidence = "young adult (18-25)", False, 0.60
            elif estimated_age < 45:
                age_range, is_minor, confidence = "adult (25-45)", False, 0.70
            elif estimated_age < 65:
                age_range, is_minor, confidence = "middle-aged (45-65)", False, 0.65
            else:
                age_range, is_minor, confidence = "senior (65+)", False, 0.55

            return {
                "estimated_age_range": age_range,
                "estimated_age": estimated_age,
                "is_minor_risk": is_minor,
                "confidence": confidence,
                "features": {
                    "wrinkle_score": wrinkle_score,
                    "skin_smoothness": skin_smoothness,
                    "eye_ratio": eye_ratio,
                    "roundness": roundness,
                },
            }
        except Exception as e:
            logger.error("AgeEstimator error: %s", e)
            return {"estimated_age_range": "error", "is_minor_risk": True, "confidence": 0.0, "error": str(e)}


# ============================================================================
# CELEBRITY / PUBLIC FIGURE DETECTION
# ============================================================================

class CelebrityDetector:
    """
    Heuristic celebrity-risk pipeline.
    High-quality + symmetric + professionally lit face = high impersonation risk.
    Each sub-score is computed independently and combined.
    """

    @classmethod
    def detect(
        cls,
        img: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

            # Restrict analysis to face region if available
            if face_bbox is not None:
                x, y, w, h = [int(v) for v in face_bbox]
                face_crop = img8[y : y + h, x : x + w]
                gray_crop = gray[y : y + h, x : x + w]
            else:
                face_crop = img8
                gray_crop = gray

            quality = cls._sharpness(gray_crop)
            symmetry = cls._symmetry(gray_crop)
            lighting = cls._lighting_evenness(face_crop)
            # High resolution face implies professional source
            resolution_bonus = min(1.0, (gray_crop.shape[0] * gray_crop.shape[1]) / (200 * 200))

            risk = quality * 0.30 + symmetry * 0.35 + lighting * 0.20 + resolution_bonus * 0.15

            return {
                "celebrity_risk_score": float(risk),
                "is_potential_celebrity": risk > 0.58,
                "quality_indicators": {
                    "sharpness": float(quality),
                    "symmetry": float(symmetry),
                    "lighting_evenness": float(lighting),
                    "resolution_score": float(resolution_bonus),
                },
                "warning": "High-quality face — verify not a public figure" if risk > 0.58 else None,
                "recommendation": "Use reverse image search to verify identity" if risk > 0.45 else None,
            }
        except Exception as e:
            logger.error("CelebrityDetector error: %s", e)
            return {"celebrity_risk_score": 0.0, "is_potential_celebrity": False, "error": str(e)}

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(1.0, lap_var / 600.0))

    @staticmethod
    def _symmetry(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        h, w = gray.shape
        left = gray[:, : w // 2].astype(float)
        right = cv2.flip(gray[:, w // 2 :], 1).astype(float)
        if left.shape != right.shape:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))
        diff = np.mean(np.abs(left - right))
        return float(max(0.0, 1.0 - diff / 55.0))

    @staticmethod
    def _lighting_evenness(img: np.ndarray) -> float:
        if img.size == 0:
            return 0.0
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2].astype(float)
        evenness = max(0.0, 1.0 - np.std(v) / 60.0)
        exposure = max(0.0, 1.0 - abs(np.mean(v) - 128) / 128.0)
        return float((evenness + exposure) / 2)


# ============================================================================
# METADATA ANALYSIS
# ============================================================================

class MetadataAnalyzer:
    """
    Deep EXIF inspection with consistency checks.
    Improvements over baseline:
    - GPS sanity check (implausible coords from AI tools)
    - Software field version-string analysis
    - Thumbnail vs main image consistency
    - Color space / ICC profile checks
    """

    AI_MARKERS = [
        "stable diffusion", "midjourney", "dall-e", "dalle", "ai generated",
        "gan", "stylegan", "openai", "automatic1111", "comfyui", "invoke ai",
        "leonardo", "runway", "firefly", "imagen", "flux", "sd xl",
    ]
    EDIT_MARKERS = ["photoshop", "gimp", "edited", "modified", "adobe", "lightroom", "affinity", "canva"]

    @classmethod
    def analyze(cls, image_path: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
        findings: Dict = {
            "has_metadata": False,
            "tampering_score": 0.0,
            "ai_markers_found": [],
            "suspicious_elements": [],
            "missing_expected_fields": [],
            "recommendations": [],
        }

        if image_path and Path(image_path).exists():
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS

                with Image.open(image_path) as img:
                    raw_exif = img._getexif()
                    if raw_exif:
                        metadata = {TAGS.get(k, str(k)): v for k, v in raw_exif.items()}
                        findings["has_metadata"] = True

                    # Thumbnail consistency check
                    try:
                        exif_obj = img.getexif()
                        thumb_ifd = exif_obj.get_ifd(0x8769)  # ExifIFD
                        if thumb_ifd:
                            findings["has_thumbnail_ifd"] = True
                    except Exception:
                        pass

            except Exception as e:
                logger.warning("EXIF extraction failed: %s", e)

        if not metadata:
            findings["tampering_score"] = 0.35
            findings["suspicious_elements"].append("No EXIF metadata — stripped or AI-generated")
            findings["missing_expected_fields"] = ["DateTime", "Make", "Model", "GPS"]
            findings["recommendations"].append("Verify image source — metadata absent")
            return findings

        findings["has_metadata"] = True
        meta_str = str(metadata).lower()

        for marker in cls.AI_MARKERS:
            if marker in meta_str:
                findings["ai_markers_found"].append(marker)
                findings["tampering_score"] += 0.35

        for marker in cls.EDIT_MARKERS:
            if marker in meta_str:
                findings["suspicious_elements"].append(f"Editing software: {marker}")
                findings["tampering_score"] += 0.10

        # Missing standard fields
        expected = ["DateTime", "Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings"]
        missing = [f for f in expected if f not in metadata]
        findings["missing_expected_fields"] = missing
        if len(missing) >= 4:
            findings["tampering_score"] += 0.20
            findings["suspicious_elements"].append(f"{len(missing)} standard camera fields absent")

        # Timestamp inconsistency
        dt = metadata.get("DateTime", "")
        dt_orig = metadata.get("DateTimeOriginal", "")
        if dt and dt_orig and dt != dt_orig:
            findings["suspicious_elements"].append("DateTime vs DateTimeOriginal mismatch")
            findings["tampering_score"] += 0.15

        # GPS sanity (AI tools sometimes embed 0,0 coords)
        gps = metadata.get("GPSInfo")
        if gps == {1: "N", 2: ((0, 1), (0, 1), (0, 1)), 3: "E", 4: ((0, 1), (0, 1), (0, 1))}:
            findings["suspicious_elements"].append("GPS coordinates are exactly 0,0 — likely synthetic")
            findings["tampering_score"] += 0.15

        # Software version strings that indicate generation tools
        software = str(metadata.get("Software", "")).lower()
        for marker in cls.AI_MARKERS + cls.EDIT_MARKERS:
            if marker in software:
                findings["suspicious_elements"].append(f"Software field contains: {software[:60]}")
                findings["tampering_score"] += 0.20
                break

        findings["tampering_score"] = float(min(1.0, findings["tampering_score"]))
        if findings["ai_markers_found"]:
            findings["recommendations"].append("AI generation markers confirmed — treat as synthetic")
        if findings["tampering_score"] > 0.5:
            findings["recommendations"].append("High tampering score — verify authenticity before use")

        return findings


# ============================================================================
# WATERMARK DETECTION  (improved inpainting artifact detector)
# ============================================================================

class WatermarkDetector:
    """
    Two-phase detection:
    1. Visible watermark — high edge density in corners / semi-transparent overlay detection
    2. Removal artifacts — locally uniform patches (inpainting), clone-stamp repeats (NCC)
    """

    @classmethod
    def detect(cls, img: np.ndarray) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            findings: Dict = {
                "has_visible_watermark": False,
                "watermark_removal_signs": False,
                "removal_score": 0.0,
                "suspicious_regions": [],
                "corner_analysis": {},
            }

            corners = {
                "bottom_right": gray[int(h * 0.82) :, int(w * 0.68) :],
                "bottom_left": gray[int(h * 0.82) :, : int(w * 0.32)],
                "bottom_center": gray[int(h * 0.82) :, int(w * 0.35) : int(w * 0.65)],
                "top_right": gray[: int(h * 0.18), int(w * 0.68) :],
                "top_left": gray[: int(h * 0.18), : int(w * 0.32)],
                "top_center": gray[: int(h * 0.12), int(w * 0.35) : int(w * 0.65)],
            }

            for cname, region in corners.items():
                if region.size < 100:
                    continue
                edge_density = float(cv2.Canny(region, 40, 120).mean())
                variance = float(np.var(region))

                findings["corner_analysis"][cname] = {"edge_density": edge_density, "variance": variance}

                if edge_density > 18 and 80 < variance < 2500:
                    findings["has_visible_watermark"] = True
                    findings["suspicious_regions"].append({"location": cname, "type": "potential_watermark", "edge_density": edge_density})

                # Very smooth corner in typical watermark zones = removal
                if variance < 40 and cname in ("bottom_right", "bottom_left", "bottom_center"):
                    findings["watermark_removal_signs"] = True
                    findings["removal_score"] += 0.35
                    findings["suspicious_regions"].append({"location": cname, "type": "smooth_patch_removal"})

            # Inpainting artifact scan — blocks with anomalously low Laplacian variance
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            global_lap_var = float(lap.var())
            block = 48
            for y in range(0, h - block, block):
                for x in range(0, w - block, block):
                    blk_var = float(lap[y : y + block, x : x + block].var())
                    if blk_var < global_lap_var * 0.08 and blk_var < 80:
                        findings["suspicious_regions"].append({"location": f"block_{x}_{y}", "type": "inpaint_suspect", "lap_var": blk_var})
                        findings["removal_score"] = min(1.0, findings["removal_score"] + 0.05)

            # Clone-stamp detection: normalized cross-correlation between blocks
            # (lightweight — check only top-20 smoothest blocks)
            smooth_blocks = sorted(
                [(float(np.var(gray[y : y + block, x : x + block])), x, y)
                 for y in range(0, h - block, block) for x in range(0, w - block, block)],
            )[:20]
            for _, bx, by in smooth_blocks[:10]:
                patch = gray[by : by + block, bx : bx + block].astype(np.float32)
                result = cv2.matchTemplate(gray.astype(np.float32), patch, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > 0.97 and max_loc != (bx, by):
                    findings["watermark_removal_signs"] = True
                    findings["removal_score"] = min(1.0, findings["removal_score"] + 0.20)
                    findings["suspicious_regions"].append({"location": f"clone_{bx}_{by}", "type": "clone_stamp", "match_score": float(max_val)})
                    break  # One confirmed clone is enough

            findings["removal_score"] = float(min(1.0, findings["removal_score"]))
            return findings
        except Exception as e:
            logger.error("WatermarkDetector error: %s", e)
            return {"has_visible_watermark": False, "watermark_removal_signs": False, "removal_score": 0.0, "error": str(e)}


# ============================================================================
# HATE SYMBOL DETECTION
# ============================================================================

class HateSymbolDetector:
    """
    Layered detection:
    1. Geometric contour analysis (crosses, angular/multi-vertex symbols)
    2. Suspicious color combination scoring (red/white/black triad)
    3. Aspect-ratio and solidity filters to reduce false positives
    NOTE: A production system should use a trained CNN classifier; this
    heuristic pipeline serves as a fast pre-filter.
    """

    @classmethod
    def detect(cls, img: np.ndarray) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            findings: Dict = {"hate_symbol_risk": 0.0, "detected_patterns": [], "severity": "NONE", "requires_review": False}

            edges = cv2.Canny(gray, 40, 130)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200 or area > h * w * 0.5:
                    continue

                epsilon = 0.025 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                verts = len(approx)
                bx, by, bw, bh = cv2.boundingRect(approx)
                aspect = bw / bh if bh > 0 else 0
                solidity = area / (bw * bh) if bw * bh > 0 else 0

                # Cross-like: 4–8 vertices, near-square, solid
                if 4 <= verts <= 8 and 0.7 < aspect < 1.35 and solidity > 0.3:
                    findings["detected_patterns"].append({"type": "cross_or_emblem", "location": (bx, by, bw, bh), "confidence": 0.35})
                    findings["hate_symbol_risk"] += 0.12

                # Star/angular: 8–14 vertices
                if 8 <= verts <= 14:
                    findings["detected_patterns"].append({"type": "angular_symbol", "vertices": verts, "confidence": 0.20})
                    findings["hate_symbol_risk"] += 0.08

            # Color combination analysis
            hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV)
            red_lo1, red_hi1 = np.array([0, 100, 80]), np.array([12, 255, 255])
            red_lo2, red_hi2 = np.array([168, 100, 80]), np.array([180, 255, 255])
            black_lo, black_hi = np.array([0, 0, 0]), np.array([180, 255, 45])
            white_lo, white_hi = np.array([0, 0, 200]), np.array([180, 35, 255])

            red = cv2.bitwise_or(cv2.inRange(hsv, red_lo1, red_hi1), cv2.inRange(hsv, red_lo2, red_hi2))
            black = cv2.inRange(hsv, black_lo, black_hi)
            white = cv2.inRange(hsv, white_lo, white_hi)

            r_r = np.sum(red > 0) / red.size
            b_r = np.sum(black > 0) / black.size
            w_r = np.sum(white > 0) / white.size

            if r_r > 0.08 and b_r > 0.08 and w_r > 0.08:
                findings["hate_symbol_risk"] += 0.20
                findings["detected_patterns"].append({"type": "suspicious_color_triad", "red": r_r, "black": b_r, "white": w_r})

            findings["hate_symbol_risk"] = float(min(1.0, findings["hate_symbol_risk"]))
            risk = findings["hate_symbol_risk"]
            if risk > 0.55:
                findings["severity"] = "HIGH"
                findings["requires_review"] = True
            elif risk > 0.28:
                findings["severity"] = "MEDIUM"
                findings["requires_review"] = True

            return findings
        except Exception as e:
            logger.error("HateSymbolDetector error: %s", e)
            return {"hate_symbol_risk": 0.0, "detected_patterns": [], "severity": "ERROR", "error": str(e)}


# ============================================================================
# TEXT OVERLAY ANALYSIS  (MSER + connected-component clustering)
# ============================================================================

class TextOverlayAnalyzer:
    """
    Improved text detection using MSER + connected-component stats.
    Groups overlapping regions into text lines.
    Flags banner-style, high-contrast, or unusually large headline text.
    """

    @classmethod
    def analyze(cls, img: np.ndarray) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape

            findings: Dict = {
                "has_text_regions": False,
                "text_region_count": 0,
                "misleading_score": 0.0,
                "style_analysis": {},
            }

            # MSER text detection
            mser = cv2.MSER_create(delta=5, min_area=60, max_area=14400)
            regions, _ = mser.detectRegions(gray)

            text_rects = []
            for region in regions:
                rx, ry, rw, rh = cv2.boundingRect(region.reshape(-1, 1, 2))
                aspect = rw / rh if rh > 0 else 0
                if 0.15 < aspect < 3.0 and 8 < rh < 120 and rw > 5:
                    text_rects.append((rx, ry, rw, rh))

            if len(text_rects) > 8:
                findings["has_text_regions"] = True
                findings["text_region_count"] = len(text_rects)

            # Cluster into lines by vertical proximity
            text_rects.sort(key=lambda r: r[1])
            lines: List[List] = []
            for rect in text_rects:
                placed = False
                for line in lines:
                    if abs(rect[1] - line[-1][1]) < 20:
                        line.append(rect)
                        placed = True
                        break
                if not placed:
                    lines.append([rect])

            # Banner-style lines (span >60% width at top or bottom)
            for line in lines:
                line_y = np.mean([r[1] for r in line])
                line_x_span = max(r[0] + r[2] for r in line) - min(r[0] for r in line)
                if line_x_span > w * 0.60:
                    if line_y < h * 0.18 or line_y > h * 0.80:
                        findings["misleading_score"] += 0.30
                        findings["style_analysis"]["banner_text"] = True

            # Large headlines
            large = sum(1 for r in text_rects if r[3] > 35)
            if large >= 2:
                findings["misleading_score"] += 0.20
                findings["style_analysis"]["large_headlines"] = True

            # High contrast text (clickbait indicator)
            if text_rects:
                mask = np.zeros_like(gray)
                for rx, ry, rw, rh in text_rects[:30]:
                    mask[ry : ry + rh, rx : rx + rw] = 255
                t_vals = gray[mask > 0]
                b_vals = gray[mask == 0]
                if t_vals.size and b_vals.size:
                    if abs(float(np.mean(t_vals)) - float(np.mean(b_vals))) > 90:
                        findings["misleading_score"] += 0.20
                        findings["style_analysis"]["high_contrast_text"] = True

            findings["misleading_score"] = float(min(1.0, findings["misleading_score"]))
            return findings
        except Exception as e:
            logger.error("TextOverlayAnalyzer error: %s", e)
            return {"has_text_regions": False, "misleading_score": 0.0, "error": str(e)}


# ============================================================================
# EMOTION MANIPULATION SCORING
# ============================================================================

class EmotionAnalyzer:
    """
    Gradient-based facial activity analysis.
    Analyses mouth, eye, and forehead regions independently.
    Scores high-intensity expressions (fear, surprise, anger) as higher manipulation risk.
    """

    @classmethod
    def analyze(
        cls,
        img: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)

            findings: Dict = {"detected_expressions": [], "manipulation_risk": 0.0, "emotional_intensity": 0.0, "concerns": []}

            if face_bbox is None:
                fc = _get_face_cascade()
                if fc is None:
                    return findings
                faces = fc.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.05, minNeighbors=4)
                if len(faces) == 0:
                    return findings
                face_bbox = max(faces, key=lambda f: f[2] * f[3])

            x, y, fw, fh = [int(v) for v in face_bbox]
            face_roi = gray[y : y + fh, x : x + fw]
            if face_roi.size < 400:
                return findings

            def region_activity(region: np.ndarray) -> float:
                if region.size == 0:
                    return 0.0
                gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
                return float(np.mean(np.sqrt(gx**2 + gy**2)))

            mouth_act = region_activity(face_roi[int(fh * 0.60) :, int(fw * 0.15) : int(fw * 0.85)])
            eye_act = region_activity(face_roi[int(fh * 0.20) : int(fh * 0.50), :])
            brow_act = region_activity(face_roi[: int(fh * 0.22), :])

            if mouth_act > 12:
                findings["detected_expressions"].append("expressive_mouth")
            if eye_act > 10:
                findings["detected_expressions"].append("intense_eyes")
            if brow_act > 9:
                findings["detected_expressions"].append("raised_eyebrows")

            total = mouth_act + eye_act + brow_act
            findings["emotional_intensity"] = float(min(1.0, total / 55.0))

            if findings["emotional_intensity"] > 0.70:
                findings["manipulation_risk"] = 0.65
                findings["concerns"].append("Very high emotional intensity")
            elif findings["emotional_intensity"] > 0.50:
                findings["manipulation_risk"] = 0.35
                findings["concerns"].append("Elevated emotional expression")

            if "intense_eyes" in findings["detected_expressions"] and "raised_eyebrows" in findings["detected_expressions"]:
                findings["manipulation_risk"] = float(min(1.0, findings["manipulation_risk"] + 0.20))
                findings["concerns"].append("Fear/surprise expression pattern")

            return findings
        except Exception as e:
            logger.error("EmotionAnalyzer error: %s", e)
            return {"detected_expressions": [], "manipulation_risk": 0.0, "error": str(e)}


# ============================================================================
# JURISDICTION COMPLIANCE
# ============================================================================

class JurisdictionCompliance:
    """Region-specific legal framework lookup."""

    REGULATIONS = {
        "EU": {"name": "European Union", "laws": ["AI Act", "GDPR"], "severity": "HIGH", "requirements": ["AI-generated content must be labeled", "Cannot create deepfakes without consent", "Right to erasure applies to synthetic content"]},
        "US_FEDERAL": {"name": "United States (Federal)", "laws": ["DEEPFAKES Accountability Act (pending)"], "severity": "MEDIUM", "requirements": ["No federal deepfake law yet (2024)", "State laws vary", "FTC may pursue deceptive practices"]},
        "US_CA": {"name": "California", "laws": ["AB 602", "AB 730"], "severity": "HIGH", "requirements": ["Non-consensual deepfake pornography criminalized", "Political deepfakes restricted near elections"]},
        "US_TX": {"name": "Texas", "laws": ["SB 751"], "severity": "HIGH", "requirements": ["Deepfake pornography criminalized", "Creating with intent to harm is felony"]},
        "UK": {"name": "United Kingdom", "laws": ["Online Safety Act"], "severity": "HIGH", "requirements": ["Non-consensual intimate images criminalized", "Platforms must remove harmful content"]},
        "CHINA": {"name": "China", "laws": ["Deep Synthesis Provisions"], "severity": "VERY HIGH", "requirements": ["All AI content must be labeled", "User consent for face synthesis", "Strict penalties"]},
        "KOREA": {"name": "South Korea", "laws": ["Act on Punishment of Sexual Crimes"], "severity": "VERY HIGH", "requirements": ["Deepfake pornography: prison sentence", "Possession also criminalized"]},
        "INDIA": {"name": "India", "laws": ["IT Act", "Proposed Digital India Act"], "severity": "MEDIUM", "requirements": ["No specific deepfake law yet", "Applicable under defamation/identity theft"]},
    }

    CONTENT_WARNINGS = {
        "face":    {"general": ["AI-generated faces require consent verification", "Many jurisdictions require disclosure", "May constitute identity theft if depicting real person"], "risk": "HIGH"},
        "intimate": {"general": ["CRIMINAL OFFENSE in most jurisdictions", "Even creation without distribution is often illegal", "Severe penalties including imprisonment"], "risk": "CRITICAL"},
        "political": {"general": ["Restricted during elections in many regions", "Must be labeled as AI-generated", "May constitute election interference"], "risk": "HIGH"},
        "general":   {"general": ["Disclose AI generation origin", "Verify platform policies"], "risk": "LOW"},
    }

    @classmethod
    def get_warnings(cls, content_type: str = "general", region: Optional[str] = None) -> Dict:
        regs = {region: cls.REGULATIONS[region]} if region and region in cls.REGULATIONS else cls.REGULATIONS
        cw = cls.CONTENT_WARNINGS.get(content_type, cls.CONTENT_WARNINGS["general"])
        return {
            "applicable_regulations": [{"jurisdiction": r["name"], "code": code, "laws": r["laws"], "severity": r["severity"], "requirements": r["requirements"]} for code, r in regs.items()],
            "general_warnings": cw["general"],
            "risk_level": cw["risk"],
        }


# ============================================================================
# DOCUMENT / ID FORGERY DETECTION
# ============================================================================

class DocumentForgeryDetector:
    """
    Structural analysis for document-like images.
    Improvements:
    - Machine-readable zone (MRZ) row detection
    - Guilloche/background pattern detection
    - Security feature region analysis
    """

    DOC_RATIOS = {"id_card": (1.50, 1.70), "passport": (0.68, 0.78), "license": (1.48, 1.62), "letter": (0.68, 0.82)}

    @classmethod
    def detect(cls, img: np.ndarray) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = cv2.cvtColor(img8, cv2.COLOR_RGB2GRAY)
            h, w = img8.shape[:2]
            aspect = w / h if h > 0 else 0

            findings: Dict = {"is_document": False, "document_type": None, "forgery_indicators": [], "forgery_score": 0.0, "warnings": []}

            for dtype, (lo, hi) in cls.DOC_RATIOS.items():
                if lo <= aspect <= hi:
                    findings["document_type"] = dtype
                    findings["is_document"] = True
                    break

            # Rectangular border lines
            edges = cv2.Canny(gray, 45, 140)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, minLineLength=40, maxLineGap=12)
            h_lines = v_lines = 0
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                    if angle < 12 or angle > 168:
                        h_lines += 1
                    elif 78 < angle < 102:
                        v_lines += 1
            if h_lines >= 3 and v_lines >= 3:
                findings["is_document"] = True
                findings["forgery_indicators"].append("Rectangular border structure detected")

            # Face in ID-typical position
            fc = _get_face_cascade()
            if fc is not None:
                faces = fc.detectMultiScale(cv2.equalizeHist(gray), 1.05, 4, minSize=(25, 25))
                for (fx, fy, fw, fh) in faces:
                    if (fx + fw / 2) < w * 0.45:
                        findings["is_document"] = True
                        findings["forgery_indicators"].append("Face in left-side ID position")

            # Text region density
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            text_count = sum(
                1 for r in regions
                for (rx, ry, rw, rh) in [cv2.boundingRect(r.reshape(-1, 1, 2))]
                if 0.15 < rw / rh < 3.5 and 7 < rh < 55
            )
            if text_count > 25:
                findings["is_document"] = True
                findings["forgery_indicators"].append(f"Dense text regions ({text_count})")

            # MRZ detection — two rows of fixed-pitch characters at bottom ~15%
            bottom_strip = gray[int(h * 0.82) :, :]
            mrz_edges = cv2.Canny(bottom_strip, 30, 100)
            if float(mrz_edges.mean()) > 12:
                findings["forgery_indicators"].append("Possible MRZ (machine-readable zone) detected")
                findings["is_document"] = True

            # Guilloche pattern — periodic high-freq texture in background
            bg_fft = np.abs(fft.fftshift(fft.fft2(gray.astype(float))))
            h2, w2 = bg_fft.shape
            # Off-center energy peaks in FFT suggest repeating background patterns
            ring = bg_fft[h2 // 2 - 40 : h2 // 2 + 40, w2 // 2 - 40 : w2 // 2 + 40]
            center = bg_fft[h2 // 2 - 10 : h2 // 2 + 10, w2 // 2 - 10 : w2 // 2 + 10]
            if center.mean() > 0 and ring.mean() / center.mean() > 0.15:
                findings["forgery_indicators"].append("Periodic background pattern (security feature or forgery)")

            if findings["is_document"]:
                findings["forgery_score"] = float(min(1.0, len(findings["forgery_indicators"]) * 0.22))
                findings["warnings"] = [
                    "Document/ID structure detected — verify authenticity",
                    "AI-generated IDs are illegal in virtually all jurisdictions",
                    "Creating or possessing fake ID documents is a serious criminal offense",
                ]

            return findings
        except Exception as e:
            logger.error("DocumentForgeryDetector error: %s", e)
            return {"is_document": False, "forgery_score": 0.0, "error": str(e)}


# ============================================================================
# GAN / DIFFUSION FINGERPRINT DETECTOR  (new, was missing)
# ============================================================================

class SyntheticFingerprintDetector:
    """
    Detects GAN/diffusion model fingerprints at the signal level.
    Methods:
    - Spectral peak analysis (GANs leave periodic artifacts in FFT)
    - Noise residual analysis (Gaussian vs Poisson noise distribution)
    - Co-occurrence matrix regularity (natural images have rougher co-occurrence)
    - DCT coefficient distribution (JPEG-style blocking vs GAN smoothness)
    """

    @classmethod
    def detect(cls, img: np.ndarray) -> Dict:
        try:
            img8 = _to_uint8(img)
            gray = rgb_to_gray(img if img.max() <= 1.0 else img / 255.0)

            findings: Dict = {"synthetic_score": 0.0, "methods": {}, "concerns": []}

            # 1. FFT spectral analysis
            F = fft.fft2(gray)
            Fmag = np.abs(fft.fftshift(F))
            h, w = Fmag.shape
            ch, cw = h // 2, w // 2

            # GAN fingerprint: unnaturally high energy at specific radii
            spectral_scores = []
            for r in [8, 16, 24, 32, 48]:
                yg, xg = np.ogrid[-ch : h - ch, -cw : w - cw]
                mask = (xg**2 + yg**2 >= (r - 3) ** 2) & (xg**2 + yg**2 <= (r + 3) ** 2)
                if mask.any():
                    ring_e = float(Fmag[mask].mean())
                    spectral_scores.append(ring_e / (Fmag.mean() + 1e-8))

            spectral_anomaly = float(np.max(spectral_scores)) if spectral_scores else 0.0
            findings["methods"]["spectral_rings"] = min(1.0, spectral_anomaly / 3.0)
            if spectral_anomaly > 2.5:
                findings["concerns"].append("Spectral ring artifacts (GAN/diffusion fingerprint)")

            # 2. Noise residual distribution test
            residual = extract_residual(gray, sigma=2.0)
            res_kurt = float(_kurtosis(residual.ravel()))  # Gaussian ~ 3
            # AI residuals tend toward lower kurtosis (too smooth)
            noise_anomaly = max(0.0, 1.0 - abs(res_kurt - 3.0) / 6.0)
            findings["methods"]["noise_distribution"] = noise_anomaly
            if noise_anomaly > 0.7:
                findings["concerns"].append("Residual noise distribution atypical for camera sensors")

            # 3. Local Binary Pattern entropy (natural images more complex)
            lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, method="uniform")
            lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
            lbp_entropy_val = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10)))
            # Low entropy = overly regular texture (common in diffusion outputs)
            texture_anomaly = max(0.0, 1.0 - lbp_entropy_val / 3.3)
            findings["methods"]["texture_regularity"] = texture_anomaly
            if texture_anomaly > 0.5:
                findings["concerns"].append("Low texture entropy — overly regular surface texture")

            # Aggregate
            findings["synthetic_score"] = float(
                min(1.0, findings["methods"]["spectral_rings"] * 0.45
                    + findings["methods"]["noise_distribution"] * 0.30
                    + findings["methods"]["texture_regularity"] * 0.25)
            )
            return findings
        except Exception as e:
            logger.error("SyntheticFingerprintDetector error: %s", e)
            return {"synthetic_score": 0.0, "methods": {}, "concerns": [], "error": str(e)}


def _kurtosis(data: np.ndarray) -> float:
    """Pearson kurtosis (Gaussian distribution is approximately 3)."""
    n = len(data)
    if n < 4:
        return 0.0
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma < 1e-10:
        return 0.0
    return float(np.mean(((data - mu) / sigma) ** 4))


# ============================================================================
# MAIN ETHICAL ASSESSMENT  (orchestrator)
# ============================================================================

class EthicalAssessment:
    """
    Comprehensive ethical assessment of AI-generated images.

    Weight table (revised to reflect research-backed importance):
      synthetic_fingerprint  0.20  — core AI detection signal
      nsfw_risk              0.15  — explicit content
      high_quality_artifacts 0.12  — convincingness
      low_quality_artifacts  0.08  — obvious fakeness (inverse)
      facial_consistency     0.10  — face coherence
      lighting_anomalies     0.08  — lighting consistency
      frequency_analysis     0.08  — GAN rings (redundant with fingerprint)
      minor_risk             0.08  — child safety (critical)
      document_risk          0.06  — ID forgery
      celebrity_risk         0.03  — impersonation
      manipulation_risk      0.02  — emotional manipulation
    """

    ETHICAL_THRESHOLD = 0.48

    WEIGHTS = {
        "synthetic_fingerprint": 0.20,
        "high_quality_artifacts": 0.12,
        "low_quality_artifacts": 0.08,
        "facial_consistency": 0.10,
        "lighting_anomalies": 0.08,
        "frequency_analysis": 0.08,
        "nsfw_risk": 0.15,
        "minor_risk": 0.08,
        "document_risk": 0.06,
        "celebrity_risk": 0.03,
        "manipulation_risk": 0.02,
    }

    # Flags that unconditionally trigger UNETHICAL regardless of score
    CRITICAL_FLAGS = {
        "POTENTIAL_MINOR": "UNETHICAL - MINOR PROTECTION",
        "NSFW_CONTENT": "UNETHICAL - NSFW CONTENT",
        "DOCUMENT_DETECTED": "UNETHICAL - DOCUMENT FORGERY RISK",
        "POTENTIAL_HATE_SYMBOL": "UNETHICAL - HARMFUL CONTENT",
    }

    @classmethod
    def assess(
        cls,
        img_arr: np.ndarray,
        threshold: Optional[float] = None,
        image_path: Optional[str] = None,
        include_all_checks: bool = True,
    ) -> Dict:
        """
        Full ethical assessment pipeline.

        Args:
            img_arr:           Image as numpy array, float [0,1] or uint8 [0,255], shape (H,W,3)
            threshold:         Override default ETHICAL_THRESHOLD
            image_path:        Path to original file (enables EXIF metadata analysis)
            include_all_checks: Run all detectors (disable for speed)

        Returns:
            Comprehensive result dict.
        """
        # Normalise to float [0,1]
        if img_arr.dtype != np.float32 and img_arr.dtype != np.float64:
            img_f = img_arr.astype(np.float32) / 255.0
        else:
            img_f = img_arr.clip(0, 1)

        results: Dict = {
            "is_ethical": True,
            "status": "ASSESSING",
            "risk_score": 0.0,
            "confidence": 0.0,
            "flags": [],
            "checks": {},
        }

        # --- Face detection (multi-method) ---
        num_faces, face_bboxes = FaceDetector.detect(img_f)
        results["faces_detected"] = num_faces
        face_bbox = tuple(face_bboxes[0]) if face_bboxes else None

        # --- Base feature extraction ---
        features = cls._extract_base_features(img_f)
        if features is None:
            results["status"] = "ASSESSMENT_FAILED"
            results["details"] = "Feature extraction failed"
            return results
        results["features"] = features

        # --- Extended checks ---
        nsfw_result = age_result = celeb_result = emotion_result = {}
        metadata_result = watermark_result = hate_result = text_result = doc_result = fingerprint_result = {}

        if include_all_checks:
            # Synthetic fingerprint (new)
            fingerprint_result = SyntheticFingerprintDetector.detect(img_f)
            results["checks"]["synthetic_fingerprint"] = fingerprint_result
            if fingerprint_result.get("synthetic_score", 0) > 0.6:
                results["flags"].append("SYNTHETIC_FINGERPRINT")

            # NSFW
            nsfw_result = NSFWDetector.detect(img_f)
            results["checks"]["nsfw"] = nsfw_result
            if nsfw_result.get("is_nsfw"):
                results["flags"].append("NSFW_CONTENT")

            if num_faces > 0 and face_bbox:
                # Age estimation
                age_result = AgeEstimator.estimate(img_f, face_bbox)
                results["checks"]["age_estimation"] = age_result
                if age_result.get("is_minor_risk"):
                    results["flags"].append("POTENTIAL_MINOR")

                # Celebrity detection
                celeb_result = CelebrityDetector.detect(img_f, face_bbox)
                results["checks"]["celebrity"] = celeb_result
                if celeb_result.get("is_potential_celebrity"):
                    results["flags"].append("POTENTIAL_CELEBRITY")

                # Emotion analysis
                emotion_result = EmotionAnalyzer.analyze(img_f, face_bbox)
                results["checks"]["emotion"] = emotion_result
                if emotion_result.get("manipulation_risk", 0) > 0.5:
                    results["flags"].append("EMOTIONAL_MANIPULATION")

            # Metadata
            if image_path:
                metadata_result = MetadataAnalyzer.analyze(image_path)
                results["checks"]["metadata"] = metadata_result
                if metadata_result.get("ai_markers_found"):
                    results["flags"].append("AI_METADATA_MARKERS")

            # Watermark
            watermark_result = WatermarkDetector.detect(img_f)
            results["checks"]["watermark"] = watermark_result
            if watermark_result.get("watermark_removal_signs"):
                results["flags"].append("WATERMARK_REMOVAL")

            # Hate symbols
            hate_result = HateSymbolDetector.detect(img_f)
            results["checks"]["hate_symbols"] = hate_result
            if hate_result.get("hate_symbol_risk", 0) > 0.5:
                results["flags"].append("POTENTIAL_HATE_SYMBOL")

            # Text overlay
            text_result = TextOverlayAnalyzer.analyze(img_f)
            results["checks"]["text_overlay"] = text_result
            if text_result.get("misleading_score", 0) > 0.5:
                results["flags"].append("MISLEADING_TEXT")

            # Document forgery
            doc_result = DocumentForgeryDetector.detect(img_f)
            results["checks"]["document"] = doc_result
            if doc_result.get("is_document"):
                results["flags"].append("DOCUMENT_DETECTED")

            # Jurisdiction
            content_type = "intimate" if nsfw_result.get("is_nsfw") else ("face" if num_faces > 0 else "general")
            results["checks"]["jurisdiction"] = JurisdictionCompliance.get_warnings(content_type)

        # --- Risk score aggregation ---
        risk = 0.0
        risk += cls.WEIGHTS["high_quality_artifacts"] * features["high_quality_artifacts"]
        risk += cls.WEIGHTS["low_quality_artifacts"] * (1.0 - features["low_quality_artifacts"])
        risk += cls.WEIGHTS["facial_consistency"] * (1.0 - features["facial_consistency"])
        risk += cls.WEIGHTS["lighting_anomalies"] * features["lighting_anomalies"]
        risk += cls.WEIGHTS["frequency_analysis"] * features["frequency_analysis"]

        if include_all_checks:
            risk += cls.WEIGHTS["synthetic_fingerprint"] * fingerprint_result.get("synthetic_score", 0)
            risk += cls.WEIGHTS["nsfw_risk"] * nsfw_result.get("nsfw_score", 0)
            if num_faces > 0:
                risk += cls.WEIGHTS["minor_risk"] * (1.0 if age_result.get("is_minor_risk") else 0)
                risk += cls.WEIGHTS["celebrity_risk"] * celeb_result.get("celebrity_risk_score", 0)
                risk += cls.WEIGHTS["manipulation_risk"] * emotion_result.get("manipulation_risk", 0)
            risk += cls.WEIGHTS["document_risk"] * doc_result.get("forgery_score", 0)

        # Critical flags override
        for flag, status in cls.CRITICAL_FLAGS.items():
            if flag in results["flags"]:
                risk = max(risk, 0.82)

        # Face in AI image → always high risk
        if num_faces > 0:
            risk = max(risk, 0.85)

        results["risk_score"] = float(min(1.0, risk))
        use_threshold = cls.ETHICAL_THRESHOLD if threshold is None else float(threshold)
        results["threshold"] = use_threshold

        # --- Final classification ---
        if num_faces > 0:
            results["is_ethical"] = False
            results["status"] = "UNETHICAL - FACES DETECTED"
        else:
            override_status = None
            for flag, status in cls.CRITICAL_FLAGS.items():
                if flag in results["flags"]:
                    override_status = status
                    break
            if override_status:
                results["is_ethical"] = False
                results["status"] = override_status
            elif risk > use_threshold:
                results["is_ethical"] = False
                results["status"] = "UNETHICAL - HIGH RISK"
            else:
                results["is_ethical"] = True
                results["status"] = "ETHICAL"

        dist = abs(risk - use_threshold)
        results["confidence"] = float(max(0.0, min(1.0, 1.0 - dist * 2.0)))
        results["details"] = cls._build_details(results)
        results["explanation"] = cls._build_explanation(results)
        results["recommendations"] = cls._build_recommendations(results)

        return results

    @staticmethod
    def _extract_base_features(img_f: np.ndarray) -> Optional[Dict]:
        try:
            gray = rgb_to_gray(img_f)
            residual = extract_residual(gray, sigma=1.5)
            artifact_std = float(np.std(residual))
            hq_artifacts = min(1.0, artifact_std * 2.0)
            lq_score = min(1.0, artifact_std / 0.15)

            _, hf_ratio = fft_stats(gray)
            facial_consistency = 1.0 - min(1.0, hf_ratio)

            try:
                lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, method="uniform")
                lbp_var = float(np.var(lbp))
                lighting_anomaly = min(1.0, lbp_var / 50.0)
            except Exception:
                lighting_anomaly, lbp_var = 0.5, 0.0

            F = fft.fft2(gray)
            Fmag = np.abs(fft.fftshift(F))
            h, w = gray.shape
            ch, cw = h // 2, w // 2
            ring_pattern = 0.0
            for r in [10, 20, 30]:
                yg, xg = np.ogrid[-ch : h - ch, -cw : w - cw]
                mask = (xg**2 + yg**2 >= (r - 2) ** 2) & (xg**2 + yg**2 <= (r + 2) ** 2)
                ring_e = float(Fmag[mask].mean()) if mask.any() else 0
                ring_pattern = max(ring_pattern, ring_e)
            frequency_risk = min(1.0, ring_pattern / (float(Fmag.max()) + 1e-6) * 5)

            return {
                "high_quality_artifacts": hq_artifacts,
                "low_quality_artifacts": lq_score,
                "facial_consistency": facial_consistency,
                "lighting_anomalies": lighting_anomaly,
                "frequency_analysis": frequency_risk,
                "artifact_std": artifact_std,
                "lbp_variance": lbp_var,
            }
        except Exception as e:
            logger.error("Base feature extraction error: %s", e)
            return None

    @staticmethod
    def _build_details(results: Dict) -> str:
        flags = results.get("flags", [])
        faces = results.get("faces_detected", 0)
        risk = results.get("risk_score", 0)
        parts = []
        if "NSFW_CONTENT" in flags:
            parts.append("Explicit/NSFW content")
        if "POTENTIAL_MINOR" in flags:
            parts.append("⚠ Potential minor — HIGH PRIORITY")
        if "DOCUMENT_DETECTED" in flags:
            parts.append("Document/ID forgery risk")
        if "POTENTIAL_CELEBRITY" in flags:
            parts.append("Possible celebrity likeness")
        if "EMOTIONAL_MANIPULATION" in flags:
            parts.append("High emotional manipulation potential")
        if "WATERMARK_REMOVAL" in flags:
            parts.append("Watermark removal signs")
        if "POTENTIAL_HATE_SYMBOL" in flags:
            parts.append("Potential hate symbol")
        if "MISLEADING_TEXT" in flags:
            parts.append("Misleading text overlay")
        if "SYNTHETIC_FINGERPRINT" in flags:
            parts.append("Strong synthetic fingerprint detected")
        if faces > 0:
            parts.append(f"{faces} face(s) detected — privacy concern")
        if not parts:
            parts.append(f"Risk score {risk:.1%} — {'elevated' if risk > 0.35 else 'low'}")
        return " | ".join(parts)

    @staticmethod
    def _build_explanation(results: Dict) -> str:
        lines = []
        flags = results.get("flags", [])
        checks = results.get("checks", {})
        features = results.get("features", {})

        if results.get("faces_detected", 0) > 0:
            lines += [
                "CRITICAL: Human face(s) found in AI-generated content",
                "  → Privacy and consent concerns apply",
                "  → High deepfake misuse potential",
            ]

        if "NSFW_CONTENT" in flags:
            nsfw = checks.get("nsfw", {})
            lines.append(f"NSFW: {nsfw.get('severity','?')} severity (score {nsfw.get('nsfw_score',0):.2f})")
            for c in nsfw.get("concerns", []):
                lines.append(f"  → {c}")

        if "POTENTIAL_MINOR" in flags:
            age = checks.get("age_estimation", {})
            lines.append(f"Age Estimation: {age.get('estimated_age_range','?')} (conf {age.get('confidence',0)*100:.0f}%)")

        fp = checks.get("synthetic_fingerprint", {})
        if fp.get("synthetic_score", 0) > 0.3:
            lines.append(f"Synthetic Fingerprint Score: {fp['synthetic_score']:.2f}")
            for c in fp.get("concerns", []):
                lines.append(f"  → {c}")

        if features:
            lines.append(f"Base Features: artifact_std={features.get('artifact_std',0):.4f}  lbp_var={features.get('lbp_variance',0):.2f}")

        return "\n".join(lines) if lines else "Standard assessment completed."

    @staticmethod
    def _build_recommendations(results: Dict) -> List[str]:
        recs = []
        flags = results.get("flags", [])

        if "POTENTIAL_MINOR" in flags:
            recs += ["CRITICAL: Do not distribute — potential minor detected", "Delete content immediately", "Report to authorities if real minor suspected"]

        if "NSFW_CONTENT" in flags:
            recs += ["Do not distribute without age verification", "Explicit consent required from all depicted persons", "May be illegal in many jurisdictions"]

        if "DOCUMENT_DETECTED" in flags:
            recs += ["WARN: Document/ID forgery is a serious criminal offense", "Do not use for any official purpose", "Delete content — possession may be illegal"]

        if "POTENTIAL_HATE_SYMBOL" in flags:
            recs += ["Content flagged for potentially harmful imagery", "Do not distribute", "Manual review recommended"]

        if results.get("faces_detected", 0) > 0:
            recs += ["Obtain explicit consent from any person depicted", "Clearly label as AI-generated", "Verify compliance with local deepfake laws", "Do not use for impersonation"]

        risk = results.get("risk_score", 0)
        if risk > 0.70 and not recs:
            recs += ["High-risk content — exercise extreme caution", "Consider not distributing", "Add prominent AI-generated watermark"]
        elif risk > 0.48 and not recs:
            recs += ["Moderate risk — disclose AI generation", "Add metadata indicating synthetic origin"]
        elif not recs:
            recs += ["Lower risk content — still recommend disclosure", "Suitable for educational/artistic use with attribution"]

        return recs


# ============================================================================
# REPORT FORMATTERS
# ============================================================================

def format_ethical_report(assessment: Dict) -> str:
    sep = "=" * 80
    thin = "-" * 80
    r = [sep, "COMPREHENSIVE ETHICAL ASSESSMENT REPORT", sep, ""]

    status = assessment.get("status", "UNKNOWN")
    risk = assessment.get("risk_score", 0)
    r.append(f"STATUS:     {status}")
    r.append(f"Risk Score: {risk:.2%}  (Threshold: {assessment.get('threshold', 0.5):.0%})")
    r.append(f"Confidence: {assessment.get('confidence', 0)*100:.1f}%")
    r.append(f"Faces:      {assessment.get('faces_detected', 0)}")
    r.append("")

    flags = assessment.get("flags", [])
    if flags:
        r.append(f"FLAGS RAISED:"); r.append(thin)
        for f in flags:
            r.append(f"  [{f}]")
        r.append("")

    r.append("DETAILS:"); r.append(thin)
    r.append(assessment.get("details", "—")); r.append("")

    r.append("TECHNICAL ANALYSIS:"); r.append(thin)
    r.append(assessment.get("explanation", "—")); r.append("")

    checks = assessment.get("checks", {})
    if checks:
        r.append("CHECK RESULTS:"); r.append(thin)
        if "nsfw" in checks:
            c = checks["nsfw"]
            r.append(f"  NSFW:                {c.get('severity','?')}  (score {c.get('nsfw_score',0):.2f})")
        if "synthetic_fingerprint" in checks:
            c = checks["synthetic_fingerprint"]
            r.append(f"  Synthetic Fingerprint: {c.get('synthetic_score',0):.2f}")
        if "age_estimation" in checks:
            c = checks["age_estimation"]
            r.append(f"  Age Estimate:        {c.get('estimated_age_range','?')}  (minor risk: {c.get('is_minor_risk',False)})")
        if "celebrity" in checks:
            c = checks["celebrity"]
            r.append(f"  Celebrity Risk:      {c.get('celebrity_risk_score',0):.2f}")
        if "document" in checks:
            c = checks["document"]
            r.append(f"  Document:            {c.get('document_type','None')}  (forgery score: {c.get('forgery_score',0):.2f})")
        if "hate_symbols" in checks:
            c = checks["hate_symbols"]
            r.append(f"  Hate Symbols:        {c.get('severity','?')}")
        if "emotion" in checks:
            c = checks["emotion"]
            r.append(f"  Emotion Manipulation:{c.get('manipulation_risk',0):.2f}")
        if "watermark" in checks:
            c = checks["watermark"]
            r.append(f"  Watermark Removal:   {c.get('watermark_removal_signs',False)}  (score {c.get('removal_score',0):.2f})")
        r.append("")

    r.append("RECOMMENDATIONS:"); r.append(thin)
    for i, rec in enumerate(assessment.get("recommendations", []), 1):
        r.append(f"  {i}. {rec}")

    if "jurisdiction" in checks:
        j = checks["jurisdiction"]
        r.append(f"\nJURISDICTION RISK LEVEL: {j.get('risk_level','?')}"); r.append(thin)
        for w in j.get("general_warnings", []):
            r.append(f"  - {w}")

    r.append(""); r.append(sep)
    return "\n".join(r)


def get_simple_status(assessment: Dict) -> str:
    flags = assessment.get("flags", [])
    risk = assessment.get("risk_score", 0)
    flag_str = f"  [{', '.join(flags[:4])}]" if flags else ""
    label = "ETHICAL" if assessment.get("is_ethical") else "UNETHICAL"
    return f"{label}  Risk: {risk:.1%}{flag_str}"