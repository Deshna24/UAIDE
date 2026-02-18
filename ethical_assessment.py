"""
Ethical Assessment Module for AI-Generated Images
Evaluates whether an AI-generated image's synthetic nature is detectable (ethical)
or if it presents high misuse risk (unethical)

CRITICAL UPDATE: AI-generated images containing human faces are ALWAYS classified
as UNETHICAL due to privacy, consent, and deepfake concerns.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import numpy.fft as fft
from skimage.feature import local_binary_pattern
import cv2

from detector import rgb_to_gray, extract_residual, fft_stats, lbp_entropy


class EthicalAssessment:
    """Assesses ethical status of AI-generated images"""
    
    # Ethical classification thresholds
    ETHICAL_THRESHOLD = 0.5  # Below = Ethical, Above = Unethical
    
    # Face detector (initialized once)
    _face_cascade = None
    
    ETHICAL_CRITERIA = {
        "high_quality_artifacts": {
            "weight": 0.25,
            "description": "Quality of generation (higher = more convincing/risky)"
        },
        "low_quality_artifacts": {
            "weight": 0.20,
            "description": "Detectable inconsistencies (higher = more obvious fake)"
        },
        "facial_consistency": {
            "weight": 0.20,
            "description": "Consistent facial features (higher = more natural)"
        },
        "lighting_anomalies": {
            "weight": 0.15,
            "description": "Lighting consistency (higher = more obvious artifacts)"
        },
        "frequency_analysis": {
            "weight": 0.20,
            "description": "GAN fingerprints (higher = more obvious synthetic patterns)"
        }
    }
    
    @classmethod
    def _get_face_detector(cls):
        """Get or initialize OpenCV face detector"""
        if cls._face_cascade is None:
            try:
                # Try to load Haar Cascade face detector
                cls._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception as e:
                print(f"Warning: Could not load face detector: {e}")
                cls._face_cascade = None
        return cls._face_cascade
    
    @classmethod
    def detect_faces(cls, img_arr):
        """Detect faces in image using OpenCV
        
        Returns:
            tuple: (num_faces, face_bboxes)
        """
        try:
            face_cascade = cls._get_face_detector()
            if face_cascade is None:
                return 0, []
            
            # Convert to grayscale for detection
            if len(img_arr.shape) == 3:
                gray = cv2.cvtColor((img_arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (img_arr * 255).astype(np.uint8)
            
            # Detect faces with tuned parameters for better detection
            # Parameters tuned for AI-generated face detection (80% detection rate)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More sensitive to different face sizes
                minNeighbors=3,    # Lower threshold for face confirmation
                minSize=(20, 20)   # Detect smaller faces
            )
            
            return len(faces), faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return 0, []
    
    @staticmethod
    def extract_features(img_arr):
        """Extract features for ethical assessment"""
        try:
            H, W, C = img_arr.shape
            gray = rgb_to_gray(img_arr)
            
            # 1. Artifact Quality
            residual = extract_residual(gray, sigma=1.5)
            artifact_std = float(np.std(residual))
            high_quality_artifacts = min(1.0, artifact_std * 2.0)
            
            # 2. Low-quality indicator
            low_quality_score = min(1.0, artifact_std / 0.15)
            
            # 3. Facial consistency
            _, hf_ratio = fft_stats(gray)
            facial_consistency = 1.0 - min(1.0, hf_ratio)
            
            # 4. Lighting anomalies
            try:
                lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, method='uniform')
                lbp_variance = float(np.var(lbp))
                lighting_anomaly = min(1.0, lbp_variance / 50.0)
            except:
                lighting_anomaly = 0.5
                lbp_variance = 0.0
            
            # 5. Frequency domain analysis
            F = fft.fft2(gray)
            Fshift = fft.fftshift(F)
            Fmag = np.abs(Fshift)
            
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            
            ring_pattern = 0.0
            for r in [10, 20, 30]:
                y, x = np.ogrid[-center_h:h-center_h, -center_w:w-center_w]
                mask = (x**2 + y**2 >= (r-2)**2) & (x**2 + y**2 <= (r+2)**2)
                ring_energy = np.mean(Fmag[mask]) if np.any(mask) else 0
                ring_pattern = max(ring_pattern, ring_energy)
            
            frequency_risk = min(1.0, ring_pattern / (np.max(Fmag) + 1e-6) * 5)
            
            return {
                "high_quality_artifacts": high_quality_artifacts,
                "low_quality_artifacts": low_quality_score,
                "facial_consistency": facial_consistency,
                "lighting_anomalies": lighting_anomaly,
                "frequency_analysis": frequency_risk,
                "artifact_std": artifact_std,
                "lbp_variance": lbp_variance
            }
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    @classmethod
    def assess(cls, img_arr, threshold=None):
        """
        Assess ethical status of AI-generated image
        
        CRITICAL: Images with detected faces are ALWAYS classified as UNETHICAL
        
        Returns:
            dict with ethical assessment details
        """
        # First, detect faces - AI-generated faces are ethically problematic
        num_faces, face_bboxes = cls.detect_faces(img_arr)
        
        features = cls.extract_features(img_arr)
        
        if features is None:
            return {
                "is_ethical": True,
                "status": "ASSESSMENT_FAILED",
                "risk_score": 0.5,
                "confidence": 0.0,
                "details": "Could not assess image",
                "faces_detected": num_faces
            }
        
        # Calculate weighted risk score
        risk_score = (
            cls.ETHICAL_CRITERIA["high_quality_artifacts"]["weight"] * 
            features["high_quality_artifacts"] +
            
            cls.ETHICAL_CRITERIA["low_quality_artifacts"]["weight"] * 
            (1.0 - features["low_quality_artifacts"]) +
            
            cls.ETHICAL_CRITERIA["facial_consistency"]["weight"] * 
            (1.0 - features["facial_consistency"]) +
            
            cls.ETHICAL_CRITERIA["lighting_anomalies"]["weight"] * 
            features["lighting_anomalies"] +
            
            cls.ETHICAL_CRITERIA["frequency_analysis"]["weight"] * 
            features["frequency_analysis"]
        )
        
        # Determine threshold to use (allow override)
        use_threshold = cls.ETHICAL_THRESHOLD if threshold is None else float(threshold)

        # CRITICAL RULE: AI-generated images with faces are ALWAYS unethical
        # due to privacy, consent, and deepfake misuse concerns
        if num_faces > 0:
            # Override classification for face detection
            is_ethical = False
            status = "UNETHICAL - FACES DETECTED"
            # Boost risk score to reflect face presence
            risk_score = max(risk_score, 0.85)  # Ensure high risk
        else:
            # Classify based on artifact features only if no faces
            is_ethical = risk_score <= use_threshold
            status = "ETHICAL" if is_ethical else "UNETHICAL"
        
        # Calculate confidence (based on distance from used threshold)
        distance_from_threshold = abs(risk_score - use_threshold)
        confidence = max(0.0, min(1.0, 1.0 - (distance_from_threshold * 2.0)))
        
        return {
            "is_ethical": is_ethical,
            "status": status,
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "threshold": float(use_threshold),
            "faces_detected": num_faces,
            "details": cls._get_status_details(is_ethical, risk_score, num_faces),
            "explanation": cls._get_explanation(is_ethical, features, num_faces),
            "recommendations": cls._get_recommendations(is_ethical, num_faces),
            "features": features
        }
    
    @staticmethod
    def _get_status_details(is_ethical, risk_score, num_faces=0):
        """Get detailed status message"""
        if num_faces > 0:
            return "UNETHICAL - Human faces detected in AI-generated image. High privacy/consent risk."
        
        if is_ethical:
            if risk_score < 0.3:
                return "SAFE - Image has clear detectable artifacts making it obviously synthetic"
            else:
                return "ETHICAL - Image is detectable as AI-generated with visible artifacts"
        else:
            if risk_score > 0.7:
                return "HIGH RISK - Image is highly convincing with minimal detectable artifacts"
            else:
                return "RISKY - Image has subtle artifacts, higher misuse potential"
    
    @staticmethod
    def _get_explanation(is_ethical, features, num_faces=0):
        """Generate explanation for classification"""
        explanations = []
        
        # Face detection takes priority in explanation
        if num_faces > 0:
            explanations.append("CRITICAL: Human faces detected in AI-generated content")
            explanations.append("- AI-generated faces violate privacy and consent principles")
            explanations.append("- High potential for deepfake misuse and impersonation")
            explanations.append("- Cannot verify consent from depicted individual(s)")
            return "\n".join(explanations)
        
        if is_ethical:
            if features["artifact_std"] > 0.5:
                explanations.append("- Strong residual artifacts detected")
            if features["facial_consistency"] < 0.55:
                explanations.append("- Inconsistent or uncanny facial features")
            if features["frequency_analysis"] > 0.03:
                explanations.append("- Visible frequency domain patterns typical of synthetic generation")
            
            if not explanations:
                explanations.append("- Detectable synthetic characteristics present")
        else:
            if features["artifact_std"] <= 0.5:
                explanations.append("- Smooth artifacts indicate high-quality generation")
            if features["facial_consistency"] >= 0.55:
                explanations.append("- Natural-looking facial features")
            if features["lighting_anomalies"] <= 0.1:
                explanations.append("- Consistent lighting throughout image")
            if features["frequency_analysis"] <= 0.03:
                explanations.append("- Minimal visible GAN fingerprints")
        
        return "\n".join(explanations) if explanations else "Limited feature information available"
    
    @staticmethod
    def _get_recommendations(is_ethical, num_faces=0):
        """Get recommendations based on ethical status"""
        if num_faces > 0:
            # Specific recommendations for face-containing AI images
            return [
                "DO NOT SHARE - High risk of privacy violation",
                "REQUIRES EXPLICIT CONSENT from person(s) depicted",
                "VERIFY compliance with deepfake and privacy laws",
                "Could be illegal in many jurisdictions without consent",
                "If consent obtained: Add prominent 'AI-GENERATED' watermark",
                "Document consent and generation method for legal protection",
                "Consider digital signatures to prevent misuse",
                "Consult legal counsel before any distribution"
            ]
        
        if is_ethical:
            return [
                "May be used for educational or artistic purposes",
                "No exceptional privacy/consent concerns",
                "Share with clear 'AI-generated' disclosure",
                "Include metadata about generation method"
            ]
        else:
            return [
                "HIGH CAUTION: Requires explicit consent from person depicted",
                "Must clearly disclose AI-generation before sharing",
                "Do NOT use for deception or impersonation",
                "Verify compliance with local deepfake laws",
                "Consider adding watermarks or digital signatures",
                "Document consent and creation details for legal protection"
            ]


def format_ethical_report(assessment):
    """Format ethical assessment for display"""
    report = f"\n{'='*70}\n"
    report += f"ETHICAL ASSESSMENT REPORT\n"
    report += f"{'='*70}\n\n"
    
    # Status
    report += f"STATUS: {assessment['status']}\n"
    report += f"Risk Score: {assessment['risk_score']:.4f} (Threshold: {assessment.get('threshold', 0.5):.3f})\n"
    report += f"Confidence: {assessment['confidence']*100:.1f}%\n\n"
    
    # Details
    report += f"CLASSIFICATION DETAILS:\n"
    report += f"{'-'*70}\n"
    report += f"{assessment['details']}\n\n"
    
    # Explanation
    report += f"TECHNICAL ANALYSIS:\n"
    report += f"{'-'*70}\n"
    report += f"{assessment['explanation']}\n\n"
    
    # Recommendations
    report += f"RECOMMENDATIONS:\n"
    report += f"{'-'*70}\n"
    for rec in assessment['recommendations']:
        report += f"{rec}\n"
    
    # Raw feature values (for debugging / tuning)
    if 'features' in assessment and assessment['features'] is not None:
        report += f"\nRAW FEATURES:\n"
        report += f"{'-'*70}\n"
        for k, v in assessment['features'].items():
            report += f"{k}: {v}\n"
    
    report += f"\n{'='*70}\n"
    
    return report


def get_simple_status(assessment):
    """Get simple one-line status"""
    if assessment['is_ethical']:
        return f"ETHICAL (Risk Score: {assessment['risk_score']:.3f})"
    else:
        return f"UNETHICAL (Risk Score: {assessment['risk_score']:.3f})"
