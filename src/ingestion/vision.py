import base64
import hashlib
import io
import os
import time
from typing import Dict, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# Load environment configuration
load_dotenv(".env.local")
load_dotenv()


class VisionDescriber:
    """Multimodal vision client using NVIDIA NIM Vision-Language Models (e.g. meta/llama-3.2-11b-vision-instruct).

    Analyzes lecture slides, diagrams, system charts, and textbook figures to generate
    detailed, grounded academic descriptions suitable for dense vector indexing.
    """

    DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        min_width: int = 150,
        min_height: int = 150,
        min_bytes: int = 1000,
        max_dim: int = 1024,
    ):
        self.api_key = (
            api_key
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("NVIDIA_EMBEDDER_KEY")
        )
        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("VISION_MODEL", self.DEFAULT_MODEL)

        self.min_width = min_width
        self.min_height = min_height
        self.min_bytes = min_bytes
        self.max_dim = max_dim

        # In-memory SHA-256 deduplication cache to eliminate duplicate slide template calls
        self._cache: Dict[str, str] = {}

        if self.api_key:
            self.client: Optional[OpenAI] = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = None

    def is_meaningful_diagram(self, image_bytes: bytes) -> bool:
        """Evaluate if an extracted image is an informative academic diagram or figure.

        Filters out tiny icons, bullets, divider lines, and small header/footer logos.
        """
        if not image_bytes or len(image_bytes) < self.min_bytes:
            return False

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                w, h = img.size
                if w < self.min_width or h < self.min_height:
                    return False

                # Reject extreme thin dividers or borders
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.15 or aspect_ratio > 6.5:
                    return False

                return True
        except Exception:
            return False

    def prepare_image_b64(self, image_bytes: bytes) -> str:
        """Preprocess and base64-encode image for vision API transfer."""
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Handle alpha channel transparency by compositing over white
            if img.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    bg.paste(img, mask=img.split()[3])
                else:
                    bg.paste(img.convert("RGBA"))
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Scale thumbnail if exceeding max dimension to minimize network latency
            if max(img.size) > self.max_dim:
                img.thumbnail((self.max_dim, self.max_dim), Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def describe_diagram(
        self,
        image_bytes: bytes,
        context_hint: str = "",
        max_retries: int = 2,
    ) -> str:
        """Generate structured academic explanation of a lecture slide or textbook diagram.

        Args:
            image_bytes: Raw image bytes.
            context_hint: Optional surrounding text or slide title to guide description.
            max_retries: Transient retry attempts.

        Returns:
            Pedagogical text description of the figure, or empty string if not a valid diagram.
        """
        if not self.is_meaningful_diagram(image_bytes):
            return ""

        # Check deduplication cache
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        if content_hash in self._cache:
            return self._cache[content_hash]

        # Resilient offline fallback if API client is not configured
        if self.client is None:
            fallback = f"Academic diagram illustrating {context_hint or 'lecture course concepts'}."
            self._cache[content_hash] = fallback
            return fallback

        b64_image = self.prepare_image_b64(image_bytes)

        prompt = (
            "You are an academic visual tutor. Analyze this lecture slide figure or academic diagram in detail. "
            "Provide a clear, factual, pedagogical explanation for students:\n"
            "1. Diagram Type & Topic: State what concept, system, or mechanism is illustrated.\n"
            "2. Structural Components & Flow: Detail each labelled box, component, connection, or axis.\n"
            "3. Key Takeaway & Formulas: Note any equations, numerical metrics, or key conclusions shown.\n"
            "Be concise, precise, and objective. Do not speculate."
        )
        if context_hint:
            prompt += f"\nContext hint: {context_hint}"

        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                                },
                            ],
                        }
                    ],
                    max_tokens=400,
                    temperature=0.2,
                )
                description = response.choices[0].message.content.strip()
                self._cache[content_hash] = description
                return description
            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
                else:
                    break

        # Fallback on remote failure
        fallback_msg = f"Visual diagram excerpt: {context_hint or 'lecture material'} (Vision model unavailable: {str(last_err)})"
        return fallback_msg
