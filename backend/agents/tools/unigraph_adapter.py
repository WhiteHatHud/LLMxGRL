# agents/tools/unigraph_adapter.py
from typing import Dict, Any, List
from sqlalchemy.orm import Session
import pypdf
from pathlib import Path

class UniGraphAdapter:
    """Normalize multimodal inputs to text"""
    
    def normalize_pdf(self, filepath: str) -> str:
        """Extract text from PDF"""
        try:
            with open(filepath, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text_parts = []
                for page in reader.pages[:10]:  # Limit pages
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                return "\n".join(text_parts)
        except Exception as e:
            return f"PDF extraction failed: {str(e)}"
    
    def normalize_image(self, filepath: str) -> str:
        """Stub for image normalization"""
        # Placeholder: return filename and basic info
        filename = Path(filepath).name
        return f"Image: {filename} (OCR not implemented)"
    
    def normalize_node_features(
        self,
        features: Dict[str, Any]
    ) -> str:
        """Convert node features to text"""
        text_parts = []
        for key, value in features.items():
            text_parts.append(f"{key}: {value}")
        return "; ".join(text_parts)
    
    def process_graph(
        self,
        db: Session,
        graph_id: str,
        artifacts: List[str]
    ) -> Dict[str, Any]:
        """Process all multimodal inputs for a graph"""
        results = {
            "pdfs_processed": 0,
            "images_processed": 0,
            "texts_updated": 0
        }
        
        for artifact in artifacts:
            if artifact.endswith('.pdf'):
                text = self.normalize_pdf(artifact)
                # Store in NodeText or Artifact
                results["pdfs_processed"] += 1
            elif artifact.endswith(('.png', '.jpg', '.jpeg')):
                text = self.normalize_image(artifact)
                results["images_processed"] += 1
        
        return results# Multimodal→text normalization stubs
