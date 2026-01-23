# Research Visualization Asset Generator

This project generates 4 high-quality PNG visualizations for slides 8-11 of the LLM x GRL research presentation.

## Generated Assets

1. **slide08_misalignment.png** - Misalignment: Token Semantics vs Graph Topology
   - Split-panel visualization showing LLM embedding space vs graph structure
   - Highlights fundamental representation mismatch

2. **slide09_cross_domain_drop.png** - Cross-Domain Generalization Drop
   - Bar chart showing accuracy degradation across domains
   - Illustrative placeholder for real evaluation data

3. **slide10_hop_accuracy_curve.png** - Accuracy vs Graph Traversal Depth
   - Line chart showing performance degradation with multi-hop reasoning
   - Illustrative placeholder for Spider dataset stratification

4. **slide11_graphrag_pipeline.png** - GraphRAG Pipeline for Structured Reasoning
   - Flow diagram of the complete GraphRAG architecture
   - Shows retrieval, LLM agent, verification, and feedback loops

## Requirements

- Python 3.10 or higher
- Dependencies listed in `requirements_assets.txt`

## Installation

```bash
# Install dependencies
pip install -r requirements_assets.txt
```

## Usage

```bash
# Generate all 4 PNG assets
python generate_assets.py
```

The script will:
1. Create an `assets/` directory (if it doesn't exist)
2. Generate all 4 PNG files at 300 DPI in 16:9 aspect ratio (1920x1080)
3. Print success messages with output paths

## Output

All generated files will be saved to `./assets/`:

```
assets/
├── slide08_misalignment.png
├── slide09_cross_domain_drop.png
├── slide10_hop_accuracy_curve.png
└── slide11_graphrag_pipeline.png
```

## Technical Details

- **Resolution**: 1920x1080 (16:9 aspect ratio)
- **DPI**: 300 (high quality for presentations)
- **Background**: White
- **Style**: Clean, professional, large readable fonts
- **Deterministic**: Uses fixed random seed (42) for reproducibility

## Libraries Used

- **matplotlib**: Charts, plots, and general visualization
- **networkx**: Graph visualization (schema graphs)
- **numpy**: Data generation and numerical operations

## Customization

To modify the visualizations, edit `generate_assets.py`:

- Change colors in the respective `generate_slideXX_*()` functions
- Adjust font sizes via `plt.rcParams` global settings
- Modify data values (currently illustrative placeholders)
- Update text labels and annotations

## Integration with Presentation

These assets are designed to be manually inserted into slides 8-11 of your presentation. They are:

- **High resolution** (suitable for projection)
- **16:9 aspect ratio** (matches standard presentation format)
- **White background** (integrates with light-themed slides)
- **Self-contained** (no external dependencies in the images)

## Replacing Placeholder Data

The bar chart (slide 9) and line chart (slide 10) use illustrative data. To replace with real evaluation results:

1. Edit `generate_assets.py`
2. Update the data arrays in:
   - `generate_slide09_cross_domain()`: `accuracies = [...]`
   - `generate_slide10_hop_accuracy()`: `accuracies = [...]`
3. Re-run `python generate_assets.py`

## Troubleshooting

**ImportError: No module named 'matplotlib'**
```bash
pip install -r requirements_assets.txt
```

**Assets directory already exists**
- The script will not overwrite the directory
- Existing PNG files will be replaced with new versions

**Font warnings**
- Default sans-serif fonts are used
- System fonts will be automatically selected
- No action needed unless you want specific fonts

## License

These visualizations are part of the LLM x GRL research project.
