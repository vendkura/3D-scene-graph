# 3D Scene Graph for Spatial AI / LLM

A Python pipeline that transforms a semantically labelled 3D point cloud into a
structured scene graph, exports it to OpenUSD, and provides multiple visualization
layers for spatial AI and LLM-based reasoning tasks.

## Pipeline overview

```
CSV point cloud → object extraction → feature computation
→ spatial relationships → scene graph (NetworkX) → USD export + JSON summary
```

## Steps

| Step | What it does |
|------|-------------|
| Load | Reads a `;`-delimited CSV with `x y z semantic_label` columns |
| Extract objects | DBSCAN clustering per semantic class → individual object instances |
| Compute features | Volume, surface area, compactness, height, point density per object |
| Spatial relationships | Pairwise classification: `above`, `below`, `adjacent`, `near`, `inside`, `contains` |
| Scene graph | NetworkX `DiGraph` with rich node/edge attributes |
| Validate | Checks for empty graph and missing node attributes |
| Export | OpenUSD `.usda` stage + `_summary.json` analytics file |

## Visualizations

- **`plot_scene_graph_stats(G)`** — semantic label distribution, relationship type distribution, node degree histogram
- **`visualize_scene_graph(G)`** — 2D NetworkX layout, nodes coloured by semantic label, edge relationship labels
- **`visualize_scene_graph_3d(G)`** — 3D Matplotlib plot, nodes at real XYZ centroids, coloured by semantic label
- **`visualize_3d_scene(objects)`** — Open3D point cloud + axis-aligned bounding boxes per object

## Usage

```python
result = process_semantic_pointcloud_to_usd(
    'path/to/labelled.csv',
    'output/scene.usda',
    eps=0.2,
    min_samples=20,
    distance_threshold=3.0
)
print(result)
```

## Requirements

```
numpy pandas scikit-learn networkx open3d matplotlib scipy
usd-core  # optional — required for .usda export
```

## Output

- `scene.usda` — OpenUSD stage with object prims and relationship metadata
- `scene_summary.json` — node/edge counts, semantic distribution, relationship types
