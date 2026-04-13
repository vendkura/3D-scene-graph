from typing import Dict

#Base libraries
import numpy as np
import pandas as pd
import json

#Graph and algorithm-related libraries
import networkx as nx
from sklearn.cluster import DBSCAN

# For visualization
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from pxr import Usd, UsdGeom, Sdf, Gf, UsdShade
    USD_AVAILABLE = True
except ImportError:
    print("USD not available. Install with: pip install usd-core")
    USD_AVAILABLE = False


def load_semantic_point_cloud(file_path, column_name = 'semantic_label'):
    """Load semantic point cloud data from ASCII formats."""
        
    df = pd.read_csv(file_path, delimiter=';')
    class_names = ['ceiling', 'floor', 'wall', 'chair', 'furniture', 'table']
    
    # Assuming the numerical labels are 0.0, 1.0, 2.0, ...
    label_map = {float(i): class_names[i] for i in range(len(class_names))}
    
    df[column_name] = df[column_name].map(label_map)
    
    # Sample up to 150k rows — safe if the file has fewer points
    n = min(150000, len(df))
    return df.sample(n=n, random_state=1)

raw_data = load_semantic_point_cloud('data/indoor_room_labelled.csv')
demo_data = raw_data.sample(n=150000, random_state=1)
# print(demo_data.head(1000))


def visualize_semantic_pointcloud(df, point_size = 2.0):
    """Visualize semantic point cloud with flat colors per semantic label using Open3D."""
    
    # Extract coordinates
    points = df[['x', 'y', 'z']].values
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create color mapping for semantic labels
    unique_labels = df['semantic_label'].unique()
    color_map = {}
    
    # Generate distinct colors for each label
    for i, label in enumerate(unique_labels):
        hue = i / len(unique_labels)
        color_map[label] = plt.cm.tab10(hue)[:3]
    
    # Assign colors based on semantic labels
    colors = np.array([color_map[label] for label in df['semantic_label']])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Semantic Point Cloud", width=1200, height=800)
    vis.add_geometry(pcd)
    
    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    
    print(f"Visualizing {len(points)} points with {len(unique_labels)} semantic classes:")
    for label in unique_labels:
        count = (df['semantic_label'] == label).sum()
        print(f"  {label}: {count} points")
    
    vis.run()
    vis.destroy_window()
# visualize_semantic_pointcloud(demo_data, point_size=3.0)


def extract_semantic_objects(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 10) -> Dict:
    """Extract individual objects from semantic point cloud using clustering."""
    """🦥 Florent’s Geeky Note: The eps parameter controls clustering sensitivity — smaller values create more separate objects, larger values merge nearby items. Start with 0.5–1.0 for room-scale scenes and adjust based on your point cloud density. The optimal eps often equals 2–3 times your average point spacing. Dense LiDAR scans need smaller eps values than sparse smartphone captures. You can check out the book for a complete deep dive"""
    objects = {}
    
    for label in df['semantic_label'].unique():
        label_points = df[df['semantic_label'] == label]
        
        if len(label_points) < min_samples:
            continue
            
        # Apply DBSCAN clustering
        coords = label_points[['x', 'y', 'z']].values
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Group by cluster
        label_points_copy = label_points.copy()
        label_points_copy['cluster'] = clustering.labels_
        
        for cluster_id in np.unique(clustering.labels_):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_points = label_points_copy[label_points_copy['cluster'] == cluster_id]
            object_key = f"{label}_{cluster_id}"
            
            objects[object_key] = {
                'points': cluster_points,
                'centroid': cluster_points[['x', 'y', 'z']].mean().values,
                'bounds': {
                    'min': cluster_points[['x', 'y', 'z']].min().values,
                    'max': cluster_points[['x', 'y', 'z']].max().values
                },
                'semantic_label': label,
                'point_count': len(cluster_points)
            }
    
    return objects



def visualize_room_furniture_graph(furniture_data):
    """Builds and visualizes a graph of room furniture."""
    
    G = nx.Graph()
    for item, connections in furniture_data.items():
        G.add_node(item)
        for connected_item in connections:
            G.add_edge(item, connected_item)
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout
    nx.draw(G, pos, with_labels=True, node_color='salmon',
            node_size=500, font_size=10, font_weight='bold')
    plt.title("Room Furniture Graph")
    plt.show()
# Example of a room description:
# room_layout = {
#     "bed": ["nightstand", "lamp", "rug"],
#     "nightstand": ["bed", "lamp"],
#     "lamp": ["bed", "nightstand"],
#     "rug": ["bed", "sofa", "bookshelf"],
#     "sofa": ["coffee table", "TV", "rug"],
#     "coffee table": ["sofa", "TV"],
#     "TV": ["sofa", "coffee table", "TV stand"],
#     "TV stand": ["TV"],
#     "bookshelf": ["desk"],
#     "desk": ["bookshelf", "chair"],
#     "chair": ["desk"]
# }
# visualize_room_furniture_graph(room_layout)

def estimate_surface_area(points):
    """
        Why Convex Hull? Fast computation even for thousands of points Always produces valid results (no self-intersections) Good approximation for most furniture objects Handles irregular point distributions gracefully For highly concave objects (L-shaped desks, hollow objects), consider alpha shapes or Poisson surface reconstruction for more accurate area estimates.
    """
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.area  # Surface area of convex hull
    except:
        return 0.0  # Fallback for degenerate cases


def compute_object_features(objects):
    """
        Compute geometric and semantic features for each object.
        🦥 Geeky Note: Compactness equals 1.0 for perfect spheres and approaches 0 for needle-like objects. It’s scale-invariant, making it perfect for comparing objects of different sizes.
    
    🌱 Growing Note: Add custom features for your domain. For furniture analysis, compute seat height, leg count, or symmetry measures. The feature dictionary is your canvas.
    """
    features = {}
    
    for obj_name, obj_data in objects.items():
        points = obj_data['points'][['x', 'y', 'z']].values
        
        # Geometric features
        volume = np.prod(obj_data['bounds']['max'] - obj_data['bounds']['min'])
        surface_area = estimate_surface_area(points)
        compactness = (surface_area ** 3) / (36 * np.pi * volume ** 2) if volume > 0 else 0
        
        features[obj_name] = {
            'volume': volume,
            'surface_area': surface_area,
            'compactness': compactness,
            'height': obj_data['bounds']['max'][2] - obj_data['bounds']['min'][2],
            'semantic_label': obj_data['semantic_label'],
            'centroid': obj_data['centroid'],
            'point_density': obj_data['point_count'] / volume if volume > 0 else 0
        }
    
    return features

objects = extract_semantic_objects(demo_data,0.5,10)
features = compute_object_features(objects)
# print(features)

# Step 7. Spatial Relationship Computation + Topology

# Stage 1: Topology analytics : I want to determine if one object is completely inside another, and if it is adjacent (Essential for hierarchical scene understanding.)

# For object A to be contained in object B, all of A’s boundaries must be within B’s boundaries. This means we want to check all six faces (min/max for X, Y, Z)
def is_contained(bounds1, bounds2):
    """Check if object1 is contained within object2."""
    return (np.all(bounds1['min'] >= bounds2['min']) and 
            np.all(bounds1['max'] <= bounds2['max']))
# Each 3D “bounding-box” object has six faces (front/back, left/right, top/bottom). I check if any face pair is closer than the tolerance threshold
def are_adjacent(bounds1, bounds2, tolerance=0.1):
    # Check if faces are close along each axis
    for axis in range(3):  # X, Y, Z axes
        # Face-to-face proximity checks
        if (abs(bounds1['max'][axis] - bounds2['min'][axis]) < tolerance or
            abs(bounds2['max'][axis] - bounds1['min'][axis]) < tolerance):
            return True
    return False

# Stage 2: Relationship Classification : For each nearby pair, we want to classify their spatial connection.

def determine_relationship_type(obj1, obj2, threshold):
    centroid1 = obj1['centroid']
    centroid2 = obj2['centroid']
    
    distance = np.linalg.norm(centroid1 - centroid2)
    if distance > threshold:
        return None  # Too far apart
    
    # Vertical relationship analysis
    z_diff = centroid1[2] - centroid2[2]
    if abs(z_diff) > 0.5:  # Significant height difference
        return 'above' if z_diff > 0 else 'below'
    
    # Containment analysis
    bounds1 = obj1['bounds']
    bounds2 = obj2['bounds']
    
    if is_contained(bounds1, bounds2):
        return 'inside'
    elif is_contained(bounds2, bounds1):
        return 'contains'
    
    # Adjacency analysis
    if are_adjacent(bounds1, bounds2, tolerance=0.3):
        return 'adjacent'
    
    return 'near'  # Default fallback

# Stage 3: Exhaustive Pairwise Analysis : We are going to check every object against every other object.
# 🦥 Geeky Note: The nested loop with enumerate(object_names[i+1:], i+1) ensures each pair is checked exactly once. No duplicate relationships.
def compute_spatial_relationships(objects, distance_threshold=2.0):
    relationships = []
    object_names = list(objects.keys())
    
    for i, obj1 in enumerate(object_names):
        for j, obj2 in enumerate(object_names[i+1:], i+1):  # Avoid duplicates
            rel_type = determine_relationship_type(objects[obj1], objects[obj2], distance_threshold)
            if rel_type:  # Only keep valid relationships
                relationships.append((obj1, obj2, rel_type))
    
    return relationships
# Step 8. Building the Scene Graph
def build_scene_graph(objects, relationships, features):
    """
        Add edge weights based on relationship strength or confidence. This enables weighted graph algorithms for pathfinding and centrality analysis.
    """
    G = nx.DiGraph()
    
    # Add nodes with rich attributes
    for obj_name, obj_data in objects.items():
        obj_features = features.get(obj_name, {}).copy()
        obj_features.pop('semantic_label', None)  # The obj_features.pop('semantic_label', None) line prevents duplicate keyword arguments when adding nodes. NetworkX would crash on duplicate key
        obj_features.pop('centroid', None)  # centroid is passed explicitly below
        
        G.add_node(obj_name, 
                  semantic_label=obj_data['semantic_label'],
                  centroid=obj_data['centroid'].tolist(),
                  point_count=obj_data['point_count'],
                  **obj_features)
    
    # Add relationship edges
    for obj1, obj2, rel_type in relationships:
        G.add_edge(obj1, obj2, relationship=rel_type)
    
    return G

# Step 9. NetworkX Analytics That Matter : Raw graphs are hard to understand. analyze_scene_graph() computes summary statistics that reveal the scene's spatial structure and complexity
# 🦥 Ville’s Geeky Note: weakly_connected_components counts components in the underlying undirected graph. It tells you if your scene has isolated object clusters.
def analyze_scene_graph(G):
    analysis = {
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges(),
        'semantic_distribution': {},
        'relationship_types': {},
        'connected_components': nx.number_weakly_connected_components(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }
    
    # Analyze semantic distribution
    for node, data in G.nodes(data=True):
        label = data.get('semantic_label', 'unknown')
        analysis['semantic_distribution'][label] = analysis['semantic_distribution'].get(label, 0) + 1
    
    # Analyze relationship types
    for _, _, data in G.edges(data=True):
        rel = data.get('relationship', 'unknown')
        analysis['relationship_types'][rel] = analysis['relationship_types'].get(rel, 0) + 1

    return analysis



# -------------------------------------------
# Visualize Scene Graph Statistics
def plot_scene_graph_stats(G):
    """Plot semantic label distribution, relationship type distribution, and node degree distribution."""
    analysis = analyze_scene_graph(G)

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    pastel_colors = ['#AEC6CF', '#B5A9C8', '#A8C8A0', '#F4A8A8', '#F4D78A', '#F9C58A']

    # 1. Semantic Label Distribution
    sem = analysis['semantic_distribution']
    axes[0].bar(sem.keys(), sem.values(),
                color=pastel_colors[:len(sem)], edgecolor='white')
    axes[0].set_title('Semantic Label Distribution')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=15)

    # 2. Relationship Type Distribution
    rel = analysis['relationship_types']
    rel_colors = ['#F4A8A8', '#F4D78A', '#D3D3D3']
    axes[1].bar(rel.keys(), rel.values(),
                color=rel_colors[:len(rel)], edgecolor='white')
    axes[1].set_title('Relationship Type Distribution')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=15)

    # 3. Node Degree Distribution
    degrees = [d for _, d in G.degree()]
    axes[2].hist(degrees, bins=10, color='#AEC6CF', edgecolor='white')
    axes[2].set_title('Node Degree Distribution')
    axes[2].set_xlabel('Degree')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Visualize the Scene Graph (2D - NetworkX)
def visualize_scene_graph(G):
    """Render the scene graph with semantic labels and relationship types."""
    plt.figure(figsize=(14, 10))

    # Use spatial centroids (x, y) as layout positions
    pos = {}
    for node, data in G.nodes(data=True):
        c = data.get('centroid', [0, 0, 0])
        pos[node] = (c[0], c[1])

    # Color nodes by semantic label
    labels_unique = list({data['semantic_label'] for _, data in G.nodes(data=True)})
    cmap = plt.colormaps.get_cmap('tab20')
    label_to_color = {lbl: cmap(i / max(len(labels_unique) - 1, 1)) for i, lbl in enumerate(labels_unique)}
    node_colors = [label_to_color[G.nodes[n]['semantic_label']] for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7)

    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, alpha=0.5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Legend
    patches = [mpatches.Patch(color=label_to_color[l], label=l) for l in labels_unique]
    plt.legend(handles=patches, loc='upper left', fontsize=8)
    plt.title("3D Scene Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


#  Visualize the Scene Graph (3D - Matplotlib)
def visualize_scene_graph_3d(G):
    """Render the scene graph in 3D using node centroids as spatial positions."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color nodes by semantic label
    labels_unique = list({data['semantic_label'] for _, data in G.nodes(data=True)})
    cmap = plt.colormaps.get_cmap('tab20')
    label_to_color = {lbl: cmap(i / max(len(labels_unique) - 1, 1)) for i, lbl in enumerate(labels_unique)}

    # Draw edges using 3D centroid coordinates
    for u, v in G.edges():
        c1 = G.nodes[u].get('centroid', [0, 0, 0])
        c2 = G.nodes[v].get('centroid', [0, 0, 0])
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                color='gray', alpha=0.4, linewidth=0.8)

    # Draw nodes
    for node, data in G.nodes(data=True):
        c = data.get('centroid', [0, 0, 0])
        color = label_to_color[data['semantic_label']]
        ax.scatter(c[0], c[1], c[2], color=color, s=120, zorder=5)
        ax.text(c[0], c[1], c[2], node, fontsize=6, zorder=6)

    # Legend
    patches = [mpatches.Patch(color=label_to_color[l], label=l) for l in labels_unique]
    ax.legend(handles=patches, loc='upper left', fontsize=8, title='Semantic Labels')

    ax.set_title("3D Scene Graph Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


# Visualize Objects in 3D (Open3D bounding boxes)
def visualize_3d_scene(objects):
    """Render point clusters with colored bounding boxes per semantic label."""
    labels_unique = list({d['semantic_label'] for d in objects.values()})
    cmap = plt.colormaps.get_cmap('tab20')
    label_to_color = {lbl: list(cmap(i / max(len(labels_unique) - 1, 1))[:3]) for i, lbl in enumerate(labels_unique)}

    geometries = []
    for obj_name, obj_data in objects.items():
        pts = obj_data['points'][['x', 'y', 'z']].values
        color = label_to_color[obj_data['semantic_label']]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        # Bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=obj_data['bounds']['min'],
            max_bound=obj_data['bounds']['max']
        )
        bbox.color = color
        geometries.append(bbox)

    o3d.visualization.draw_geometries(geometries, window_name="3D Scene Graph", width=1280, height=720)

# -----------------------------------
# Building Pattern Detection Functions 
def create_pattern_matcher(scene_graph, pattern_graph, 
                          node_attr='semantic_label', 
                          edge_attr='relationship'):
    """Create flexible pattern matcher for scene graph analysis."""
    
    def semantic_node_match(node1_attrs, node2_attrs):
        """
        Check if nodes match semantically with flexibility.
        
        🐦 Ville’s Note: The DiGraphMatcher uses the VF2 algorithm internally, which has exponential worst-case complexity but performs well on sparse graphs typical of spatial scenes. The semantic groupings reduce false negatives while maintaining precision.
        """
        label1 = node1_attrs.get(node_attr, 'unknown')
        label2 = node2_attrs.get(node_attr, 'unknown')


        
        # Exact match first
        if label1 == label2:
            return True
            
        # Semantic equivalence groups
        furniture_group = {'chair', 'seat', 'stool'}
        surface_group = {'table', 'desk', 'surface'}
        
        if label1 in furniture_group and label2 in furniture_group:
            return True
        if label1 in surface_group and label2 in surface_group:
            return True
            
        return False
    
    def relationship_edge_match(edge1_attrs, edge2_attrs):
        """Check if relationships match with spatial tolerance."""
        rel1 = edge1_attrs.get(edge_attr, 'unknown')
        rel2 = edge2_attrs.get(edge_attr, 'unknown')
        
        # Direct match
        if rel1 == rel2:
            return True
            
        # Spatial relationship equivalence
        proximity_group = {'adjacent', 'near', 'close'}
        support_group = {'above', 'on', 'supported_by'}
        
        if rel1 in proximity_group and rel2 in proximity_group:
            return True
        if rel1 in support_group and rel2 in support_group:
            return True
            
        return False
    
    # Create NetworkX matcher with custom comparison functions
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        scene_graph,
        pattern_graph,
        node_match=semantic_node_match,
        edge_match=relationship_edge_match
    )
    
    return matcher


# Pattern Definition and Detection
def create_meeting_room_pattern():
    """Define meeting room spatial pattern as scene graph."""
    pattern = nx.DiGraph()
    
    # Core meeting room components
    pattern.add_node('table_1', semantic_label='table')
    pattern.add_node('chair_1', semantic_label='chair')
    pattern.add_node('chair_2', semantic_label='chair')
    pattern.add_node('chair_3', semantic_label='chair')
    
    # Spatial relationships
    pattern.add_edge('chair_1', 'table_1', relationship='adjacent')
    pattern.add_edge('chair_2', 'table_1', relationship='adjacent')
    pattern.add_edge('chair_3', 'table_1', relationship='adjacent')
    
    return pattern

def create_workstation_pattern():
    """Define individual workstation spatial pattern."""
    pattern = nx.DiGraph()
    
    # Workstation components
    pattern.add_node('desk_1', semantic_label='desk')
    pattern.add_node('chair_1', semantic_label='chair')
    pattern.add_node('storage_1', semantic_label='furniture')
    
    # Functional relationships
    pattern.add_edge('chair_1', 'desk_1', relationship='adjacent')
    pattern.add_edge('storage_1', 'desk_1', relationship='near')
    
    return pattern
def find_spatial_patterns(scene_graph, pattern_graph):
    """Find all occurrences of pattern within scene graph."""
    matcher = create_pattern_matcher(scene_graph, pattern_graph)
    
    matches = list(matcher.subgraph_isomorphisms_iter())
    
    # Extract match details with spatial context
    pattern_matches = []
    for match in matches:
        # match maps pattern nodes to scene nodes
        scene_objects = list(match.values())
        
        # Calculate pattern centroid for spatial reference
        centroids = []
        for obj_id in scene_objects:
            if scene_graph.has_node(obj_id):
                centroid = scene_graph.nodes[obj_id].get('centroid', [0,0,0])
                centroids.append(centroid)
        
        if centroids:
            pattern_center = np.mean(centroids, axis=0)
            pattern_matches.append({
                'objects': scene_objects,
                'mapping': match,
                'center': pattern_center.tolist(),
                'confidence': calculate_pattern_confidence(scene_graph, match)
            })
    
    return pattern_matches

def calculate_pattern_confidence(scene_graph, node_mapping):
    """Calculate confidence score for pattern match."""
    confidence_factors = []
    
    # Geometric consistency
    positions = []
    for scene_node in node_mapping.values():
        centroid = scene_graph.nodes[scene_node].get('centroid', [0,0,0])
        positions.append(centroid)
    
    if len(positions) > 1:
        # Measure spatial compactness
        distances = []
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distances.append(np.linalg.norm(np.array(pos1) - np.array(pos2)))
        
        avg_distance = np.mean(distances)
        # Lower average distance = higher confidence for compact patterns
        spatial_confidence = max(0, 1.0 - avg_distance / 5.0)  # 5m normalization
        confidence_factors.append(spatial_confidence)
    
    # Semantic consistency (could add more factors)
    semantic_confidence = 1.0  # Placeholder for semantic validation
    confidence_factors.append(semantic_confidence)
    
    return np.mean(confidence_factors)
"""
Below is a generic, reusable helper that allows you to search any large scene graph (E.g., a building) for occurrences of another small scene graph (E.g., a place described with a scene graph).

🐦 Ville’s note: The helper relies on NetworkX’s built-in subgraph-isomorphism engine and works with the attribute conventions you have already adopted (semantic_label on nodes, relationship on edges).
```
matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        G,
        pattern,
        node_match=lambda d1, d2: node_cmp(d1.get(node_attr), d2.get(node_attr)),
        edge_match=lambda d1, d2: edge_cmp(d1.get(edge_attr), d2.get(edge_attr)),
    )

    # NetworkX streams one mapping at a time
    yield from matcher.subgraph_isomorphisms_iter()
```
"""

# Step 10. OpenUSD Integration and Export

def create_usd_object(stage, node_name: str, data: dict):
    """Create a USD prim for a single scene graph node."""
    safe_name = node_name.replace('-', '_')
    prim_path = f'/Scene/Geometry/{safe_name}'
    prim = stage.DefinePrim(prim_path, 'Xform')

    centroid = data.get('centroid', [0, 0, 0])
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(centroid[0]), float(centroid[1]), float(centroid[2])))

    prim.SetCustomDataByKey('semantic_label', data.get('semantic_label', 'unknown'))
    prim.SetCustomDataByKey('point_count', int(data.get('point_count', 0)))

    bounds = data.get('bounds', {})
    if bounds:
        mn = bounds.get('min', [0, 0, 0])
        mx = bounds.get('max', [0, 0, 0])
        cube = UsdGeom.Cube.Define(stage, prim_path + '/bbox')
        size = [float(mx[i] - mn[i]) for i in range(3)]
        cube.GetSizeAttr().Set(max(size) if max(size) > 0 else 1.0)


def add_relationships_to_stage(stage, scene_graph: nx.DiGraph):
    """Store edge relationships as USD custom metadata on the root prim."""
    root = stage.GetPrimAtPath('/Scene')
    relationships = []
    for u, v, data in scene_graph.edges(data=True):
        relationships.append({
            'from': u,
            'to': v,
            'relationship': data.get('relationship', 'unknown')
        })
    root.SetCustomDataByKey('relationships', str(relationships))


def validate_scene_graph(G: nx.DiGraph) -> dict:
    """Basic validation checks on the scene graph."""
    issues = []
    if G.number_of_nodes() == 0:
        issues.append('Scene graph has no nodes')
    for node, data in G.nodes(data=True):
        if 'semantic_label' not in data:
            issues.append(f'Node {node} missing semantic_label')
        if 'centroid' not in data:
            issues.append(f'Node {node} missing centroid')
    return {'is_valid': len(issues) == 0, 'issues': issues}


def export_scene_summary(G: nx.DiGraph, output_path: str):
    """Export scene graph analysis as a JSON summary file."""
    analysis = analyze_scene_graph(G)
    # Convert any non-serializable types
    summary = {
        'node_count': analysis['node_count'],
        'edge_count': analysis['edge_count'],
        'connected_components': analysis['connected_components'],
        'avg_degree': float(analysis['avg_degree']),
        'semantic_distribution': analysis['semantic_distribution'],
        'relationship_types': analysis['relationship_types'],
    }
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def create_usd_stage(scene_graph: nx.DiGraph, output_path: str) -> bool:
    """Create USD stage from scene graph and export to file."""
    if not USD_AVAILABLE:
        print("USD not available. Cannot create USD stage.")
        return False
    
    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)
    
    # Set up scene hierarchy
    root_prim = stage.DefinePrim('/Scene', 'Xform')
    stage.SetDefaultPrim(root_prim)
    
    # Create geometry scope for objects
    geom_scope = UsdGeom.Scope.Define(stage, '/Scene/Geometry')
    
    # Add each object as a primitive
    for node, data in scene_graph.nodes(data=True):
        create_usd_object(stage, node, data)
    
    # Add relationships as metadata
    add_relationships_to_stage(stage, scene_graph)
    
    # Save stage
    stage.Save()
    return True

# Step 11. Practical Applications and Extensions
def process_semantic_pointcloud_to_usd(input_path, output_usd, eps = 0.8, min_samples = 15, distance_threshold = 3.0):
    """Complete pipeline from semantic point cloud to USD scene graph."""
    results = {'success': False, 'files_created': [], 'analysis': {}}
    
    try:
        # Load and validate data
        print("Loading semantic point cloud...")
        df = load_semantic_point_cloud(input_path)

        print(f"Loaded {len(df)} points with {df['semantic_label'].nunique()} semantic classes")
        
        # Extract objects
        print("Extracting semantic objects...")
        objects = extract_semantic_objects(df, eps=eps, min_samples=min_samples)
        print(f"Found {len(objects)} objects")
        
        # Compute features
        print("Computing object features...")
        features = compute_object_features(objects)
        
        # Find relationships
        print("Computing spatial relationships...")
        relationships = compute_spatial_relationships(objects, distance_threshold)
        print(f"Found {len(relationships)} spatial relationships")
        
        # Build scene graph
        print("Building scene graph...")
        scene_graph = build_scene_graph(objects, relationships, features)
        
        # Validate scene graph
        validation = validate_scene_graph(scene_graph)
        if not validation['is_valid']:
            print("Warning: Scene graph validation found issues:", validation['issues'])
        
        # Export to USD
        if USD_AVAILABLE:
            print(f"Exporting to USD: {output_usd}")
            usd_success = create_usd_stage(scene_graph, output_usd)
            if usd_success:
                results['files_created'].append(output_usd)
        
        # Export summary
        summary_path = output_usd.replace('.usda', '_summary.json')
        export_scene_summary(scene_graph, summary_path)
        results['files_created'].append(summary_path)
        
        # Store analysis results
        results['analysis'] = analyze_scene_graph(scene_graph)
        results['validation'] = validation
        results['success'] = True
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        results['error'] = str(e)
    
    return results

# Complete pipeline execution



# Step 12. LLM Integration for Spatial Reasoning : To include later after setting up the entire project

def query_scene_with_llm(scene_graph: nx.DiGraph, user_question: str, 
                        llm_client=None) -> str:
    """ 
        Query scene graph using LLM reasoning capabilities.
    
        🦥 Our Geeky Note: The prompt structure includes object positions, semantic labels, and relationship types in a format that LLMs can parse effectively. Temperature settings around 0.3–0.7 work well for spatial reasoning tasks — low enough for accuracy, high enough for creative insights.
    """
    
    # Create structured scene description
    scene_context = scene_graph_to_llm_prompt(scene_graph)
    
    # Build complete prompt
    full_prompt = f"""{scene_context}"""

# USER QUESTION: {user_question}
# Please provide a detailed answer based on the spatial relationships and object positions in the scene. Consider:
# 1. Direct spatial relationships between objects
# 2. Accessibility and navigation paths
# 3. Functional groupings and usage patterns
# 4. Safety or optimization concerns

# --- Run everything ---
relationships = compute_spatial_relationships(objects, distance_threshold=2.0)
scene_graph = build_scene_graph(objects, relationships, features)
analysis = analyze_scene_graph(scene_graph)
result = process_semantic_pointcloud_to_usd('data/indoor_room_labelled.csv', './data/output/demo_scene_c.usda', eps=0.2, min_samples=20, distance_threshold=3.0)
print(result)

# plot_scene_graph_stats(scene_graph)   # Bar charts + degree histogram
# visualize_scene_graph(scene_graph)    # 2D NetworkX graph
# visualize_scene_graph_3d(scene_graph) # 3D Matplotlib graph
visualize_3d_scene(objects)          # 3D point cloud + bounding boxes
