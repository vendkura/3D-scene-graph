from typing import Dict

#Base libraries
import numpy as np
import pandas as pd
import json
import os

#Graph and algorithm-related libraries
import networkx as nx
from sklearn.cluster import DBSCAN

# For visualization
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Streamlit + Plotly
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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

# raw_data = load_semantic_point_cloud('data/indoor_room_labelled.csv')
# demo_data = raw_data.sample(n=150000, random_state=1)
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

# objects = extract_semantic_objects(demo_data,0.5,10)
# features = compute_object_features(objects)

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

# --- Run everything (disabled — Streamlit app runs the pipeline) ---
# relationships = compute_spatial_relationships(objects, distance_threshold=2.0)
# scene_graph = build_scene_graph(objects, relationships, features)
# analysis = analyze_scene_graph(scene_graph)
# result = process_semantic_pointcloud_to_usd('data/indoor_room_labelled.csv', './data/output/demo_scene_c.usda', eps=0.2, min_samples=20, distance_threshold=3.0)
# print(result)
# plot_scene_graph_stats(scene_graph)
# visualize_scene_graph(scene_graph)
# visualize_scene_graph_3d(scene_graph)
# visualize_3d_scene(objects)


# ============================================================
# STREAMLIT APP
# ============================================================

LABEL_COLORS = {
    'ceiling':   '#AEC6CF',
    'floor':     '#A8C8A0',
    'wall':      '#F4D78A',
    'chair':     '#F4A8A8',
    'table':     '#B5A9C8',
    'furniture': '#F9C58A',
}

def _color(label):
    return LABEL_COLORS.get(label, '#CCCCCC')


def plotly_pointcloud(df, max_points=40000):
    sample = df.sample(n=min(max_points, len(df)), random_state=42)
    colors = sample['semantic_label'].map(LABEL_COLORS).fillna('#CCCCCC')
    fig = go.Figure(data=[go.Scatter3d(
        x=sample['x'], y=sample['y'], z=sample['z'],
        mode='markers',
        marker=dict(size=1.5, color=colors, opacity=0.7),
        text=sample['semantic_label'],
        hovertemplate='<b>%{text}</b><br>x:%{x:.2f} y:%{y:.2f} z:%{z:.2f}<extra></extra>'
    )])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550
    )
    return fig


def plotly_scene_graph_2d(G):
    edge_x, edge_y = [], []
    for u, v in G.edges():
        c1 = G.nodes[u].get('centroid', [0, 0, 0])
        c2 = G.nodes[v].get('centroid', [0, 0, 0])
        edge_x += [c1[0], c2[0], None]
        edge_y += [c1[1], c2[1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='#BBBBBB', width=0.8), hoverinfo='none'
    )

    node_x, node_y, node_text, node_colors = [], [], [], []
    for node, data in G.nodes(data=True):
        c = data.get('centroid', [0, 0, 0])
        node_x.append(c[0])
        node_y.append(c[1])
        node_text.append(node)
        node_colors.append(_color(data.get('semantic_label', '')))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=12, color=node_colors, line=dict(width=1, color='white')),
        text=node_text, textposition='top center',
        textfont=dict(size=8)
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, height=520,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


def plotly_scene_graph_3d(G):
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        c1 = G.nodes[u].get('centroid', [0, 0, 0])
        c2 = G.nodes[v].get('centroid', [0, 0, 0])
        edge_x += [c1[0], c2[0], None]
        edge_y += [c1[1], c2[1], None]
        edge_z += [c1[2], c2[2], None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z, mode='lines',
        line=dict(color='#BBBBBB', width=1), hoverinfo='none'
    )

    node_x, node_y, node_z, node_text, node_colors = [], [], [], [], []
    for node, data in G.nodes(data=True):
        c = data.get('centroid', [0, 0, 0])
        node_x.append(c[0])
        node_y.append(c[1])
        node_z.append(c[2])
        node_text.append(f"{node} ({data.get('semantic_label', '')})")
        node_colors.append(_color(data.get('semantic_label', '')))

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z, mode='markers+text',
        marker=dict(size=6, color=node_colors, line=dict(width=0.5, color='white')),
        text=[n.split(' ')[0] for n in node_text],
        hovertext=node_text, hoverinfo='text'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    return fig


def plotly_processed_pointcloud(objects, max_points_per_obj=2000):
    """Show the clustered point cloud coloured by object instance + bounding boxes."""
    cmap = px.colors.qualitative.Light24
    traces = []

    for i, (obj_name, obj_data) in enumerate(objects.items()):
        pts = obj_data['points'][['x', 'y', 'z']]
        pts = pts.sample(n=min(max_points_per_obj, len(pts)), random_state=42)
        color = cmap[i % len(cmap)]

        # Point cloud per object
        traces.append(go.Scatter3d(
            x=pts['x'], y=pts['y'], z=pts['z'],
            mode='markers',
            marker=dict(size=1.5, color=color, opacity=0.7),
            name=obj_name,
            hovertemplate=f'<b>{obj_name}</b><br>({obj_data["semantic_label"]})<extra></extra>'
        ))

        # Label at centroid
        c = obj_data['centroid']
        traces.append(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]],
            mode='text',
            text=[obj_name],
            textfont=dict(size=9, color=color),
            showlegend=False, hoverinfo='none'
        ))

        # Bounding box wireframe
        mn = obj_data['bounds']['min']
        mx = obj_data['bounds']['max']
        corners = [
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mn[2]], [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]], [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]], [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]], [mn[0], mx[1], mn[2]],
        ]
        bx = [c[0] for c in corners] + [None]
        by = [c[1] for c in corners] + [None]
        bz = [c[2] for c in corners] + [None]
        traces.append(go.Scatter3d(
            x=bx, y=by, z=bz, mode='lines',
            line=dict(color=color, width=1.5),
            showlegend=False, hoverinfo='none'
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        legend=dict(title='Objects', font=dict(size=8))
    )
    return fig


# ── NEW HELPERS FOR ASSET MANAGER ─────────────────────────────

def plotly_floor_plan(objects):
    """Top-down 2D floor plan showing each object's footprint as a filled rectangle."""
    fig = go.Figure()
    legend_added = set()

    for obj_name, obj_data in objects.items():
        label = obj_data['semantic_label']
        mn = obj_data['bounds']['min']
        mx = obj_data['bounds']['max']
        c  = obj_data['centroid']
        color = _color(label)
        show_in_legend = label not in legend_added
        legend_added.add(label)

        # Filled footprint rectangle
        fig.add_trace(go.Scatter(
            x=[mn[0], mx[0], mx[0], mn[0], mn[0]],
            y=[mn[1], mn[1], mx[1], mx[1], mn[1]],
            fill='toself',
            fillcolor=color,
            line=dict(color=color, width=1),
            opacity=0.55,
            mode='lines',
            name=label.capitalize(),
            legendgroup=label,
            showlegend=show_in_legend,
            hovertemplate=(
                f'<b>{obj_name}</b><br>'
                f'Type: {label}<br>'
                f'Position: ({c[0]:.2f} m, {c[1]:.2f} m)<br>'
                f'Size: {mx[0]-mn[0]:.2f} × {mx[1]-mn[1]:.2f} m'
                '<extra></extra>'
            )
        ))

        # Short label at centroid
        short = obj_name.rsplit('_', 1)
        label_text = short[0][:3].upper() + (short[1] if len(short) > 1 else '')
        fig.add_trace(go.Scatter(
            x=[c[0]], y=[c[1]],
            mode='text',
            text=[label_text],
            textfont=dict(size=8, color='#333333'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        height=520,
        xaxis=dict(title='X (m)', scaleanchor='y', scaleratio=1, showgrid=True, gridcolor='#EEEEEE'),
        yaxis=dict(title='Y (m)', showgrid=True, gridcolor='#EEEEEE'),
        plot_bgcolor='#FAFAFA',
        legend=dict(title='Type', font=dict(size=10)),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def build_inventory_df(objects, features):
    """Build a human-readable inventory DataFrame from extracted objects + features."""
    rows = []
    for obj_name, obj_data in objects.items():
        f  = features.get(obj_name, {})
        mn = obj_data['bounds']['min']
        mx = obj_data['bounds']['max']
        c  = obj_data['centroid']
        rows.append({
            'Asset ID':   obj_name,
            'Type':       obj_data['semantic_label'].capitalize(),
            'X (m)':      round(float(c[0]), 2),
            'Y (m)':      round(float(c[1]), 2),
            'Floor Z (m)':round(float(mn[2]), 2),
            'W (m)':      round(float(mx[0] - mn[0]), 2),
            'D (m)':      round(float(mx[1] - mn[1]), 2),
            'H (m)':      round(float(f.get('height', 0)), 2),
            'Status':     'OK',
        })
    return pd.DataFrame(rows)


def plotly_asset_detail_3d(objects, selected_asset=None, max_points_per_obj=2000):
    """3D scene with optional asset highlight + camera zoom to selected asset."""
    cmap   = px.colors.qualitative.Light24
    traces = []

    for i, (obj_name, obj_data) in enumerate(objects.items()):
        is_selected = (obj_name == selected_asset)
        is_dimmed   = (selected_asset is not None and not is_selected)
        color       = cmap[i % len(cmap)]

        pts = obj_data['points'][['x', 'y', 'z']].sample(
            n=min(max_points_per_obj, len(obj_data['points'])), random_state=42
        )
        traces.append(go.Scatter3d(
            x=pts['x'], y=pts['y'], z=pts['z'],
            mode='markers',
            marker=dict(
                size=3.5 if is_selected else 1.5,
                color=color,
                opacity=0.9 if is_selected else (0.12 if is_dimmed else 0.7),
            ),
            name=obj_name,
            hovertemplate=f'<b>{obj_name}</b> ({obj_data["semantic_label"]})<extra></extra>',
        ))

        # Bounding box wireframe
        mn, mxb = obj_data['bounds']['min'], obj_data['bounds']['max']
        box_pts = [
            [mn[0],  mn[1],  mn[2]],  [mxb[0], mn[1],  mn[2]],
            [mxb[0], mxb[1], mn[2]],  [mn[0],  mxb[1], mn[2]],
            [mn[0],  mn[1],  mn[2]],  [mn[0],  mn[1],  mxb[2]],
            [mxb[0], mn[1],  mxb[2]], [mxb[0], mxb[1], mxb[2]],
            [mn[0],  mxb[1], mxb[2]], [mn[0],  mn[1],  mxb[2]],
            [mxb[0], mn[1],  mxb[2]], [mxb[0], mn[1],  mn[2]],
            [mxb[0], mxb[1], mn[2]],  [mxb[0], mxb[1], mxb[2]],
            [mn[0],  mxb[1], mxb[2]], [mn[0],  mxb[1], mn[2]],
        ]
        traces.append(go.Scatter3d(
            x=[p[0] for p in box_pts] + [None],
            y=[p[1] for p in box_pts] + [None],
            z=[p[2] for p in box_pts] + [None],
            mode='lines',
            line=dict(color=color, width=3.0 if is_selected else (0.5 if is_dimmed else 1.5)),
            opacity=1.0 if is_selected else (0.15 if is_dimmed else 0.6),
            showlegend=False, hoverinfo='none',
        ))

        # Label only for selected (or all when nothing is selected)
        if is_selected or selected_asset is None:
            cp = obj_data['centroid']
            traces.append(go.Scatter3d(
                x=[cp[0]], y=[cp[1]], z=[cp[2]],
                mode='text',
                text=[obj_name],
                textfont=dict(size=11 if is_selected else 8, color=color),
                showlegend=False, hoverinfo='none',
            ))

    # Zoom to selected asset (with 2.5 m context padding)
    if selected_asset and selected_asset in objects:
        mn, mxb = objects[selected_asset]['bounds']['min'], objects[selected_asset]['bounds']['max']
        pad = 2.5
        scene_cfg = dict(
            xaxis=dict(range=[mn[0] - pad, mxb[0] + pad], title='X'),
            yaxis=dict(range=[mn[1] - pad, mxb[1] + pad], title='Y'),
            zaxis=dict(range=[mn[2] - pad, mxb[2] + pad], title='Z'),
        )
    else:
        scene_cfg = dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=scene_cfg,
        legend=dict(title='Objects', font=dict(size=8)),
    )
    return fig


# ── APP LAYOUT ────────────────────────────────────────────────

st.set_page_config(page_title="3D Scene Graph | Asset Manager", layout="wide")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("3D Scene Graph")
    uploaded_file = st.file_uploader("Upload semantic point cloud (.csv)", type=['csv'])
    st.divider()
    view_mode = st.radio("View", ["Asset Manager", "Pipeline Explorer"], index=0)
    st.divider()
    with st.expander("Advanced Parameters"):
        eps                = st.slider("DBSCAN eps",         0.1, 2.0, 0.2,  0.05)
        min_samples        = st.slider("DBSCAN min_samples", 5,   50,  20,   1)
        distance_threshold = st.slider("Distance threshold", 0.5, 5.0, 3.0,  0.5)

if uploaded_file is None:
    st.info("Upload a CSV file in the sidebar to start.")
    st.stop()

# ── RUN PIPELINE (cached in session_state) ────────────────────
cache_key = (uploaded_file.name, eps, min_samples, distance_threshold)
if st.session_state.get('_cache_key') != cache_key:
    with st.spinner("Processing point cloud..."):
        _df            = load_semantic_point_cloud(uploaded_file)
        _objects       = extract_semantic_objects(_df, eps=eps, min_samples=min_samples)
        _features      = compute_object_features(_objects)
        _relationships = compute_spatial_relationships(_objects, distance_threshold)
        _scene_graph   = build_scene_graph(_objects, _relationships, _features)
        _validation    = validate_scene_graph(_scene_graph)
        _analysis      = analyze_scene_graph(_scene_graph)
    st.session_state.update({
        '_cache_key':    cache_key,
        '_df':           _df,
        '_objects':      _objects,
        '_features':     _features,
        '_relationships':_relationships,
        '_scene_graph':  _scene_graph,
        '_validation':   _validation,
        '_analysis':     _analysis,
    })

df            = st.session_state['_df']
objects       = st.session_state['_objects']
features      = st.session_state['_features']
relationships = st.session_state['_relationships']
scene_graph   = st.session_state['_scene_graph']
validation    = st.session_state['_validation']
analysis      = st.session_state['_analysis']

label_counts = {}
for obj_data in objects.values():
    lbl = obj_data['semantic_label']
    label_counts[lbl] = label_counts.get(lbl, 0) + 1


# ══════════════════════════════════════════════════════════════
# ASSET MANAGER VIEW
# ══════════════════════════════════════════════════════════════
if view_mode == "Asset Manager":
    st.title("Asset Inventory")
    st.caption(f"Scan: **{uploaded_file.name}** — {len(objects)} assets detected across {len(label_counts)} categories.")

    # ── Summary metrics ───────────────────────────────────────
    metric_cols = st.columns(len(label_counts) + 1)
    metric_cols[0].metric("Total Assets", len(objects))
    for col, (lbl, count) in zip(metric_cols[1:], label_counts.items()):
        col.metric(lbl.capitalize(), count)

    st.divider()

    # Initialise selection to first asset
    if 'selected_asset' not in st.session_state or st.session_state['selected_asset'] not in objects:
        st.session_state['selected_asset'] = next(iter(objects)) if objects else None

    # ── Master-detail layout ──────────────────────────────────
    left_col, right_col = st.columns([1, 2], gap="medium")

    # ── LEFT: asset browser ───────────────────────────────────
    with left_col:
        st.subheader("Assets")

        groups: Dict[str, list] = {}
        for obj_name, obj_data in objects.items():
            groups.setdefault(obj_data['semantic_label'], []).append(obj_name)

        for lbl in sorted(groups.keys()):
            asset_names    = sorted(groups[lbl])
            unit_str       = "unit" if len(asset_names) == 1 else "units"
            is_group_open  = any(st.session_state['selected_asset'] == n for n in asset_names)

            with st.expander(f"{lbl.upper()}  ·  {len(asset_names)} {unit_str}", expanded=is_group_open):
                for obj_name in asset_names:
                    obj_data = objects[obj_name]
                    f_item   = features[obj_name]
                    mn_i     = obj_data['bounds']['min']
                    mxb_i    = obj_data['bounds']['max']
                    c_i      = obj_data['centroid']
                    w_i      = round(float(mxb_i[0] - mn_i[0]), 2)
                    d_i      = round(float(mxb_i[1] - mn_i[1]), 2)
                    h_i      = round(float(f_item['height']), 2)
                    is_sel   = st.session_state['selected_asset'] == obj_name

                    with st.container(border=is_sel):
                        btn_label = f"{'▶  ' if is_sel else '      '}{obj_name}"
                        if st.button(btn_label, key=f"asset_btn_{obj_name}", use_container_width=True):
                            st.session_state['selected_asset'] = obj_name
                        st.caption(f"W {w_i} m  ·  D {d_i} m  ·  H {h_i} m")
                        st.caption(f"({c_i[0]:.2f}, {c_i[1]:.2f}, {c_i[2]:.2f})")

    # ── RIGHT: reactive 3D + detail card ─────────────────────
    with right_col:
        selected = st.session_state.get('selected_asset')

        tab_3d, tab_floor = st.tabs(["3D Scene", "Floor Plan"])
        with tab_3d:
            st.plotly_chart(plotly_asset_detail_3d(objects, selected), use_container_width=True)
        with tab_floor:
            st.plotly_chart(plotly_floor_plan(objects), use_container_width=True)

        # Detail card for selected asset
        if selected and selected in objects:
            sel_data = objects[selected]
            sel_feat = features[selected]
            mn_s     = sel_data['bounds']['min']
            mxb_s    = sel_data['bounds']['max']
            c_s      = sel_data['centroid']

            st.divider()
            st.markdown(f"**{selected}** &nbsp;—&nbsp; {sel_data['semantic_label'].capitalize()}")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Width",  f"{round(float(mxb_s[0] - mn_s[0]), 2)} m")
            d2.metric("Depth",  f"{round(float(mxb_s[1] - mn_s[1]), 2)} m")
            d3.metric("Height", f"{round(float(sel_feat['height']), 2)} m")
            d4.metric("Volume", f"{round(float(sel_feat['volume']), 2)} m³")

            p1, p2, p3 = st.columns(3)
            p1.metric("X position", f"{round(float(c_s[0]), 2)} m")
            p2.metric("Y position", f"{round(float(c_s[1]), 2)} m")
            p3.metric("Z position", f"{round(float(c_s[2]), 2)} m")

    # ── Full inventory table (collapsible) ────────────────────
    with st.expander("Full Inventory Table"):
        inv_df = build_inventory_df(objects, features)
        all_types      = sorted(inv_df['Type'].unique().tolist())
        selected_types = st.multiselect("Filter by type", all_types, default=all_types, key="inv_filter")
        st.dataframe(
            inv_df[inv_df['Type'].isin(selected_types)],
            use_container_width=True,
            column_config={
                'Asset ID':    st.column_config.TextColumn('Asset ID'),
                'Type':        st.column_config.TextColumn('Type'),
                'X (m)':       st.column_config.NumberColumn('X (m)',       format='%.2f'),
                'Y (m)':       st.column_config.NumberColumn('Y (m)',       format='%.2f'),
                'Floor Z (m)': st.column_config.NumberColumn('Floor Z (m)', format='%.2f'),
                'W (m)':       st.column_config.NumberColumn('W (m)',       format='%.2f'),
                'D (m)':       st.column_config.NumberColumn('D (m)',       format='%.2f'),
                'H (m)':       st.column_config.NumberColumn('H (m)',       format='%.2f'),
                'Status':      st.column_config.TextColumn('Status'),
            },
            hide_index=True,
        )
        st.download_button(
            "⬇ Download inventory (.csv)",
            data=inv_df.to_csv(index=False),
            file_name='asset_inventory.csv',
            mime='text/csv',
        )


# ══════════════════════════════════════════════════════════════
# PIPELINE EXPLORER VIEW
# ══════════════════════════════════════════════════════════════
else:
    st.title("🧊 3D Scene Graph Pipeline")
    st.caption("End-to-end pipeline: semantic point cloud → object extraction → scene graph → USD export.")

    # Step 1
    st.header("Step 1 · Point Cloud Preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total points",     f"{len(df):,}")
    c2.metric("Semantic classes", df['semantic_label'].nunique())
    c3.metric("Sampled points",   f"{len(df):,}")
    st.plotly_chart(plotly_pointcloud(df), use_container_width=True)

    # Step 2
    st.header("Step 2 · Object Extraction")
    cols = st.columns(len(label_counts))
    for col, (lbl, count) in zip(cols, label_counts.items()):
        col.metric(lbl.capitalize(), count)
    st.caption(f"Total objects detected: **{len(objects)}**")

    # Step 3
    st.header("Step 3 · Object Features")
    feat_rows = []
    for obj_name, f in features.items():
        feat_rows.append({
            'Object':        obj_name,
            'Label':         f['semantic_label'],
            'Volume':        round(float(f['volume']), 4),
            'Surface Area':  round(float(f['surface_area']), 4),
            'Height':        round(float(f['height']), 4),
            'Compactness':   round(float(f['compactness']), 4),
            'Point Density': round(float(f['point_density']), 2),
        })
    st.dataframe(feat_rows, use_container_width=True)

    # Step 4
    st.header("Step 4 · Spatial Relationships")
    st.caption(f"Total relationships found: **{len(relationships)}**")
    rel_counts = {}
    for _, _, rel in relationships:
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    fig_rel = px.bar(
        x=list(rel_counts.keys()), y=list(rel_counts.values()),
        labels={'x': 'Relationship', 'y': 'Count'},
        color=list(rel_counts.keys()),
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_rel.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_rel, use_container_width=True)

    # Step 5
    st.header("Step 5 · Scene Graph (2D)")
    st.plotly_chart(plotly_scene_graph_2d(scene_graph), use_container_width=True)

    # Step 6
    st.header("Step 6 · Scene Graph (3D)")
    st.plotly_chart(plotly_scene_graph_3d(scene_graph), use_container_width=True)

    # Step 7
    st.header("Step 7 · Processed Point Cloud")
    st.caption("Each object instance is coloured independently with its bounding box.")
    st.plotly_chart(plotly_processed_pointcloud(objects), use_container_width=True)

    # Step 8
    st.header("Step 8 · Summary")
    valid_label = "Valid" if validation['is_valid'] else f"{len(validation['issues'])} issue(s)"
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Nodes",      analysis['node_count'])
    s2.metric("Edges",      analysis['edge_count'])
    s3.metric("Components", analysis['connected_components'])
    s4.metric("Avg Degree", round(float(analysis['avg_degree']), 2))
    s5.metric("Validation", valid_label)

    col_a, col_b = st.columns(2)
    with col_a:
        sem = analysis['semantic_distribution']
        fig_sem = px.bar(
            x=list(sem.keys()), y=list(sem.values()),
            labels={'x': 'Label', 'y': 'Count'}, title='Semantic Distribution',
            color=list(sem.keys()), color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_sem.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_sem, use_container_width=True)

    with col_b:
        rel = analysis['relationship_types']
        fig_rel2 = px.bar(
            x=list(rel.keys()), y=list(rel.values()),
            labels={'x': 'Relationship', 'y': 'Count'}, title='Relationship Types',
            color=list(rel.keys()), color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_rel2.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_rel2, use_container_width=True)

    st.subheader("Downloads")
    summary_json = json.dumps({
        'node_count':            analysis['node_count'],
        'edge_count':            analysis['edge_count'],
        'connected_components':  analysis['connected_components'],
        'avg_degree':            float(analysis['avg_degree']),
        'semantic_distribution': analysis['semantic_distribution'],
        'relationship_types':    analysis['relationship_types'],
        'validation':            validation,
    }, indent=2)
    st.download_button(
        label="⬇ Download summary.json",
        data=summary_json,
        file_name='scene_summary.json',
        mime='application/json',
    )
