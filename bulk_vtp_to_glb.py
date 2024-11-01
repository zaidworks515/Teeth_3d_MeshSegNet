import pyvista as pv
import numpy as np
import os


def load_vtp_file(vtp_path):
    """Load a 3D mesh from a .vtp file."""
    return pv.read(vtp_path)

def generate_color_map(num_instances):
    """Generate a fixed color map based on the number of instances."""
    color_map = [
        [0.8, 0.8, 0.8],    # Off White
        [1.0, 0.0, 0.0],       # Red
        [0.0, 1.0, 0.0],       # Green
        [0.0, 0.0, 1.0],       # Blue
        [1.0, 1.0, 0.0],       # Yellow
        [1.0, 0.0, 1.0],       # Magenta
        [0.0, 1.0, 1.0],       # Cyan
        [0.5, 0.5, 0.5],       # Gray
        [0.5, 0.0, 0.0],       # Maroon
        [0.0, 0.5, 0.0],       # Dark Green
        [0.0, 0.0, 0.5],       # Navy
        [0.5, 0.5, 0.0],       # Olive
        [0.5, 0.0, 0.5],       # Purple
        [0.0, 0.5, 0.5],       # Teal
        [0.8, 0.6, 0.4],       # Brown
        [0.4, 0.8, 0.6],       # Light Green
        [0.6, 0.4, 0.8]        # Lavender
    ]
    
    if num_instances > len(color_map):
        raise ValueError(f"Not enough predefined colors for {num_instances} instances. Add more colors.")

    return color_map[:num_instances]


def mesh_export(mesh, output_gltf, filename, base_path='outputs/'):
    """Visualize the 3D mesh and save it as a GLTF file."""
    if 'Label' not in mesh.cell_data.keys():
        raise ValueError("No 'Label' data found in cell data.")
    
    labels = mesh.cell_data['Label']
    
    unique_labels = np.unique(labels)
    print(f"NO OF UNIQUE LABELS::: {unique_labels} ")
    
    color_map = generate_color_map(len(unique_labels))
    label_to_color = {label: color_map[idx] for idx, label in enumerate(unique_labels)}
    
    cell_colors = np.array([label_to_color[label] for label in labels])
    mesh.cell_data['Colors'] = cell_colors
    
    mesh = mesh.cell_data_to_point_data()

    pl = pv.Plotter()
    pl.add_mesh(mesh, scalars='Colors', rgb=True, show_edges=False)
    pl.add_title("Segmented Mesh Visualization")

    pl.export_gltf(output_gltf)
    print(f"Saved the visualization as GLTF to {output_gltf}")

base_path='outputs'
files = os.listdir(base_path)
output_dir = 'glb outputs/'

os.makedirs(output_dir, exist_ok=True)

for f in files:
    if f.endswith('u_predicted_refined.vtp') or f.endswith('l_predicted_refined.vtp'):
        vtp_path = os.path.join(base_path, f)
        mesh = load_vtp_file(vtp_path)  
        filename = os.path.join(output_dir, f.replace('vtp', 'glb'))
        mesh_export(, mesh=mesh, output_gltf=filename)

print("ALL FILES ARE CONVERTED INTO GLB FORMAT WITH SEGMENTED COLORS")
