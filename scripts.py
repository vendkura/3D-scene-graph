def load_semantic_point_cloud(file_path, column_name = 'semantic_label'):
    """Load semantic point cloud data from ASCII formats."""
        
    df = pd.read_csv(file_path, delimiter=';')
    class_names = ['ceiling', 'floor', 'wall', 'chair', 'furniture', 'table']
    
    # Assuming the numerical labels are 0.0, 1.0, 2.0, ...
    label_map = {float(i): class_names[i] for i in range(len(class_names))}
    
    df[column_name] = df[column_name].map(label_map)
    
    # I sample here for replication goals
    return df.sample(n=150000, random_state=1)

raw_data = load_semantic_point_cloud('data/indoor_room_labelled.csv')
demo_data = raw_data.sample(n=10000, random_state=1)
