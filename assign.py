# libraries
import networkx as nx

def AONloading(graph, zone2centroid, demand, compute_sptt=True):
    """
    All-or-Nothing (AON) traffic assignment with shortest path travel time (SPTT) computation.

    Parameters:
        graph (nx.DiGraph): A directed graph representing the road network, with 'cost' as the edge weight.
        zone2centroid (dict): Maps zones to lists of centroid nodes within each zone.
        demand (dict): OD demand values as a dictionary {(origin_zone, destination_zone): demand}.
        compute_sptt (bool): Flag to compute SPTT and EODTT for OD pairs.

    Returns:
        tuple: 
            - SPTT (float): Total shortest path travel time across all OD pairs.
            - x_bar (dict): Edge flow for each edge in the graph.
            - spedges (dict): Shortest paths for each OD pair (optional, if `compute_sptt` is True).
            - EODTT (dict): End-to-end travel times for OD pairs (optional, if `compute_sptt` is True).
    """
    # Initialize outputs
    x_bar = {edge: 0 for edge in graph.edges()}  # Edge flows
    spedges = {}  # Shortest paths for OD pairs
    EODTT = {}  # End-to-end travel times
    SPTT = 0  # Shortest path travel time
    
    # Iterate through origin zones and their centroid nodes
    for origin_zone, origin_nodes in zone2centroid.items():
        # Compute shortest paths from all centroids in the origin zone
        dijkstra_results = [
            nx.single_source_dijkstra(graph, origin_node, weight="cost") for origin_node in origin_nodes
        ]
        
        # Iterate through destination zones and their centroid nodes
        for destination_zone, destination_nodes in zone2centroid.items():
            if origin_zone == destination_zone:
                continue  # Skip intra-zone flows
            
            # Get the demand for the OD pair
            od_demand = demand.get((origin_zone, destination_zone), 0)
            if od_demand <= 0:
                continue  # Skip if no demand
            
            # Find the shortest path among all centroid-to-centroid paths
            shortest_paths = [
                (dijkstra_results[i][0][dest_node], dijkstra_results[i], dest_node)
                for i in range(len(dijkstra_results))
                for dest_node in destination_nodes
                if dest_node in dijkstra_results[i][0]
            ]
            
            if not shortest_paths:
                continue  # Skip if no valid path exists
            
            # Select the shortest path
            shortest_paths.sort(key=lambda x: x[0])  # Sort by travel time
            min_time, dijkstra_result, destination_node = shortest_paths[0]
            path = dijkstra_result[1][destination_node]

            # Compute SPTT and store path/EODTT if required
            if compute_sptt:
                SPTT += min_time * od_demand
                spedges[(origin_zone, destination_zone)] = path
                EODTT[(origin_zone, destination_zone)] = min_time
            
            # Update edge flows
            for u, v in zip(path[:-1], path[1:]):
                x_bar[(u, v)] += od_demand
    
    return SPTT, x_bar, spedges, EODTT

def MSA(graph, zone2centroid, demand, max_iterations=100, convergence_threshold=0.05):
    """
    Perform User Equilibrium (UE) Assignment using the Method of Successive Averages (MSA).

    Parameters:
        graph (nx.DiGraph): Directed graph representing the network, with 'cost' as edge weights.
        zone2centroid (dict): Mapping of zones to lists of centroid nodes within each zone.
        demand (dict): OD demand as a dictionary {(origin_zone, destination_zone): demand}.
        max_iterations (int): Maximum number of iterations to run.
        convergence_threshold (float): Threshold for the relative gap to check convergence.

    Returns:
        dict: Final edge flows (volumes) for each edge as {(u, v): flow}.
        list: Convergence history of relative gaps.
    """
    # Initialize edge flows and cost attributes
    for u, v, data in graph.edges(data=True):
        data['flow'] = 0  # Initialize edge flows to zero
        if 'cost' not in data:
            raise ValueError("Graph edges must have an initial 'cost' attribute.")
    
    relative_gaps = []  # To track convergence

    # Step 1: Initial All-or-Nothing (AON) assignment
    print("Running initial AON assignment...")
    SPTT, x_bar, _, _ = AONloading(graph, zone2centroid, demand, compute_sptt=True)

    for iteration in range(1, max_iterations + 1):
        print(f"Iteration {iteration}...")

        # Update flows using MSA: x = x + (1 / iteration) * (x_bar - x)
        for (u, v) in graph.edges():
            graph[u][v]['flow'] = (
                graph[u][v]['flow'] + (1 / iteration) * (x_bar.get((u, v), 0) - graph[u][v]['flow'])
            )

        # Update travel times/costs on the network based on new flows
        update_edge_costs(graph)

        # Step 2: Run shortest path (Dijkstra's algorithm) and AON loading with updated costs
        SPTT, x_bar, spedges, EODTT = AONloading(graph, zone2centroid, demand, compute_sptt=True)
        # Step 3: Compute the relative gap
        TSTT = sum(graph[u][v]['flow'] * graph[u][v]['cost'] for u, v in graph.edges())

        relative_gap = abs(TSTT - SPTT) / SPTT
        relative_gaps.append(relative_gap)
        
        print(f"Iteration {iteration}: Relative Gap = {relative_gap:.6f} TSTT = {TSTT}")

        # Check for convergence (relative gap < 0.05)
        if relative_gap < convergence_threshold:
            print("Convergence achieved!")
            break

    # Fixed return statement
    return graph, TSTT, spedges, EODTT



def update_edge_costs(graph):
    """
    Update edge travel times (costs) based on current flows using the BPR (Bureau of Public Roads) function.

    Parameters:
        graph (nx.DiGraph): Directed graph representing the network.

    Notes:
        The BPR function is defined as:
            cost = free_flow_time * (1 + alpha * (flow / capacity)^beta)
    """
    # alpha = 0.15
    # beta = 4
    for u, v, data in graph.edges(data=True):
        free_flow_time = data.get('FFT', 1)
        capacity = data.get('capacity', 1)
        flow = data['flow']
        data['cost'] = free_flow_time * (1 + data["alpha"] * (flow / capacity) ** data["beta"])
