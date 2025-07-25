def reorder_traces_by_sparse_mult_cost(tn, reverse=False):
    """
    Reorders traces by estimated sparse multiplication cost of node.h matrices.
    """
    trace_scores = []
    for trace in tn.traces:
        node_idx1, node_idx2, *_ = trace
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)

        cost = estimate_sparse_mult_cost(h1, h2)
        trace_scores.append((cost, trace))

    trace_scores.sort(key=lambda x: x[0], reverse=reverse)
    tn.traces = [t for (_, t) in trace_scores]


def get_weighted_avg_sparsity(tn):
    sparsity_info = []
    for node_idx1, node_idx2, join_legs1, join_legs2 in tn.traces:
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)
        sparsity_info.append(estimate_sparse_mult_cost(h1, h2))

    return round(sum((i+1) * w for i, w in enumerate(sparsity_info)), 3)


def get_weighted_avg(tn):
    sparsity_info = []
    for node_idx1, node_idx2, join_legs1, join_legs2 in tn.traces:
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)
        sparsity_info.append(np.count_nonzero(h1) + np.count_nonzero(h2))

    return sum((i+1) * w for i, w in enumerate(sparsity_info))


def run_wep_for_sorted_traces(coloring, d, num_runs=1):
    for _ in range(num_runs):
        compass_code = CompassCode(d, coloring)

        tn = compass_code.concatenated()

        weighted_avg = get_weighted_avg(tn)
        sparsity = get_weighted_avg_sparsity(tn)
        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count, score_cotengra, score_custom, flops, write, pcm_cost = get_contraction_time(tn, False)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([coloring, d, contraction_time, "default", weighted_avg, sparsity])


        tn = compass_code.concatenated()
        reorder_traces_by_sparse_mult_cost(tn)
        weighted_avg = get_weighted_avg(tn)
        sparsity = get_weighted_avg_sparsity(tn)
        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = get_contraction_time(tn, False)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([coloring, d, contraction_time, "sorted", weighted_avg, sparsity])

        # Run wep calculation for reverse sorted pcm sparsity
        tn = compass_code.concatenated()
        
        reorder_traces_by_sparse_mult_cost(tn, reverse=True)
        weighted_avg = get_weighted_avg(tn)
        sparsity = get_weighted_avg_sparsity(tn)
        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = get_contraction_time(tn, False)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([coloring, d, contraction_time, "reverse sorted", weighted_avg, sparsity])

        # Run wep calculation for randomly sorted pcm sparsity
        tn = compass_code.concatenated()
        np.random.shuffle(tn.traces)
        weighted_avg = get_weighted_avg(tn)
        sparsity = get_weighted_avg_sparsity(tn)
        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = get_contraction_time(tn, False)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([coloring, d, contraction_time, "random", weighted_avg, sparsity])

        # Run wep calculation with cotengra optimization
        tn = compass_code.concatenated()
        
        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = get_contraction_time(tn, True)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('trace_ordering_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([coloring, d, contraction_time, "cotengra", -1, -1])

    

def run_wep_for_many_traces(coloring, d):
    compass_code = CompassCode(d, coloring)

    #for name, rep in compass_code.get_representations().items():
    for _ in range(30):
        tn = compass_code.concatenated()
        np.random.shuffle(tn.traces)

        # need to loop through the traces and analyze cost somehow
        sparsity_info = []
        for node_idx1, node_idx2, join_legs1, join_legs2 in tn.traces:
            h1 = np.array(tn.nodes[node_idx1].h)
            h2 = np.array(tn.nodes[node_idx2].h)
            sparsity_info.append(np.count_nonzero(h1) + np.count_nonzero(h2))

        numerator = sum((i+1) * w for i, w in enumerate(sparsity_info))
        weighted_avg = round(numerator / len(sparsity_info), 3)

        contraction_time, wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = get_contraction_time(tn)
        avg_intermediate_tensor_size = round(np.mean(intermediate_tensor_sizes), 3)

        with open('sparsity_tests.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([d, "concatenated", round(contraction_time, 3), contraction_cost, contraction_width, weighted_avg, avg_intermediate_tensor_size, total_ops_count])


def reorder_traces_by_sparsity(tn, reverse=False):
    """
    Compute sparsity = (nnz of node1.h) + (nnz of node2.h) for each trace,
    then sort tn.traces ascending by that sum and assign back.
    """
    trace_scores = []
    for trace in tn.traces:
        node_idx1, node_idx2, join_legs1, join_legs2 = trace
        h1 = np.array(tn.nodes[node_idx1].h)
        h2 = np.array(tn.nodes[node_idx2].h)
        sparsity = np.count_nonzero(h1) + np.count_nonzero(h2)
        trace_scores.append((sparsity, trace))

    # sort by the sparsity (the first element of each tuple)
    with open('test.txt', 'a') as f:
        f.write("trace scores:\n")
        f.write(str(trace_scores) + "\n")

    trace_scores.sort(key=lambda x: x[0], reverse=reverse)

    # extract just the traces in sorted order
    tn.traces = [t for (_, t) in trace_scores]