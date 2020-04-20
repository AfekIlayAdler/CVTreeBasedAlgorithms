from .node import InternalNode, Leaf


def node_based_feature_importance(tree, method='gain'):
    # based on https://stats.stackexchange.com/questions/162162/relative-variable-importance-for-boosting
    feature_importance = {feature: 0 for feature in tree.column_dtypes.keys()}
    if isinstance(tree.root, Leaf):
        return feature_importance
    queue = [[tree.root]]
    tree_depth = 0
    while queue:
        level_nodes = queue.pop(0)
        next_level_nodes = []
        for node in level_nodes:
            if isinstance(node.left, InternalNode):
                next_level_nodes.append(node.left)
            if isinstance(node.right, InternalNode):
                next_level_nodes.append(node.right)
            children_nodes_weighted_purity = node.left.purity + node.right.purity
            node_mean_purity_reduction = (node.purity - children_nodes_weighted_purity) / node.n_examples
            # actually node.n_examples is not needed here.
            if method == 'gain':
                feature_importance[node.field] += (node.n_examples / tree.root.n_examples) * node_mean_purity_reduction
            if method == 'split_count':  # method == number of modes
                feature_importance[node.field] += 1
        if next_level_nodes:
            queue.append(next_level_nodes)
        tree_depth += 1
    feature_importance = {i: v for i, v in feature_importance.items()}
    return feature_importance
