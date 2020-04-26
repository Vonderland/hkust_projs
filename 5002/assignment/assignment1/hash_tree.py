class TreeNode:
    def __init__(self):
        self.vals = []
        self.split = False
        self.left = None
        self.mid = None
        self.right = None

def add_data_to_tree(d, node, level):
    # has children, this node has been split.
    # the first level always need split
    # print(d, level)
    if node.split:
        if d[level] in left:
            # print("left")
            if node.left is None:
                node.left = TreeNode()
                node.left.vals.append(d)
            else:
                add_data_to_tree(d, node.left, level + 1)
        if d[level] in right:
            # print("right")
            if node.right is None:
                node.right = TreeNode()
                node.right.vals.append(d)
            else:
                add_data_to_tree(d, node.right, level + 1)
        if d[level] in mid:
            # print("mid")
            if node.mid is None:
                node.mid = TreeNode()
                node.mid.vals.append(d)
            else:
                add_data_to_tree(d, node.mid, level + 1)
    else:
        # the last level should not limit the amount of data
        if level == 3:
            # print("last", d)
            node.vals.append(d)
        else:
            # add current data
            # print("add to this")
            node.vals.append(d)
            if len(node.vals) > max_leaf_size:
                node.split = True
                for val in node.vals:
                    add_data_to_tree(val, node, level)
                # release the space
                node.vals.clear()


def build_hash_tree(data):
    node = TreeNode()
    # the root should always split
    node.split = True
    for d in data:
        add_data_to_tree(d, node, 0)
    return node


def get_tree_hierarchy(root):
    if not root:
        return

    if not root.split:
        if len(root.vals) == 1:
            return root.vals[0]
        return root.vals

    hierarchy = []
    l = get_tree_hierarchy(root.left)
    if l:
        hierarchy.append(l)
    m = get_tree_hierarchy(root.mid)
    if m:
        hierarchy.append(m)
    r = get_tree_hierarchy(root.right)
    if r:
        hierarchy.append(r)
    return hierarchy


data = [[1,2,3], [1,3,9], [1,4,5], [1,4,6], [1,5,7], [1,5,9], [1,6,8], [1,6,9], [1,8,9],
        [2,3,9], [2,5,6], [2,5,7], [2,5,9], [2,6,7], [2,6,8], [2,6,9], [2,7,8], [2,7,9], [2,8,9],
        [3,4,6], [3,4,8], [3,7,8],
        [4,5,6], [4,5,8], [4,5,9], [4,7,8], [4,7,9], [4,8,9],
        [5,6,7], [5,6,8], [5,7,8], [5,7,9], [5,8,9],
        [6,7,9], [6,8,9],
        [7,8,9]]


left = [1, 3, 7]
mid = [2, 4, 8]
right = [5, 6, 9]

max_leaf_size = 3
root = build_hash_tree(data)
hierarchy = get_tree_hierarchy(root)

print(hierarchy)