class TreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None


def get_new_trans(node):
    new_trans = {}
    # get the parent paths of node, and return as new transactions
    while node:
        cur_node = node
        path = []
        while cur_node.parent:
            path.append(cur_node.item)
            cur_node = cur_node.parent
        # the support of the new_trans is decided by the count of node
        if len(path) > 1:
            new_trans[frozenset(path[1:])] = node.count

        node = node.next
    return new_trans


def create_head_table(item_set):
    head_table = {}
    for trans in item_set:
        count = item_set[trans]

        # calculate the counts of each item
        for j in range(count):
            for item in trans:
                if not head_table.get(item):
                    head_table[item] = 0
                head_table[item] += 1

    for key in set(head_table.keys()):
        # remove items according to min_support
        if head_table[key] < min_support:
            head_table.pop(key)

    for key in head_table:
        # store the counts and the head node in the head table
        # the first head node is None before inited
        head_table[key] = [head_table[key], None]
    return head_table


def mine_frequent_itemsets(head_table, freq_itemset, root_set, pre_fix):
    for item in head_table.keys():
        cur_prefix = pre_fix.copy()
        cur_prefix.add(item)
        freq_itemset.append((cur_prefix, head_table[item][0]))
        # get the new transactions
        new_trans = get_new_trans(head_table[item][1])
        new_root, new_head = build_tree(new_trans)
        root_set.append(new_root)
        # mine on the sub trees
        if new_head:
            mine_frequent_itemsets(new_head, freq_itemset, root_set, cur_prefix)


def build_tree(data):
    root = TreeNode(None, 1, None)
    head_table = create_head_table(data)
    for trans, count in data.items():
        freq_items = {}
        for item in trans:
            # if item is in head_table, it's support is larger than min_sup
            if item in head_table:
                # store the counts of the items for sorting
                freq_items[item] = head_table[item][0]
        # this transaction contains frequent item
        if len(freq_items):
            sorted_items = sorted(freq_items, key=freq_items.get, reverse=True)
            # add this transaction to the tree
            insert_to_tree(sorted_items, head_table, root, count)
    return root, head_table


def insert_to_tree(item_list, head_table, root, count):
    # the node is a child of root, so add the count directly
    if item_list[0] in root.children:
        root.children[item_list[0]].count += count
    # not a child of root, so add it to the children
    else:
        root.children[item_list[0]] = TreeNode(item_list[0], count, root)
        if not head_table[item_list[0]][1]:
            # it is the first node in the corresponding pos of head table
            head_table[item_list[0]][1] = root.children[item_list[0]]
        else:
            # link this node in the head table
            node = head_table[item_list[0]][1]
            while node.next:
                node = node.next
            node.next = root.children[item_list[0]]
    # add the rest nodes recursively
    if len(item_list) > 1:
        insert_to_tree(item_list[1:], head_table, root.children[item_list[0]], count)
    return head_table


def load_topic(topic):
    data_set = []
    with open('./data/topic-{}.txt'.format(topic), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data_set.append([x for x in line.split()])
    item_set = {}
    for row in data_set:
        row = frozenset(i for i in row)
        item_set[row] = item_set.get(row, 0) + 1
    return item_set


def load_vocab():
    key_map = {}
    with open('./data/vocab.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.split()
            key_map[key] = value
    return key_map


def save_result(topic, freq_itemset):
    itemset_list = []
    for item in freq_itemset:
        name_set = []
        item_set, count = item
        for i in item_set:
            name_set.append(vocab[i])
        itemset_list.append((count, name_set))
    itemset_list.sort(key=lambda x: x[0], reverse=True)

    with open('./data/pattern-{}.txt'.format(topic), 'w', encoding='utf-8') as f:
        for item in itemset_list:
            count, itemset = item
            item_str = " ".join(itemset)
            f.writelines(str(count) + "\t" + item_str)
            f.write('\n')
        f.close()


def get_tree_hierarchy(node):
    hierarchy = []
    if node:
        item_name = 'Null Set' if not node.item else vocab[node.item]
        hierarchy.append(item_name + '    ' + str(node.count))
        if node.children:
            for child in node.children:
                hierarchy.append(get_tree_hierarchy(node.children[child]))
    return hierarchy


min_support = 400

for i in range(5):
    print("processing topic-{} start".format(i))

    trans = load_topic(i)

    root, head_table = build_tree(trans)

    frequent_itemset = []
    roots = []
    mine_frequent_itemsets(head_table, frequent_itemset, roots, set())

    vocab = load_vocab()
    save_result(i, frequent_itemset)

    for root in roots:
        hierarchy = get_tree_hierarchy(root)

        if len(hierarchy) > 2:
            print(hierarchy)
    print("processing topic-{} end".format(i))