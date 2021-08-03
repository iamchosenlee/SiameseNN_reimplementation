from treelib import Node, Tree
import os
from collections import defaultdict

root_dir = './dataset/images_background/'
data_dict = defaultdict(dict) #data_dict[alphabet][character] = [list of images]
for alp in sorted(os.listdir(root_dir)):
    for char in sorted(os.listdir(root_dir + alp)):
        data_dict[alp][char] = os.listdir(os.path.join(root_dir, alp, char))

def filter_nodes(tree, node_list, idxs):
    nodes_to_delete = set(node_list)-set([node_list[i] for i in idxs])
    for node in nodes_to_delete:
        tree.remove_node(node)

def build_recursive_tree(tree, base, depth, root_dir):
    if depth >0:
        for curr in sorted(os.listdir(root_dir)):
            tree.create_node("{}".format(curr), "{}".format(str(base.identifier)+'/'+str(curr)),
                             parent=base.identifier)
            newbase = tree.get_node(str(base.identifier)+'/'+str(curr))
            build_recursive_tree(tree, newbase, depth-1, root_dir+'/'+curr)

    else:
        return


def build_data_tree(root_dir, depth=3):
    data_tree = Tree()
    base = data_tree.create_node(root_dir, root_dir)
    #data_dict = defaultdict(dict) #data_dict[alphabet][character] = [list of images]
    #for alp in os.listdir(root_dir):
    #    for char in os.listdir(root_dir + alp):
    #        data_dict[alp][char] = os.listdir(os.path.join(root_dir, alp, char))
    build_recursive_tree(data_tree, base, depth, root_dir)
    return data_tree

# def build_recursive_tree_old(tree, base, depth, data_dict):
#     """
#     Args:
#         tree: Tree with 
#         base: Node
#         depth: int
#         data_dict: #data_dict[alphabet][character] = [list of images]
#     Returns: Tree
#     """
#     if type(data_dict) == list:
#         for drawer in data_dict:
#             tree.create_node("{}".format(drawer), "{}".format(os.path.join(str(base.identifier), str(drawer))),
#                              parent=base.identifier)
#     elif depth >1:
#         for alp in data_dict.keys():
#             tree.create_node("{}".format(alp), "{}".format(os.path.join(str(base.identifier), str(alp))),
#                              parent=base.identifier)
#             newbase = tree.get_node(os.path.join(str(base.identifier), str(alp)))
#             build_recursive_tree(tree, newbase, depth-1, data_dict[alp])

#     else:
#         return


import pickle
tree = build_data_tree(root_dir)
# tree.show()
# with open("tree.pkl", 'wb') as f:
#     pickle.dump(tree, f)

#print(len(tree.leaves()))


#print(tree.leaves()[:2])
#print(tree.leaves()[1].bpointer, tree.leaves()[2].bpointer)
#print(type(tree.get_node(tree.root).fpointer))
#print(tree.children(tree.root)[0].tag)
#tag = tree.children(tree.root)[0]
#tree.remove_node(tag.identifier)
#print(tree[tree[root_dir].fpointer[2]].fpointer)
#data_dict = tree.to_dict()
#print(data_dict['./dataset/images_background/']['Alphabet_of_the_Magi']['character15'])