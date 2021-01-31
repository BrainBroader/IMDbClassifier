class TreeNode:

    def __init__(self, best_split, depth, cl=None, is_leaf=False):
        self.best_split = best_split
        self.depth = depth
        self.cl = cl
        self.is_leaf = is_leaf
        self.left = None
        self.right = None

    def print_tree(self):
        print('This node splits the dataset on ', self.best_split, 'attribute')
        print('The dataset on this node is class ', self.cl)
        print('This node is leaf', self.is_leaf)
        print('This node is in depth', self.depth)
        if self.left:
            self.left.print_tree()
        if self.right:
            self.right.print_tree()

    def predict_class(self, document):
        """
        Predicts the class of a given document by running a decision Tree
        Args:
            document: a given document

        Returns: the class of the document

        """

        if self.is_leaf:
            return self.cl
        else:
            if document[self.best_split]:
                return self.left.predict_class(document)
            else:
                return self.right.predict_class(document)
