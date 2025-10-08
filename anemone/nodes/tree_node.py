"""
This module defines the TreeNode class, which represents a node in a tree structure for a chess game.
"""

from dataclasses import dataclass, field

from anemone.basics import BranchKey, BranchKeyGeneratorP, NodeContent, NodeTag

from .itree_node import ITreeNode

# todo replace the any with a defaut value in ITReenode when availble in python; 3.13?


@dataclass(slots=True)
class TreeNode[ChildrenType: ITreeNode = ITreeNode]:
    r"""
    The TreeNode class stores information about a specific board position, including the board representation,
    the player to move, the half-move count, and the parent-child relationships with other nodes.

    Attributes:
        id\_ (int): The number to identify this node for easier debugging.
        half_move\_ (int): The number of half-moves since the start of the game to reach the board position.
        board\_ (boards.BoardChi): The board representation of the node.
        parent_nodes\_ (set[ITreeNode]): The set of parent nodes to this node.
        all_legal_moves_generated (bool): A boolean indicating whether all moves have been generated.
        non_opened_legal_moves (set[chess.Move]): The set of non-opened legal moves.
        moves_children\_ (dict[chess.Move, ITreeNode | None]): The dictionary mapping moves to child nodes.
        fast_rep (str): The fast representation of the board.
        player_to_move\_ (chess.Color): The color of the player that has to move in the board.

    Methods:
        __post_init__(): Initializes the TreeNode object after it has been created.
        id(): Returns the id of the node.
        player_to_move(): Returns the color of the player to move.
        board(): Returns the board representation.
        half_move(): Returns the number of half-moves.
        moves_children(): Returns the dictionary mapping moves to child nodes.
        parent_nodes(): Returns the set of parent nodes.
        is_root_node(): Checks if the node is a root node.
        legal_moves(): Returns the legal moves of the board.
        add_parent(new_parent_node: ITreeNode): Adds a parent node to the current node.
        is_over(): Checks if the game is over.
        print_moves_children(): Prints the moves-children links of the node.
        test(): Performs a test on the node.
        dot_description(): Returns the dot description of the node.
        test_all_legal_moves_generated(): Tests if all legal moves have been generated.
        get_descendants(): Returns a dictionary of descendants of the node.
    """

    # id is a number to identify this node for easier debug
    id_: int

    # the tree depth of this node
    tree_depth_: int

    # the node holds a content.
    content_: NodeContent

    # the set of parent nodes to this node. Note that a node can have multiple parents!
    parent_nodes_: dict[ITreeNode[ChildrenType], BranchKey]

    # all_branches_generated is a boolean saying whether all branches have been generated.
    # If true the branches are either opened in which case the corresponding opened node is stored in
    # the dictionary self.branches_children, otherwise it is stored in self.non_opened_branches
    all_branches_generated: bool = False
    non_opened_branches: set[BranchKey] = field(
        default_factory=lambda: set[BranchKey]()
    )

    # dictionary mapping moves to children nodes. Node is set to None if not created
    branches_children_: dict[BranchKey, ChildrenType | None] = field(
        default_factory=lambda: dict[BranchKey, ChildrenType | None]()
    )

    @property
    def tag(self) -> NodeTag:
        """Returns the fast representation of the board.

        Returns:
            boards.boardKey: The fast representation of the board.
        """
        return self.content_.tag

    @property
    def id(self) -> int:
        """
        Returns the ID of the tree node.

        Returns:
            int: The ID of the tree node.
        """
        return self.id_

    @property
    def content(self) -> NodeContent:
        """
        Returns the content associated with this tree node.

        Returns:
            NodeContent: The content associated with this tree node.
        """
        return self.content_

    @property
    def tree_depth(self) -> int:
        """
        Returns the tree depth of this node.

        Returns:
            int: The tree depth of this node.
        """
        return self.tree_depth_

    @property
    def branches_children(self) -> dict[BranchKey, ChildrenType | None]:
        """
        Returns a bidirectional dictionary containing the children nodes of the current tree node,
        along with the corresponding chess moves that lead to each child node.

        Returns:
            dict[BranchKey, ITreeNode | None]: A bidirectional dictionary mapping branches to
            the corresponding child nodes. If a branch does not have a corresponding child node, it is
            mapped to None.
        """
        return self.branches_children_

    @property
    def parent_nodes(self) -> dict[ITreeNode[ChildrenType], BranchKey]:
        """
        Returns the dictionary of parent nodes of the current tree node with associated move.

        :return: A dictionary of parent nodes of the current tree node with associated move.
        """
        return self.parent_nodes_

    def is_root_node(self) -> bool:
        """
        Check if the current node is a root node.

        Returns:
            bool: True if the node is a root node, False otherwise.
        """
        return not self.parent_nodes

    @property
    def all_branches_keys(self) -> BranchKeyGeneratorP:
        """
        Returns a generator that yields the branch keys for the current board state.

        Returns:
            BranchKeyGenerator: A generator that yields the branch keys.
        """
        return self.content_.branch_keys

    def add_parent(
        self, branch_key: BranchKey, new_parent_node: ITreeNode[ChildrenType]
    ) -> None:
        """
        Adds a new parent node to the current node.

        Args:
            branch_key (BranchKey): The branch key associated with the move that led to the node from the new_parent_node.
            new_parent_node (ITreeNode): The new parent node to be added.

        Raises:
            AssertionError: If the new parent node is already in the parent nodes set.

        Returns:
            None
        """
        # debug
        assert (
            new_parent_node not in self.parent_nodes
        )  # there cannot be two ways to link the same child-parent
        self.parent_nodes[new_parent_node] = branch_key

    def is_over(self) -> bool:
        """
        Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def print_moves_children(self) -> None:
        """
        Prints the moves-children link of the node.

        This method prints the moves-children link of the node, showing the move and the ID of the child node.
        If a child node is None, it will be displayed as 'None'.

        Returns:
            None
        """
        print(
            "here are the ",
            len(self.branches_children_),
            " branches-children link of node",
            self.id,
            ": ",
            end=" ",
        )
        for branch, child in self.branches_children_.items():
            if child is None:
                print(branch, child, end=" ")
            else:
                print(branch, child.id, end=" ")
        print(" ")

    def dot_description(self) -> str:
        """
        Returns a string representation of the node in the DOT format.

        The string includes the node's ID, half move, and board FEN.

        Returns:
            A string representation of the node in the DOT format.
        """
        return (
            "id:"
            + str(self.id)
            + " dep: "
            + str(self.tree_depth)
            + "\nfen:"
            + str(self.board)
        )
