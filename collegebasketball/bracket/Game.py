


class Game:
    def __init__(self, left, right, winner):

        self.left = left
        self.right = right
        self.winner = winner
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.left_pred = None
        self.right_pred = None
        self.pred = None
        self.depth

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child, type):
        if type == 'left':
            self.left_child = child
        else:
            self.right_child = child

    def add_pred(self, team, type):
        if type == 'left':
            self.left_pred = team
        else:
            self.right_pred = team

    def predict_winner(self, winner):
        self.winner = winner