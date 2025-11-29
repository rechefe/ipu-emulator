import lark

class Labels:
    def __init__(self):
        self.labels = {}

    def add_label(self, token: any):
        if token.token.value in self.labels:
            print(token)
            existing_token = self.labels[token.token.value].token
            raise ValueError(
                f"Label '{token.token.value}' is defined for the second time at Line {token.token.line}, Column {token.token.column}. "
                f"Previous definition at Line {existing_token.line}, Column {existing_token.column}."
            )
        self.labels[token.token.value] = token

    def get_address(self, token: lark.Token) -> int:
        if token.value not in self.labels:
            raise ValueError(
                f"Label '{token.value}' referenced in Line {token.line}, Column {token.column} not defined."
            )
        return self.labels[token.value].instr_id


ipu_labels = Labels()


def reset_labels():
    global ipu_labels
    ipu_labels = Labels()
