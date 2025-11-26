class Labels:
    def __init__(self):
        self.labels = {}

    def add_label(self, label: str, address: int):
        if label in self.labels:
            raise ValueError(f"Label '{label}' already defined.")
        self.labels[label] = address

    def get_address(self, label: str) -> int:
        if label not in self.labels:
            raise ValueError(f"Label '{label}' not defined.")
        return self.labels[label]


ipu_labels = Labels()


def reset_labels():
    global ipu_labels
    ipu_labels = Labels()
