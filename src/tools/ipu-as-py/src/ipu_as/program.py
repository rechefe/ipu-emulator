import ipu_as.line as line
import ipu_as.label as label

LABEL_SPLITTER = ":"


def strip_comments(program_str: str) -> str:
    stripped_lines = []
    for line_str in program_str.split("\n"):
        if "#" not in line_str:
            stripped_lines.append(line_str)
            continue
        line_no_comment = line_str.split("#")[0]
        stripped_lines.append(line_no_comment)
    return "\n".join(stripped_lines)


def stripped_program(program_str: str) -> list[str]:
    return [
        strip_comments(line_str)
        for line_str in program_str.split("\n")
        if strip_comments(line_str).strip()
    ]


def strip_label(line_str: str, line_address: int) -> str:
    match line_str.count(LABEL_SPLITTER):
        case 0:
            return line_str
        case 1:
            label.ipu_labels.add_label(
                line_str.split(LABEL_SPLITTER)[0].strip(),
                line_address,
            )
            if not line_str.split(LABEL_SPLITTER)[1].strip():
                raise ValueError(f"Line with only label and no instruction: {line_str}")
            return line_str.split(LABEL_SPLITTER)[1]
        case _:
            raise ValueError(f"Invalid line with multiple labels: {line_str}")


class Program:
    def __init__(self, program_str: str):
        self.lines = []
        stripped_comments_and_labels = []
        for i, file_line in enumerate(stripped_program(program_str)):
            stripped_comments_and_labels.append(strip_label(file_line, i))
        for line_str in stripped_comments_and_labels:
            try:
                self.lines.append(line.Line(line_str))
            except ValueError as e:
                print(e)
                raise ValueError(f"Invalid line in program: {line_str}") from e

    def encode(self) -> list[int]:
        return [line.encode() for line in self.lines]
