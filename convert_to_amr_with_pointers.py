import json
import re
import sys

node_name_matcher = re.compile(r"^[a-z]{1,3}([1-9][0-9]*)?$")

def to_amr_with_pointer(amr: str):
    result = ""
    status = "find_first_left"
    level = 0
    node_name_to_pointer_map: dict[str, str] = {}
    unresolved_node_names = set()
    next_pointer_id = 0
    current_token = ""
    for c in amr:
        if status == "find_first_left":
            if c == "(":
                result += "( "
                level += 1
                status = "find_begin_of_new_node_name"
            # else: ignore
                
        elif status == "find_begin_of_new_node_name":
            if c in "abcdefghijklmnopqrstuvwxyz":
                current_token = c
                status = "find_end_of_new_node_name"
            elif not c.isspace():
                raise ValueError(f"Unexpected begin of node name: \"{c}\"")
            # else: c is a space; ignore

        elif status == "find_end_of_new_node_name":
            if c in "abcdefghijklmnopqrstuvwxyz-0123456789":
                current_token += c
            elif c.isspace() or c == "/":
                if is_node_name(current_token):
                    node_name = current_token
                    if node_name in node_name_to_pointer_map:
                        if node_name in unresolved_node_names:
                            pointer = node_name_to_pointer_map[node_name]
                            unresolved_node_names.remove(node_name)
                        else:
                            raise ValueError(f"Duplicate node name: {node_name}")
                    else:
                        pointer = f"<pointer:{next_pointer_id}>"
                        next_pointer_id += 1
                        node_name_to_pointer_map[node_name] = pointer
                    
                    result += f"{pointer} "

                    if c != "/":
                        status = "find_slash"
                    else:
                        status = "find_begin_of_concept"
                else:
                    raise ValueError(f"Unexpected node name: \"{current_token}\"")

            else:
                raise ValueError(f"Unexpected char of node name: \"{c}\"")
            
        elif status == "find_slash":
            if c == "/":
                status = "find_begin_of_concept"

            elif not c.isspace():
                raise ValueError(f"Expecting slash, got \"{c}\"")
            # else: ignore

        elif status == "find_begin_of_concept":
            if c in "abcdefghijklmnopqrstuvwxyz":
                current_token = c
                status = "find_end_of_concept"
            elif not c.isspace():
                raise ValueError(f"Unexpected begin of concept: \"{c}\"")
            # else: c is a space; ignore

        elif status == "find_end_of_concept":
            if c in "abcdefghijklmnopqrstuvwxyz-0123456789":
                current_token += c
            elif c.isspace() or c == ")":
                result += f"{current_token}"
                if c != ")":
                    status = "find_right_or_begin_of_relation"
                else:
                    level -= 1
                    result += " )"
                    if level == 0:
                        status = "end"
                    else:
                        status = "find_right_or_begin_of_relation"

            else:
                raise ValueError(f"Unexpected char of concept: \"{c}\"")

        elif status == "find_right_or_begin_of_relation":
            if c == ")":
                level -= 1
                result += " )"
                if level == 0:
                    status = "end"
                # else: keep the status
            
            elif c == ":":
                current_token = c
                status = "find_end_of_relation"
        
            elif not c.isspace():
                raise ValueError(f"Expecting right parenthesis or begin of relation, got \"{c}\"")
            
            # else: c is space; ignore
        
        elif status == "find_end_of_relation":
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789":
                current_token += c
            elif c.isspace() or c == "(" or c == "\"":
                result += f" {current_token} "
                if c == "(":
                    result += "( "
                    level += 1
                    status = "find_begin_of_new_node_name"
                elif c == "\"":
                    result += c
                    status = "find_end_of_literal_value"
                else:
                    status = "find_left_or_begin_of_value"
            else:
                raise ValueError(f"Unexpected char of relation: \"{c}\"")
            
        elif status == "find_left_or_begin_of_value":
            if c == "(":
                result += "( "
                level += 1
                status = "find_begin_of_new_node_name"
            
            elif c in "abcdefghijklmnopqrstuvwxyz+-0123456789":
                # It can be a node name or non-literal constant.
                current_token = c
                status = "find_end_of_non_literal_value"

            elif c == "\"":
                result += c
                status = "find_end_of_literal_value"

            elif not c.isspace():
                raise ValueError(f"Expecting left parenthesis or begin of value, got \"{c}\"")

            # else: ignore
            
        elif status == "find_end_of_non_literal_value":
            if c in "abcdefghijklmnopqrstuvwxyz-0123456789":
                current_token += c
            elif c.isspace() or c == ")":
                if is_node_name(current_token):
                    node_name = current_token
                    if node_name in node_name_to_pointer_map:
                        pointer = node_name_to_pointer_map[node_name]
                    else:
                        pointer = f"<pointer:{next_pointer_id}>"
                        next_pointer_id += 1
                        node_name_to_pointer_map[node_name] = pointer
                        unresolved_node_names.add(node_name)
                    
                    result += f"{pointer}"
                    
                else:
                    result += f"{current_token}"

                if c != ")":
                    status = "find_right_or_begin_of_relation"
                else:
                    level -= 1
                    result += " )"
                    if level == 0:
                        status = "end"
                    else:
                        status = "find_right_or_begin_of_relation"

            else:
                raise ValueError(f"Unexpected char of node name or concept: \"{c}\"")
            
        elif status == "find_end_of_literal_value":
            result += c
            if c == "\"":
                status = "find_right_or_begin_of_relation"

        elif status == "end":
            if not c.isspace():
                raise ValueError(f"Expecting end, got {c}")

        else:
            raise ValueError(f"Unexpected status: {status}")
        
    if status != "end":
        raise ValueError(f"Unexpected end status: {status}")
    
    if len(unresolved_node_names) > 0:
        raise ValueError(f"Unresolved node names: {unresolved_node_names}")
        
    return result

def is_node_name(current_token):
    return node_name_matcher.match(current_token) is not None

SENT_PREFIX = "# ::snt "
def sent_amr_iter(path: str):
    sent_offset = len(SENT_PREFIX)

    with open(path) as fp:
        status = "find_non_empty_line"
        current_amr = ""
        current_sent = ""

        for line in fp:
            line = line.strip()
            match status:
                case "find_non_empty_line":
                    if line == "":
                        continue

                    if line.startswith("("):
                        current_sent = ""
                        current_amr = line
                        status = "select_amr_until_blank_line"
                    else:
                        if line.startswith(SENT_PREFIX):
                            current_sent = line[sent_offset:].strip()
                        status = "find_end_of_header"

                case "find_end_of_header":
                    if line.startswith("("):
                        current_amr = line
                        status = "select_amr_until_blank_line"
                    elif line == "":
                        status = "find_begin_of_amr"
                    elif line.startswith(SENT_PREFIX):
                        if current_sent != "":
                            yield current_sent, "", "", "AMR is empty"
                        current_sent = line[sent_offset:].strip()
                    # else: ignore

                case "find_begin_of_amr":
                    if line == "":
                        continue
                    
                    if line.startswith("("):
                        current_amr = line
                        status = "select_amr_until_blank_line"
                    else:
                        yield current_sent, "", "", "AMR is empty"
                        current_sent = ""
                        status = "find_end_of_header"
                    
                case "select_amr_until_blank_line":
                    if line == "" or line.startswith("#"):
                        try:
                            amr_with_pointer = to_amr_with_pointer(current_amr)
                            if current_sent != "":
                                yield current_sent, current_amr, amr_with_pointer, ""
                            else:
                                yield current_sent, current_amr, amr_with_pointer, "Sentence is empty"
                        except ValueError as e:
                            yield current_sent, current_amr, "", str(e)
                    
                        current_sent = ""
                        current_amr = ""
                        if line.startswith("#"):
                            status = "find_end_of_header"
                        else:
                            status = "find_non_empty_line"

                    elif current_amr == "":
                        current_amr = line
                        
                    else:
                        current_amr += " " + line
            
        if status == "select_amr_until_blank_line":
            try:
                amr_with_pointer = to_amr_with_pointer(current_amr)
                if current_sent != "":
                    yield current_sent, current_amr, amr_with_pointer, ""
                else:
                    yield current_sent, current_amr, amr_with_pointer, "Sentence is empty"
            except ValueError as e:
                yield current_sent, current_amr, "", str(e)

def to_jsonl_dataset_2(input_path: str, output_path: str):
    error_count = 0
    with open(output_path, mode="w") as fp_out:
        for sent, raw_amr, amr_with_pointer, error_message in sent_amr_iter(input_path):
            if error_message == "":
                print(
                    json.dumps({"sent": sent, "amr": amr_with_pointer, "lang": "id"}),
                    file=fp_out
                )
            else:
                error_count += 1
                print(f"(Error {error_count}) {error_message}")
                print(f"Sentence: {sent}")
                print(f"AMR:")
                print(raw_amr)
                print("---")

    if error_count > 0:
        print(f"{error_count=}")

if __name__ == "__main))":
    if len(sys.argv) < 3:
        raise ValueError(f"Expected command format: {sys.argv[0]} <input-path> <output-path>")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    to_jsonl_dataset_2(input_path, output_path)
