from symengine import sympify


def push(obj, l, depth):
    """Helper function for parsing a string into XADD"""
    while depth:
        l = l[-1]
        depth -= 1

    l.append(obj)


def parse_xadd_grammar(s):
    """Helper function for parsing a string into XADD"""
    groups = []
    depth = 0
    symLst = []
    s = s.strip()

    try:
        i = 0
        while i < len(s):
            if s[i] == '[':
                # Start of an expression.
                expression_start = i
                i += 1
                # Find the corresponding closing bracket.
                while i < len(s) and s[i] != ']':
                    i += 1
                if i >= len(s):
                    raise ValueError('Closing bracket not found')
                # Extrat the whole expression.
                s_chunk = s[expression_start: i + 1]    # Include the closing bracket.
                push(s_chunk, groups, depth)
                push(sympify(s_chunk.strip('[]')), symLst, depth)
                i += 1  # Move past the closing bracket.

            elif s[i] == '(':
                # Structural parenthesis.
                push([], groups, depth)
                push([], symLst, depth)
                depth += 1
                i += 1
            elif s[i] == ')':
                # Closing structural parenthesis.
                depth -= 1
                i += 1
            else:
                # Increment to avoid infinite loop.
                i += 1

                # idx_next_open = s.find('(', i, -1)
                # idx_next_close = s.find(')', i)
                # if idx_next_open < 0 and idx_next_close < 0:
                #     break
                # elif idx_next_open < 0 and idx_next_close > 0:
                #     idx_next_paren = idx_next_close
                # elif idx_next_open > 0 and idx_next_close < 0:
                #     raise ValueError
                # elif idx_next_open > 0 and idx_next_close > 0:
                #     idx_next_paren = min(idx_next_open, idx_next_close)

                # s_chunk = s[i: idx_next_paren]
                # i += len(s_chunk)
                # s_chunk = s_chunk.strip()
                # if s_chunk:
                #     push(s_chunk, groups, depth)
                #     push(sympify(s_chunk.strip('([])')), symLst, depth)

    except IndexError:
        raise ValueError('Parentheses mismatch')

    if depth > 0:
        raise ValueError('Parentheses mismatch')
    else:
        return groups, symLst
